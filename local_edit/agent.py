"""The orchestrator: one action per model turn, executed through a review gate.

Two modes share the same step machinery:
  - chat_turn(): a single user message -> a single action, for interactive chat.
  - work(): an autonomous loop that plans a todo queue and works through it,
    one todo at a time, bounded by max_steps per todo so a confused small
    model can't loop forever.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from rich.console import Console

from . import files, shell, ui, validate
from .client import ContextTooLargeError, JanClient, JanConnectionError
from .protocol import (
    FORMAT_REMINDER,
    Action,
    ParseError,
    build_system_prompt,
    parse_action,
    parse_todo_adds,
    strip_hallucinated_continuation,
)
from .review import (
    REVIEW_MAX_TOKENS,
    ask_human,
    build_review_messages,
    parse_review_verdict,
    render_diff,
)
from .review import snippet as review_snippet
from .state import STATUS_BLOCKED, STATUS_DONE, STATUS_IN_PROGRESS, Session

MAX_HISTORY_MESSAGES = 20
DEFAULT_MAX_STEPS = 12

# Local models run with small windows - and llama.cpp's --parallel N divides
# the context between slots, so a server started with --ctx-size 8192
# --parallel 2 gives each request only 4096. Budget conservatively and learn
# the true figure from the server the first time it complains.
DEFAULT_CONTEXT_SIZE = 4096
CHARS_PER_TOKEN = 4  # rough but stable enough for budgeting
CONTEXT_SAFETY_TOKENS = 192  # chat-template overhead, BOS tokens, rounding
MIN_HISTORY_CHARS = 800  # always leave room for the current request


def _signature(text: str) -> str:
    """Normalized form of a raw reply, used to detect the model repeating itself verbatim."""
    return " ".join(text.strip().lower().split())


class ReviewAborted(Exception):
    """Raised when the human ends the review session mid-loop."""


class Agent:
    def __init__(
        self,
        client: JanClient,
        root: Path,
        session: Session,
        console: Console,
        auto_apply: bool = False,
        self_review_enabled: bool = True,
        max_steps: int = DEFAULT_MAX_STEPS,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        fast_model: str | None = None,
        allow_run: bool = True,
        auto_run_safe: bool = True,
        command_timeout: int = shell.DEFAULT_TIMEOUT,
    ):
        self.client = client
        self.root = root
        self.session = session
        self.console = console
        self.auto_apply = auto_apply
        self.self_review_enabled = self_review_enabled
        self.max_steps = max_steps
        self.context_size = context_size
        # The capable model drives the main loop (it writes the code); cheap
        # phases - planning and diff review - go to fast_model when given.
        self.fast_model = fast_model
        self.allow_run = allow_run
        self.auto_run_safe = auto_run_safe
        self.command_timeout = command_timeout
        # Edits a review has already objected to once; a second, identical
        # attempt is applied rather than deadlocking on a wrong verdict.
        self._flagged_edits: set[str] = set()
        # Rejection counters, surfaced via /status - a run where most edits
        # get bounced by validation or review is a strong signal the chosen
        # model isn't up to this task, and that's easy to miss turn-by-turn.
        self.validation_failures = 0
        self.review_rejections = 0
        self.edits_applied = 0

    @property
    def code_model(self) -> str:
        return self.client.model

    # -- model I/O -----------------------------------------------------

    def _history_budget_chars(self, system_prompt: str) -> int:
        """How much history fits, given the context window and the space the
        reply itself needs. Everything the server must hold at once - system
        prompt, history, and the tokens it is about to generate - has to fit
        inside context_size."""
        system_tokens = len(system_prompt) // CHARS_PER_TOKEN
        reply_tokens = getattr(self.client, "max_tokens", 512)
        reserved = system_tokens + reply_tokens + CONTEXT_SAFETY_TOKENS
        available = (self.context_size - reserved) * CHARS_PER_TOKEN
        return max(MIN_HISTORY_CHARS, available)

    def _messages(self) -> list[dict]:
        system_prompt = build_system_prompt(str(self.root), allow_run=self.allow_run)
        system = {"role": "system", "content": system_prompt}
        candidates = self.session.history[-MAX_HISTORY_MESSAGES:]
        kept: list[dict] = []
        budget = self._history_budget_chars(system_prompt)
        for msg in reversed(candidates):
            cost = len(msg.get("content", ""))
            if kept and cost > budget:
                break
            kept.append(msg)
            budget -= cost
        kept.reverse()

        # Even a single message can exceed the window (a big READ, a long
        # command output). Truncate the middle of the oldest kept message
        # rather than letting the request fail outright.
        if kept:
            limit = self._history_budget_chars(system_prompt)
            overflow = sum(len(m.get("content", "")) for m in kept) - limit
            if overflow > 0:
                first = dict(kept[0])
                content = first.get("content", "")
                keep_each = max(200, (len(content) - overflow) // 2)
                if len(content) > keep_each * 2:
                    first["content"] = (
                        content[:keep_each] + "\n... (trimmed to fit the context window) ...\n"
                        + content[-keep_each:]
                    )
                    kept[0] = first
        # Trimming a strictly-alternating user/assistant history from the
        # front can leave an orphaned "assistant" reply at the start (its
        # "user" prompt got cut off). Some chat templates (Gemma's, notably)
        # hard-require the conversation to open with a user turn, so drop
        # any such orphan rather than sending a malformed request.
        while kept and kept[0]["role"] != "user":
            kept.pop(0)
        return [system] + kept

    # A small/degenerate model streaming the same line over and over is a
    # common enough failure mode (especially on abliterated/heavily quantized
    # models) that it's worth cutting the connection early rather than
    # waiting out the full max_tokens budget and then blowing the context
    # window with garbage.
    REPEAT_LINE_LIMIT = 5
    REPEAT_LINE_MIN_LEN = 8

    def _stream_and_print(
        self,
        messages: list[dict],
        label: str = "model",
        max_tokens: int | None = None,
        model: str | None = None,
        waiting_message: str = "thinking",
    ) -> str:
        """Stream a completion, printing it live and cutting it off early if
        it gets stuck repeating a line.

        A spinner covers the dead air before the first token (which on a
        local model can be many seconds of prompt processing), then stops the
        instant real output arrives so it never fights with the stream.
        """
        pieces: list[str] = []
        line_counts: dict[str, int] = {}
        buffer_line = ""
        aborted = False
        started = False
        which = model or self.client.model
        spinner = ui.status(self.console, f"{waiting_message} ({which})")
        spinner_cm = spinner.__enter__()
        try:
            for piece in self.client.stream_chat(messages, max_tokens=max_tokens, model=model):
                if not started:
                    spinner.__exit__(None, None, None)
                    spinner_cm = None
                    started = True
                    self.console.print(f"[bold cyan]{label}>[/bold cyan] ", end="")
                pieces.append(piece)
                self.console.print(piece, end="", markup=False, highlight=False)
                buffer_line += piece
                while "\n" in buffer_line:
                    line, buffer_line = buffer_line.split("\n", 1)
                    norm = line.strip().lower()
                    if len(norm) >= self.REPEAT_LINE_MIN_LEN:
                        line_counts[norm] = line_counts.get(norm, 0) + 1
                        if line_counts[norm] >= self.REPEAT_LINE_LIMIT:
                            aborted = True
                            break
                if aborted:
                    break
        finally:
            if spinner_cm is not None or not started:
                # never started streaming (empty reply or an error) - make
                # sure the spinner is torn down exactly once
                with contextlib.suppress(Exception):
                    spinner.__exit__(None, None, None)
            if started:
                self.console.print()
        if aborted:
            self.console.print("[yellow](cut off: model got stuck repeating a line)[/yellow]")
        return "".join(pieces).strip()

    def _adapt_to_context(self, exc: ContextTooLargeError) -> bool:
        """Learn the model's real window from the server's complaint.

        Returns True if something changed and a retry is worth attempting.
        """
        changed = False
        if exc.n_ctx and exc.n_ctx != self.context_size:
            self.console.print(
                f"[yellow]context window is {exc.n_ctx} tokens, not {self.context_size} - "
                f"trimming history to fit[/yellow]"
            )
            self.context_size = exc.n_ctx
            changed = True
        # Reserving over a third of a small window for the reply leaves no
        # room for the conversation; claw some back before giving up.
        ceiling = max(256, self.context_size // 3)
        if self.client.max_tokens > ceiling:
            self.console.print(f"[yellow]reducing max reply length to {ceiling} tokens[/yellow]")
            self.client.max_tokens = ceiling
            changed = True
        if not changed and len(self.session.history) > 2:
            # Nothing left to tune - drop the oldest exchange and try again.
            del self.session.history[:2]
            self.console.print("[yellow]dropping the oldest exchange to fit the context[/yellow]")
            changed = True
        return changed

    def _model_turn(self) -> str:
        for attempt in range(3):
            try:
                return self._stream_and_print(self._messages(), label="model", waiting_message="thinking")
            except ContextTooLargeError as exc:
                if attempt == 2 or not self._adapt_to_context(exc):
                    raise
        raise AssertionError("unreachable")

    def _run_self_review(self, task_text: str, rel: str, before: str, after: str) -> tuple[bool, str]:
        try:
            reply = self._stream_and_print(
                build_review_messages(task_text, rel, before, after),
                label="review",
                max_tokens=REVIEW_MAX_TOKENS,
                model=self.fast_model,
                waiting_message="reviewing the change",
            )
        except Exception as exc:  # noqa: BLE001 - self-review is advisory, never fatal
            return True, f"(self-review unavailable: {exc})"
        return parse_review_verdict(reply)

    # -- single step -----------------------------------------------------

    def step(self, active_todo=None) -> tuple[bool, str, str]:
        """Run one model turn and execute whatever it asked for.

        Returns (finished, action_kind, raw_text). `finished` means the
        current task (chat turn, or active todo) should stop advancing.
        """
        streamed = self._model_turn()
        # Drop any fabricated "Result:/You:" continuation before it becomes
        # history - otherwise the model reads its own invented transcript
        # next turn and doubles down on pretending the work is done.
        raw = strip_hallucinated_continuation(streamed)
        if raw != streamed:
            self.console.print("[yellow](ignored invented output after the command)[/yellow]")
        if not raw.strip():
            self.session.add_message("assistant", streamed[:500] or "(empty)")
            self.session.add_message(
                "user",
                "You wrote out imagined results instead of a command. Send exactly ONE command "
                "and stop - the real result will be given to you.",
            )
            return False, "parse_error", streamed

        self.session.add_message("assistant", raw)

        try:
            action = parse_action(raw)
        except ParseError as exc:
            msg = FORMAT_REMINDER.format(error=str(exc))
            self.session.add_message("user", msg)
            self.console.print(f"[red]parse error:[/red] {exc}")
            return False, "parse_error", raw

        # The reply was already streamed to the terminal as it arrived, so
        # neither the preamble nor a SAY body gets echoed a second time.

        result_text, finished = self._handle_action(action, active_todo)
        if result_text is not None:
            self.session.add_message("user", result_text)
        return finished, action.kind, raw

    def _handle_action(self, action: Action, active_todo) -> tuple[str | None, bool]:
        kind = action.kind

        if kind == "SAY":
            if "```" in action.text:
                # A small model that meant to write/change a file often just
                # narrates and dumps the code in chat instead of using a real
                # file command. That code never gets written anywhere, so
                # push it to redo the same content as an actual INSERT/EDIT.
                return (
                    "That message contained a code block, but SAY doesn't write anything to "
                    "disk. If you meant to create or change a file, resend that code as "
                    "INSERT <path> <line> (new file or new lines) or EDIT <path> <start>-<end> "
                    "(existing lines), with the code in a fenced block.",
                    False,
                )
            if active_todo is not None:
                return (
                    "Continue working on the active task. Use READ/GREP to inspect code, "
                    "EDIT/INSERT/DELETE to change it, or reply DONE if it's finished.",
                    False,
                )
            return None, True

        if kind == "DONE":
            return None, True

        if kind == "LIST":
            try:
                out = files.list_tree(self.root, action.path or ".")
            except files.PathError as exc:
                return f"ERROR: {exc}", False
            self.console.print(out)
            return f"[LIST {action.path}]\n{out}", False

        if kind == "READ":
            try:
                out = files.read_lines(self.root, action.path, action.start, action.end)
            except files.PathError as exc:
                return f"ERROR: {exc}", False
            self.console.print(out)
            return out, False

        if kind == "GREP":
            try:
                out = files.grep(self.root, action.pattern, action.path)
            except files.PathError as exc:
                return f"ERROR: {exc}", False
            self.console.print(out)
            return f"[GREP '{action.pattern}']\n{out}", False

        if kind == "RUN":
            return self._handle_run(action)

        if kind == "TODO_ADD":
            item = self.session.todos.add(action.text)
            if item is None:
                return "ERROR: the todo queue is full. Finish or drop existing todos first.", False
            self.console.print(f"[yellow]+ todo #{item.id}: {item.text}[/yellow]")
            return f"Added TODO #{item.id}.", False

        if kind == "TODO_DONE":
            item = self.session.todos.mark(action.todo_id, STATUS_DONE)
            if not item:
                return f"ERROR: no todo #{action.todo_id}", False
            self.console.print(f"[yellow]x todo #{item.id} done[/yellow]")
            finished = bool(active_todo) and active_todo.id == action.todo_id
            return f"TODO #{item.id} marked done.", finished

        if kind == "TODO_LIST":
            out = self.session.todos.render()
            self.console.print(out)
            return out, False

        if kind in ("EDIT", "INSERT", "DELETE"):
            return self._handle_write(action, active_todo)

        return f"ERROR: unhandled command {kind}", False

    # -- edits: plan -> review -> apply -----------------------------------------------------

    def _handle_write(self, action: Action, active_todo) -> tuple[str | None, bool]:
        try:
            if action.kind == "EDIT":
                if action.payload_truncated:
                    return (
                        "ERROR: your replacement block was cut off (no closing ```). "
                        "Send a shorter EDIT.",
                        False,
                    )
                result = files.plan_replace(self.root, action.path, action.start, action.end, action.payload)
            elif action.kind == "INSERT":
                if action.payload_truncated:
                    return (
                        "ERROR: your inserted block was cut off (no closing ```). "
                        "Send a shorter INSERT.",
                        False,
                    )
                result = files.plan_insert(self.root, action.path, action.start, action.payload)
            else:
                result = files.plan_delete(self.root, action.path, action.start, action.end)
        except files.PathError as exc:
            return f"ERROR: {exc}", False

        # Structural check before anything else: a syntactically broken edit
        # is never worth a review round-trip or the user's attention, and it
        # gives the model a precise error to fix instead of silent breakage.
        check = validate.validate(result.rel, result.new_text)
        if not check.ok:
            self.validation_failures += 1
            self.console.print(f"[bold red]rejected (would break the file):[/bold red] {check.message}")
            return (
                f"ERROR: that {action.kind} would leave {result.rel} invalid - {check.message}. "
                f"The file was NOT changed. READ the relevant lines again and send a corrected edit.",
                False,
            )

        title = f"{action.kind} {result.rel}"
        render_diff(self.console, result.diff, title)

        task_text = active_todo.text if active_todo else "(ad hoc chat edit)"
        review_note = ""
        flagged = False
        edit_signature = f"{result.rel}:{action.start}-{action.end}:{action.payload or ''}"
        if self.self_review_enabled:
            span_end = action.end or action.start or 1
            ok, reason = self._run_self_review(
                task_text,
                result.rel,
                review_snippet(result.old_text, action.start or 1, span_end),
                review_snippet(result.new_text, action.start or 1, span_end),
            )
            flagged = not ok
            review_note = f" self-review: {reason.strip()[:300]}"
            verdict = "OK" if ok else "FLAGGED"
            style = "dim" if ok else "bold red"
            self.console.print(f"[{style}]self-review verdict -> {verdict}[/{style}]")

        if flagged:
            # Bounce the change back once with the objection attached, so the
            # model can fix a genuinely bad edit - this holds even under
            # --auto-apply, whose promise is "don't ask me", not "apply things
            # known to be broken".
            #
            # But a small reviewer also produces false negatives (it will
            # happily misread a correct fix). If the model reconsiders and
            # sends the identical edit anyway, we stop arguing and apply it:
            # otherwise one stubborn wrong verdict deadlocks the session and
            # nothing can ever be written.
            if edit_signature not in self._flagged_edits:
                self._flagged_edits.add(edit_signature)
                self.review_rejections += 1
                self.console.print("[bold red]not applied:[/bold red] self-review rejected this change")
                return (
                    f"Your {action.kind} to {result.rel} was REJECTED by review and NOT applied.\n"
                    f"Reviewer said: {reason.strip()[:400]}\n"
                    f"If the reviewer is right, re-READ the lines and send a corrected edit. "
                    f"If you are confident the change is correct, send exactly the same edit again.",
                    False,
                )
            self.console.print(
                "[yellow]review still objects, but the edit is unchanged on retry - applying it.[/yellow]"
            )

        if self.auto_apply:
            decision = "apply"
        else:
            decision = ask_human(self.console, f"Apply {action.kind} to {result.rel}?")

        if decision == "quit":
            raise ReviewAborted()
        if decision == "reject":
            return (
                f"User rejected the {action.kind} to {result.rel}.{review_note} "
                "Try a different, smaller change.",
                False,
            )

        files.write(self.root, result.rel, result.new_text)
        self.session.backup_and_record(result.rel, result.old_text, result.new_text, result.is_new_file)
        self.session.save()
        self.edits_applied += 1
        self.console.print(f"[bold green]applied[/bold green] {action.kind} -> {result.rel}")
        return f"Applied. {result.rel} now reflects the change.{review_note}", False

    # -- shell commands -----------------------------------------------------

    def _handle_run(self, action: Action) -> tuple[str | None, bool]:
        command = (action.text or "").strip()
        if not self.allow_run:
            return "ERROR: running commands is disabled. Use READ/GREP to inspect files instead.", False

        check = shell.classify(command, root=self.root)
        if check.verdict == shell.BLOCKED:
            self.console.print(f"[bold red]refused:[/bold red] {command}  [dim]({check.reason})[/dim]")
            return (
                f"ERROR: refused to run '{command}' - {check.reason}. "
                f"That command will never be allowed; accomplish the task another way.",
                False,
            )

        # A command that touches a path outside the project is never
        # auto-run, --auto-apply included: the project root is the boundary
        # the user agreed to when they picked it, not something a flag about
        # trusting *edits* should silently override.
        needs_ok = check.outside_project or (
            check.verdict == shell.NEEDS_APPROVAL and not self.auto_apply
        )
        if check.verdict == shell.SAFE and not self.auto_run_safe and not self.auto_apply:
            needs_ok = True

        if needs_ok:
            style = "bold red" if check.outside_project else "bold"
            self.console.print(f"[{style}]command:[/{style}] [cyan]{command}[/cyan]  [dim]({check.reason})[/dim]")
            decision = ui.confirm(
                self.console,
                "Run it?",
                default="no" if check.outside_project else "yes",
            )
            if decision == "stop":
                raise ReviewAborted()
            if decision == "no":
                return (
                    f"User declined to run '{command}'. Continue without it, or try a different approach.",
                    False,
                )
        else:
            self.console.print(f"[bold]running:[/bold] [cyan]{command}[/cyan]")

        with ui.status(self.console, f"running: {command}"):
            outcome = shell.run(command, cwd=self.root, timeout=self.command_timeout)

        status_word = "ok" if outcome.ok else f"exit {outcome.exit_code}"
        colour = "green" if outcome.ok else "red"
        self.console.print(f"[{colour}]-> {status_word}[/{colour}]")
        if outcome.output.strip():
            self.console.print(outcome.output, markup=False, highlight=False)

        body = outcome.output.strip() or "(no output)"
        return f"$ {command}\n(exit code {outcome.exit_code})\n{body}", False

    # -- chat mode -----------------------------------------------------

    def chat_turn(self, user_text: str) -> None:
        # add_message() may merge into the existing last entry rather than
        # appending (see Session.add_message), so on failure we can't just
        # blindly pop - snapshot enough to restore either outcome cleanly.
        prev_len = len(self.session.history)
        prev_last_content = self.session.history[-1]["content"] if self.session.history else None
        self.session.add_message("user", user_text)

        # A single chat message can need several tool calls before there's
        # anything to say back (e.g. "recreate Pong" starts with LIST/READ).
        # Keep stepping - same as the /work loop, minus a todo object -
        # until the model actually replies (SAY/DONE) or we hit a cap.
        consecutive_stalls = 0
        last_signature: str | None = None
        steps_done = 0
        try:
            for _ in range(self.max_steps):
                finished, kind, raw = self.step()
                steps_done += 1
                self.session.save()
                if finished:
                    return

                signature = _signature(raw)
                is_repeat = signature and signature == last_signature
                last_signature = signature
                # A plain SAY already ended the turn above, so a SAY reaching
                # here is the "you put code in a chat message" case - it made
                # no progress, and a model that keeps doing it must not spin
                # until max_steps.
                if kind in ("parse_error", "SAY") or is_repeat:
                    consecutive_stalls += 1
                    if consecutive_stalls >= 3 or (is_repeat and consecutive_stalls >= 2):
                        self.console.print("[yellow](stopping: got stuck in a loop)[/yellow]")
                        return
                    continue
                consecutive_stalls = 0
            self.console.print(
                f"[yellow](stopped after {self.max_steps} actions without a final reply - "
                f"ask it to continue, or use /work for longer tasks)[/yellow]"
            )
        except JanConnectionError as exc:
            self.console.print(f"[bold red]{exc}[/bold red]")
            if steps_done == 0:
                # nothing from this call succeeded yet; undo just the dangling user turn
                if len(self.session.history) > prev_len:
                    self.session.history.pop()  # a new entry was appended; drop it
                elif self.session.history and prev_last_content is not None:
                    self.session.history[-1]["content"] = prev_last_content  # merged; restore prior content
        finally:
            self.session.save()

    # -- autonomous work loop -----------------------------------------------------

    def plan(self, goal: str, attempts: int = 3) -> None:
        prompt = (
            f"Goal: {goal}\n"
            "Break this into 3-7 concrete todo items using only `TODO ADD <text>` lines, "
            "one per line, most important first. No other commands or explanation."
        )
        self.session.add_message("user", prompt)
        last_signature: str | None = None
        for _ in range(attempts):
            # Planning is a cheap phase: routed to the fast model when one is
            # configured, so the capable model is reserved for writing code.
            raw = self._stream_and_print(
                self._messages(), label="plan", model=self.fast_model, waiting_message="planning"
            )
            self.session.add_message("assistant", raw)
            items = parse_todo_adds(raw)
            if items:
                for text in items:
                    item = self.session.todos.add(text)
                    if item is None:
                        break
                    self.console.print(f"[yellow]+ todo #{item.id}: {item.text}[/yellow]")
                self.session.save()
                return

            signature = _signature(raw)
            if signature and signature == last_signature:
                self.console.print("[yellow]model repeated itself while planning, giving up early[/yellow]")
                break
            last_signature = signature

            # It's reasonable for the model to want to look around before
            # committing to a breakdown. Let it, then ask again.
            try:
                action = parse_action(raw)
            except ParseError:
                action = None
            if action and action.kind in ("READ", "LIST", "GREP"):
                result_text, _ = self._handle_action(action, None)
                nudge = (result_text or "") + (
                    "\n\nNow reply with ONLY `TODO ADD <text>` lines (3-7 items), based on what "
                    "you just saw. No other commands."
                )
            else:
                nudge = "Reply with ONLY `TODO ADD <text>` lines, one per line, 3-7 items. No other text."
            self.session.add_message("user", nudge)

        # Gave up getting a breakdown; fall back to one todo for the whole goal
        # rather than leaving /work with an empty, silently-do-nothing queue.
        item = self.session.todos.add(goal)
        if item:
            self.console.print(
                f"[yellow]model never produced a todo breakdown; queued the goal itself as #{item.id}[/yellow]"
            )
        self.session.save()

    def run_todo(self, todo) -> None:
        self.session.todos.mark(todo.id, STATUS_IN_PROGRESS)
        self.console.print(f"\n[bold]--- working on #{todo.id}: {todo.text} ---[/bold]")
        prompt = (
            f"Active task (TODO #{todo.id}): {todo.text}\n"
            f"Work only on this task. READ the relevant file(s) before you EDIT/INSERT/DELETE. "
            f"When it's fully done, reply with TODO DONE {todo.id}."
        )
        self.session.add_message("user", prompt)
        consecutive_stalls = 0
        last_signature: str | None = None
        for _ in range(self.max_steps):
            finished, kind, raw = self.step(active_todo=todo)
            self.session.save()
            if finished:
                if self.session.todos.get(todo.id).status == STATUS_IN_PROGRESS:
                    self.session.todos.mark(todo.id, STATUS_DONE)
                self.console.print(f"[bold green]--- #{todo.id} done ---[/bold green]")
                return

            signature = _signature(raw)
            is_repeat = signature and signature == last_signature
            last_signature = signature

            # TODO_ADD doesn't advance the active task ("work only on this
            # task" is explicit in the prompt) - a model that spams it here
            # instead of READ/EDIT/DONE is off the rails, not making
            # progress, and left unchecked can grow the queue faster than
            # it drains, running /work forever. Count it as a stall too.
            if kind in ("parse_error", "SAY", "TODO_ADD") or is_repeat:
                consecutive_stalls += 1
                if consecutive_stalls >= 3 or (is_repeat and consecutive_stalls >= 2):
                    if is_repeat:
                        note = "model got stuck repeating the same reply"
                    elif kind == "parse_error":
                        note = "repeated format errors"
                    elif kind == "TODO_ADD":
                        note = "model kept adding todos instead of working the active one"
                    else:
                        note = "model stalled without acting"
                    self.session.todos.mark(todo.id, STATUS_BLOCKED, note=note)
                    self.console.print(f"[red]todo #{todo.id} blocked: {note}[/red]")
                    return
                continue
            consecutive_stalls = 0
        self.session.todos.mark(todo.id, STATUS_BLOCKED, note="max steps reached")
        self.console.print(f"[red]todo #{todo.id} blocked: ran out of steps ({self.max_steps})[/red]")

    # Absolute circuit breaker on a single /work call, independent of queue
    # size or per-todo step limits: caps total wall-clock/compute burned on
    # one autonomous run no matter what pathological loop a bad model finds.
    MAX_TODOS_PER_WORK_CALL = 40

    def work(self, goal: str | None = None) -> None:
        processed = 0
        try:
            if goal:
                self.plan(goal)
            while True:
                if processed >= self.MAX_TODOS_PER_WORK_CALL:
                    self.console.print(
                        f"[red]stopping: processed {processed} todos in this /work call "
                        f"(limit {self.MAX_TODOS_PER_WORK_CALL}) - the model may be stuck in a loop. "
                        f"Check /todo and run /work again to continue.[/red]"
                    )
                    break
                todo = self.session.todos.next_pending()
                if not todo:
                    break
                self.run_todo(todo)
                processed += 1
        except JanConnectionError as exc:
            self.console.print(f"[bold red]{exc}[/bold red]")
        except ReviewAborted:
            self.console.print("[yellow]stopped by user[/yellow]")
        finally:
            self.session.save()
        if not self.session.todos.next_pending():
            self.console.print("[bold green]queue empty[/bold green]")

"""The orchestrator: one action per model turn, executed through a review gate.

Two modes share the same step machinery:
  - chat_turn(): a single user message -> a single action, for interactive chat.
  - work(): an autonomous loop that plans a todo queue and works through it,
    one todo at a time, bounded by max_steps per todo so a confused small
    model can't loop forever.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from rich.console import Console

from . import files, focus, shell, ui, validate
from .client import ContextTooLargeError, JanClient, JanConnectionError
from .progress import RECENT_UNTRIMMED, Progress, trim_old_message
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

MAX_HISTORY_MESSAGES = 16
DEFAULT_MAX_STEPS = 12

# Local models run with small windows - and llama.cpp's --parallel N divides
# the context between slots, so a server started with --ctx-size 8192
# --parallel 2 gives each request only 4096. Budget conservatively and learn
# the true figure from the server the first time it complains.
DEFAULT_CONTEXT_SIZE = 4096
CHARS_PER_TOKEN = 4  # rough but stable enough for budgeting
CONTEXT_SAFETY_TOKENS = 192  # chat-template overhead, BOS tokens, rounding
MIN_HISTORY_CHARS = 800  # always leave room for the current request


_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

MAX_CONTINUATIONS = 4
MAX_DONE_CHECKS = 2  # per request  # one reply can grow to (1 + this) x max_tokens


def _drop_echo(pieces, prefix: str):
    """Yield streamed pieces minus a leading copy of `prefix`, if any.

    Servers differ on prefill: llama.cpp's own server streams only the new
    text, Jan streams the prefill back first. Buffer until it's clear which
    this is, then pass everything after the echo through untouched.
    """
    if not prefix:
        yield from pieces
        return
    buf = ""
    matching = True
    for piece in pieces:
        if not matching:
            yield piece
            continue
        buf += piece
        if len(buf) < len(prefix):
            if prefix.startswith(buf):
                continue
            matching = False
            yield buf
        elif buf.startswith(prefix):
            matching = False
            if buf[len(prefix):]:
                yield buf[len(prefix):]
        else:
            matching = False
            yield buf
    # A stream that ended while still matching was only the echo - nothing new.


def _step_label(action: Action) -> str:
    """A few words naming what an action tried, for the progress note."""
    kind, path = action.kind, action.path or ""
    if kind in ("EDIT", "DELETE"):
        return f"{kind} {path} {action.start}-{action.end or action.start}"
    if kind == "INSERT":
        return f"INSERT {path} after line {action.start}"
    if kind == "READ":
        return f"READ {path}" + (f" {action.start}-{action.end}" if action.start else "")
    if kind == "RUN":
        command = action.text or ""
        return f"RUN {command[:50]}"
    if kind == "GREP":
        return f"GREP '{action.pattern}' in {path}"
    if kind == "SAY":
        return "chat message"
    if kind == "FIX":
        return f"FIX {path}: {(action.text or '')[:60]}"
    return f"{kind} {path}".strip()


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
        self._last_stream_aborted = False
        # Code the model wrote in a chat message instead of a file command,
        # waiting for it to say where it goes (see _hold_code_from_chat).
        self._held_code: str | None = None
        # What the model is working on and how it's going, shown every turn.
        self.progress = Progress()

    @property
    def code_model(self) -> str:
        return self.client.model

    # -- model I/O -----------------------------------------------------

    def _history_budget_chars(self, system_prompt: str, reserve_chars: int = 0) -> int:
        """How much history fits, given the context window and the space the
        reply itself needs. Everything the server must hold at once - system
        prompt, history, and the tokens it is about to generate - has to fit
        inside context_size. `reserve_chars` holds back room for a partial
        reply being sent back for continuation."""
        system_tokens = len(system_prompt) // CHARS_PER_TOKEN
        reply_tokens = getattr(self.client, "max_tokens", 512)
        reserved = system_tokens + reply_tokens + CONTEXT_SAFETY_TOKENS
        available = (self.context_size - reserved) * CHARS_PER_TOKEN - reserve_chars
        return max(MIN_HISTORY_CHARS, available)

    def _messages(self, reserve_chars: int = 0) -> list[dict]:
        system_prompt = build_system_prompt(str(self.root), allow_run=self.allow_run)
        system = {"role": "system", "content": system_prompt}
        candidates = self.session.history[-MAX_HISTORY_MESSAGES:]
        # Older turns lose their bulk - the progress note carries what they
        # did - so the window holds more turns and the model reads less.
        cut = len(candidates) - RECENT_UNTRIMMED
        candidates = [trim_old_message(m) if i < cut else m for i, m in enumerate(candidates)]
        note = self.progress.render()
        reserve_chars += len(note) + 2 if note else 0
        kept: list[dict] = []
        budget = self._history_budget_chars(system_prompt, reserve_chars)
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
            limit = self._history_budget_chars(system_prompt, reserve_chars)
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
        # The note goes at the end of the newest message, not in the system
        # prompt: llama.cpp reuses its cache for an unchanged prompt prefix,
        # and a note that changes every turn would invalidate all of it.
        if note and kept and kept[-1]["role"] == "user":
            kept[-1] = {**kept[-1], "content": kept[-1]["content"] + "\n\n" + note}
        return [system] + kept

    # A small/degenerate model streaming the same line over and over is a
    # common enough failure mode (especially on abliterated/heavily quantized
    # models) that it's worth cutting the connection early rather than
    # waiting out the full max_tokens budget and then blowing the context
    # window with garbage.
    # Real code repeats lines legitimately - five product cards means five
    # identical `<div class="card">` lines - so only a run of the same line
    # back to back, or one line recurring far more than code ever does, counts.
    REPEAT_RUN_LIMIT = 5
    REPEAT_TOTAL_LIMIT = 30
    REPEAT_LINE_MIN_LEN = 8

    def _stream_and_print(
        self,
        messages: list[dict],
        label: str | None = "model",
        max_tokens: int | None = None,
        model: str | None = None,
        waiting_message: str = "thinking",
        echo_prefix: str = "",
        strip: bool = True,
    ) -> str:
        """Stream a completion, printing it live and cutting it off early if
        it gets stuck repeating a line.

        A spinner covers the dead air before the first token (which on a
        local model can be many seconds of prompt processing), then stops the
        instant real output arrives so it never fights with the stream.

        `echo_prefix` is a prefilled assistant message some servers (Jan
        among them) stream back before the continuation; it is dropped so
        only new text is printed and returned.
        """
        pieces: list[str] = []
        line_counts: dict[str, int] = {}
        last_line = None
        run_length = 0
        buffer_line = ""
        aborted = False
        started = False
        which = model or self.client.model
        spinner = ui.status(self.console, f"{waiting_message} ({which})")
        spinner_cm = spinner.__enter__()
        try:
            stream = self.client.stream_chat(messages, max_tokens=max_tokens, model=model)
            for piece in _drop_echo(stream, echo_prefix):
                if not started:
                    spinner.__exit__(None, None, None)
                    spinner_cm = None
                    started = True
                    if label:
                        self.console.print(f"[bold cyan]{label}>[/bold cyan] ", end="")
                pieces.append(piece)
                self.console.print(piece, end="", markup=False, highlight=False)
                buffer_line += piece
                while "\n" in buffer_line:
                    line, buffer_line = buffer_line.split("\n", 1)
                    norm = line.strip().lower()
                    if len(norm) < self.REPEAT_LINE_MIN_LEN:
                        continue
                    run_length = run_length + 1 if norm == last_line else 1
                    last_line = norm
                    line_counts[norm] = line_counts.get(norm, 0) + 1
                    if run_length >= self.REPEAT_RUN_LIMIT or line_counts[norm] >= self.REPEAT_TOTAL_LIMIT:
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
        self._last_stream_aborted = aborted
        if aborted:
            self.console.print("[yellow](cut off: model got stuck repeating a line)[/yellow]")
        text = "".join(pieces)
        return text.strip() if strip else text

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
                return self._generate_with_continuations()
            except ContextTooLargeError as exc:
                if attempt == 2 or not self._adapt_to_context(exc):
                    raise
        raise AssertionError("unreachable")

    def _generate_with_continuations(self) -> str:
        """One model reply, stitched across the max_tokens limit if needed.

        A small reply budget routinely cuts a whole-file INSERT off mid-block.
        When that happens, the partial reply is sent back as the start of the
        assistant's own message (a prefill), so the model carries on from the
        exact character it stopped at, and the pieces are joined. Only an
        open code block is continued - a cut-off sentence of chat isn't worth
        another round-trip.
        """
        text = self._stream_and_print(
            self._messages(), label="model", waiting_message="thinking", strip=False
        )
        for _ in range(MAX_CONTINUATIONS):
            if getattr(self.client, "last_finish_reason", None) != "length":
                break
            if self._last_stream_aborted or text.count("```") % 2 == 0:
                break
            # The partial reply rides along in the prompt; stop before it
            # crowds out the conversation the model needs to finish it.
            if len(text) > self.context_size * CHARS_PER_TOKEN // 2:
                break
            self.console.print("[dim](hit the reply length limit - continuing...)[/dim]")
            messages = self._messages(reserve_chars=len(text))
            messages.append({"role": "assistant", "content": text})
            more = self._stream_and_print(
                messages, label=None, waiting_message="continuing", echo_prefix=text, strip=False
            )
            if not more.strip():
                break
            text += more
        return text.strip()

    def _check_done(self) -> str | None:
        """Ask, once per request, whether DONE is premature.

        Small models stop after the first half of "do X and Y". A closed
        question over the progress note catches that cheaply. Returns what's
        missing, or None to accept the DONE. Capped per request, so a
        checker that keeps saying NO can't trap the model.
        """
        progress = self.progress
        if not progress.goal or progress.done_checks >= MAX_DONE_CHECKS or not self.self_review_enabled:
            return None
        progress.done_checks += 1
        prompt = (
            f"Request: {progress.goal}\n\n"
            "What was done, in order:\n" + ("\n".join(progress.steps) or "(nothing)") + "\n\n"
            "Has EVERY part of the request been done? Answer YES or NO. "
            "If NO, then say in a few words which part is missing."
        )
        messages = [
            {"role": "system", "content": "You check whether a task is finished. You answer YES or NO."},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = self._stream_and_print(
                messages, label="check", max_tokens=REVIEW_MAX_TOKENS, model=self.fast_model,
                waiting_message="checking the request is finished",
            )
        except Exception:  # noqa: BLE001 - advisory, like self-review
            return None
        finished, reason = parse_review_verdict(reply)
        if finished:
            return None
        detail = re.sub(r"^\W*NO\W*", "", reason.strip(), flags=re.IGNORECASE).strip()
        return detail[:200] or "part of the request is still missing"

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
            action = self._attach_held_code(raw)
            if action is None:
                msg = FORMAT_REMINDER.format(error=str(exc))
                self.progress.record("unparseable reply", f"ERROR: {exc}")
                self.session.add_message("user", msg)
                self.console.print(f"[red]parse error:[/red] {exc}")
                return False, "parse_error", raw

        # The reply was already streamed to the terminal as it arrived, so
        # neither the preamble nor a SAY body gets echoed a second time.

        result_text, finished = self._handle_action(action, active_todo)
        self.progress.record(_step_label(action), result_text)
        if result_text is not None:
            if result_text.startswith("ERROR:"):
                # The model sees every error; the user should too, or a run of
                # failed commands just looks like the model ignoring them.
                self.console.print(f"[red]{result_text.splitlines()[0]}[/red]", markup=False)
            self.session.add_message("user", result_text)
        return finished, action.kind, raw

    def _handle_action(self, action: Action, active_todo) -> tuple[str | None, bool]:
        kind = action.kind

        if kind == "SAY":
            if "```" in action.text:
                return self._hold_code_from_chat(action), False
            if active_todo is not None:
                return (
                    "Continue working on the active task. Use READ/GREP to inspect code, "
                    "EDIT/INSERT/DELETE to change it, or reply DONE if it's finished.",
                    False,
                )
            return None, True

        if kind == "DONE":
            missing = self._check_done()
            if missing:
                return (
                    f"Not finished yet - {missing}. Continue with ONE command "
                    f"(READ the file first if you need to see it).",
                    False,
                )
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

        if kind == "FIX":
            return self._handle_fix(action, active_todo)

        if kind in ("EDIT", "INSERT", "DELETE"):
            return self._handle_write(action, active_todo)

        return f"ERROR: unhandled command {kind}", False

    # -- edits: plan -> review -> apply -----------------------------------------------------

    # -- code written as chat -----------------------------------------------------

    def _hold_code_from_chat(self, action: Action) -> str:
        """Keep the code from a chat message so it needn't be written twice.

        A small model that means to create a file often narrates and dumps
        the code in chat instead of sending INSERT/EDIT. Asking it to resend
        the code costs another full generation - and with two copies of a
        page in history, a 4k context is full and the retry fails. So hold
        the largest block here, shrink it to a placeholder in history, and
        ask only for the command line; _attach_held_code supplies the code.
        """
        blocks = list(_FENCED_BLOCK_RE.finditer(action.raw))
        largest = max(blocks, key=lambda b: len(b.group(1))) if blocks else None
        code = largest.group(1).rstrip("\n") if largest else ""
        if not code.strip():
            return (
                "That message contained a code block, but SAY doesn't write anything to disk. "
                "Use INSERT or EDIT with the code in a fenced block."
            )
        self._held_code = code
        n_lines = len(code.splitlines())
        last = self.session.history[-1] if self.session.history else None
        if last and last["role"] == "assistant":
            placeholder = f"(code block of {n_lines} lines - held, not saved yet)"
            last["content"] = last["content"].replace(largest.group(0), placeholder, 1)
        self.console.print(f"[yellow]holding the {n_lines}-line code block - asking where to save it[/yellow]")
        return (
            f"That code ({n_lines} lines) was NOT saved - SAY doesn't write files. It is being held "
            "for you, so do NOT write it again. Reply with ONLY the command line, no code block:\n"
            "  INSERT <path> 0            to save it as a new file\n"
            "  EDIT <path> <start>-<end>  to replace those lines with it\n"
            "The held code will be used as the content."
        )

    def _attach_held_code(self, raw: str) -> Action | None:
        """A bare `INSERT path 0` / `EDIT path a-b` after code was held:
        pair the command with that code instead of failing to parse."""
        if not self._held_code:
            return None
        line = raw.strip().splitlines()[0] if raw.strip() else ""
        if "```" in raw or not re.match(r"^\W*(INSERT|EDIT)\s+\S+\s+\d", line, re.IGNORECASE):
            return None
        try:
            action = parse_action(f"{line}\n```\n{self._held_code}\n```")
        except ParseError:
            return None
        self.console.print("[dim](using the held code block)[/dim]")
        self._held_code = None
        return action

    def _as_whole_file_rewrite(self, action: Action) -> Action:
        """Turn `INSERT <existing file> 0` carrying a whole new file into a
        replace of the entire file.

        A model that doesn't know the file exists writes the full page as an
        INSERT at the top, which stacks it above the old content (two
        <html> documents in one file). When the payload opens with the same
        line the file does - "<!DOCTYPE html>", "import ...", a shebang -
        it's a rewrite, so replace instead of prepend. The diff and the
        approval prompt still show exactly what happens.
        """
        if action.kind != "INSERT" or action.start != 0 or not action.payload:
            return action
        try:
            target = files.resolve(self.root, action.path)
        except files.PathError:
            return action
        if not target.is_file():
            return action
        try:
            current = target.read_text()
        except (OSError, UnicodeDecodeError):
            return action
        first_current = next((ln.strip() for ln in current.splitlines() if ln.strip()), None)
        first_new = next((ln.strip() for ln in action.payload.splitlines() if ln.strip()), None)
        if not first_current or first_current != first_new:
            return action
        line_count = len(current.splitlines())
        self.console.print(
            f"[yellow]{action.path} already exists - treating this as a rewrite of all "
            f"{line_count} lines, not an insert above them[/yellow]"
        )
        action.kind, action.start, action.end = "EDIT", 1, line_count
        return action

    # -- FIX: harness finds the code, model rewrites it -----------------------------

    def _ask(self, prompt: str, label: str, max_tokens: int | None = None, model: str | None = None) -> str:
        """One standalone question - no history, no system prompt beyond a line.

        Each FIX step is a self-contained prompt of a few hundred tokens, so
        it never competes with the conversation for the window. Code replies
        that hit the length limit mid-block are continued like main turns.
        """
        messages = [
            {"role": "system", "content": "You are a careful assistant. Answer exactly in the format asked."},
            {"role": "user", "content": prompt},
        ]
        text = self._stream_and_print(
            messages, label=label, max_tokens=max_tokens, model=model, waiting_message=label, strip=False
        )
        for _ in range(MAX_CONTINUATIONS):
            if getattr(self.client, "last_finish_reason", None) != "length" or text.count("```") % 2 == 0:
                break
            if self._last_stream_aborted:
                break
            more = self._stream_and_print(
                messages + [{"role": "assistant", "content": text}], label=None,
                waiting_message="continuing", echo_prefix=text, strip=False,
            )
            if not more.strip():
                break
            text += more
        return text.strip()

    def _ask_for_code(self, prompt: str, block) -> str | None:
        """Ask for a block of code; one retry if it comes back as the whole file."""
        for attempt in range(2):
            code = focus.extract_code(self._ask(prompt, label="write"))
            if not code or not focus.looks_like_whole_file(code, block):
                return code
            self.console.print("[yellow](that was the whole file - asking for just the part again)[/yellow]")
            prompt += "\n\nNOT the whole file. Only the lines asked for."
        return None

    def _handle_fix(self, action: Action, active_todo) -> tuple[str | None, bool]:
        try:
            target = files.resolve(self.root, action.path)
        except files.PathError as exc:
            return f"ERROR: {exc}", False
        if not target.is_file():
            return f"ERROR: no such file: {action.path} - to create it, INSERT {action.path} 0", False
        request = action.text or self.progress.goal or ""
        if not request:
            return "ERROR: FIX needs to say what to change, e.g. FIX index.html make the header green", False
        parts = focus.split_request(request)
        if len(parts) > 1:
            self.console.print(f"[dim](doing this as {len(parts)} separate changes)[/dim]")
        results: list[str] = []
        for n, part in enumerate(parts, 1):
            result_text, finished = self._fix_one(action.path, target, part, active_todo)
            results.append(result_text or "")
            if not (result_text or "").startswith("Applied"):
                if n > 1:
                    # Report what landed, so the model doesn't redo it.
                    result_text = (
                        f"{result_text}\nAlready done by this FIX: " + " ".join(results[:-1])
                    )
                return result_text, finished
        return " ".join(results), False

    def _fix_one(self, rel: str, target: Path, request: str, active_todo) -> tuple[str | None, bool]:
        blocks = focus.split_blocks(target.read_text(errors="replace"))
        if not blocks:
            return f"ERROR: {rel} is empty - INSERT {rel} 0 with its content instead", False

        # 1. which block? (closed question; fall back to the best keyword match)
        # The user's own words count too: the model's FIX text often drops
        # the name that matters ("handle empty lists" for average()).
        # Only as a fallback, though - for one part of a split request the
        # rest of the sentence ("...header green...") is noise.
        ranking_text = request
        choices = focus.candidates(blocks, request)
        if not choices and self.progress.goal:
            ranking_text = f"{request} {self.progress.goal}"
            choices = focus.candidates(blocks, ranking_text)
        choices = choices or focus.outline(blocks)
        choice: int | str | None = "NEW"
        if choices and not focus.adds_something_absent(blocks, request):
            reply = self._ask(focus.choose_prompt(request, rel, choices), label="find", max_tokens=16,
                              model=self.fast_model)
            choice = focus.parse_choice(reply, len(choices))
            if choice == "NEW" and not focus.is_adding(request):
                choice = None  # "none of the above" wasn't offered
            if choice is None:
                best = focus.candidates(blocks, ranking_text, limit=1)
                choice = 1 + choices.index(best[0]) if best and best[0] in choices else 1

        if isinstance(choice, int):
            # 2a. rewrite just that block
            block = choices[choice - 1]
            self.console.print(f"[dim](rewriting lines {block.start}-{block.end}: {block.head[:60]})[/dim]")
            code = self._ask_for_code(focus.rewrite_prompt(request, rel, block), block)
            if code is None:
                return f"ERROR: FIX {rel}: no usable rewrite of lines {block.start}-{block.end} came back", False
            if code == "":
                # An empty rewrite is the model's way of saying "remove it".
                edit = Action(kind="DELETE", path=rel, start=block.start, end=block.end)
                where = f"removed lines {block.start}-{block.end}, {block.head[:40]}"
            else:
                edit = Action(kind="EDIT", path=rel, start=block.start, end=block.end,
                              payload=focus.match_indent(code, block.indent))
                where = f"rewrote lines {block.start}-{block.end}, {block.head[:40]}"
        else:
            # 2b. new code: where does it go, then write only that
            after = focus.obvious_place(blocks, request)
            if after is None:
                spots = focus.places(blocks)
                reply = self._ask(focus.place_prompt(request, rel, spots), label="place", max_tokens=16,
                                  model=self.fast_model)
                spot = focus.parse_choice(reply, len(spots))
                after = spots[spot - 1] if isinstance(spot, int) else spots[-1]
            self.console.print(f"[dim](adding new code after lines {after.start}-{after.end}: {after.head[:60]})[/dim]")
            code = self._ask_for_code(focus.new_code_prompt(request, rel, after), None)
            code = focus.trim_stray_closers(code) if code else code
            if not code:
                return f"ERROR: FIX {rel}: no usable new code came back", False
            edit = Action(kind="INSERT", path=rel, start=after.end,
                          payload=focus.match_indent(code, after.indent))
            where = f"added {len(code.splitlines())} lines after line {after.end}, {after.head[:40]}"

        result_text, finished = self._handle_write(edit, active_todo, task_text=request, from_fix=True)
        if result_text and result_text.startswith("Applied."):
            result_text = f"Applied ({where})." + result_text[len("Applied."):]
        return result_text, finished

    def _unrequested_new_file(self, action: Action) -> str | None:
        """Push back, once, on a new file nobody asked for.

        Asked to "add a footer" to a one-page site, a small model writes
        footer.html - or src/components/Navbar.jsx - files the page never
        loads. If the request doesn't name the file and the project already
        has files, that's almost always the wrong target. Sending the same
        INSERT again goes through, so a genuinely new file is never blocked.
        """
        if action.kind != "INSERT" or action.start != 0:
            return None
        try:
            target = files.resolve(self.root, action.path)
        except files.PathError:
            return None
        if target.exists():
            return None
        request = (self.progress.goal or "") + " " + (self.progress.earlier_goal or "")
        if Path(action.path).name.lower() in request.lower():
            return None
        existing = list(self.progress.files)
        if not existing:
            return None
        same_type = [rel for rel in existing if Path(rel).suffix == target.suffix] or existing[-1:]
        signature = f"new-file:{action.path}"
        if signature in self._flagged_edits:
            return None
        self._flagged_edits.add(signature)
        self.console.print(f"[yellow]{action.path} would be a new file - checking that's intended[/yellow]")
        return (
            f"ERROR: the request didn't ask for a new file, and {action.path} would not be part of "
            f"{same_type[0]}. To change {same_type[0]}, use: FIX {same_type[0]} <what to change>. "
            f"If a separate new file really is needed, send the same INSERT again."
        )

    def _blind_line_edit(self, action: Action) -> str | None:
        """Push back, once per file and request, on line numbers never seen.

        `INSERT index.html 100` from a model that hasn't READ the file this
        request is a guess - it landed a stall card inside the CSS. FIX
        needs no line numbers, so point there instead.
        """
        if not self.progress.goal:
            return None  # no request in progress (a /work todo sets one too)
        if action.kind not in ("EDIT", "INSERT", "DELETE") or (action.kind == "INSERT" and action.start == 0):
            return None
        try:
            if not files.resolve(self.root, action.path).is_file():
                return None
        except files.PathError:
            return None
        if any(step.startswith(f"READ {action.path}") for step in self.progress.steps):
            return None
        signature = f"blind:{self.progress.goal}:{action.path}"
        if signature in self._flagged_edits:
            return None
        self._flagged_edits.add(signature)
        self.console.print(f"[yellow]{action.kind} on lines of {action.path} it hasn't read - suggesting FIX[/yellow]")
        return (
            f"ERROR: you haven't READ {action.path} for this request, so those line numbers are a guess. "
            f"Use FIX {action.path} <what to change> instead - it finds the right lines for you."
        )

    def _handle_write(
        self, action: Action, active_todo, task_text: str | None = None, from_fix: bool = False
    ) -> tuple[str | None, bool]:
        action = self._as_whole_file_rewrite(action)
        objection = self._unrequested_new_file(action) or (None if from_fix else self._blind_line_edit(action))
        if objection:
            return objection, False
        try:
            if action.kind == "EDIT":
                if action.payload_truncated:
                    self.console.print("[red]not applied:[/red] reply hit the length limit mid-block")
                    return (
                        "ERROR: your replacement block was cut off (no closing ```) - replies are "
                        f"limited to about {self.client.max_tokens} tokens. The file was NOT changed. "
                        "Do not resend the whole file: EDIT only the few lines that need to change.",
                        False,
                    )
                result = files.plan_replace(self.root, action.path, action.start, action.end, action.payload)
            elif action.kind == "INSERT":
                if action.payload_truncated:
                    self.console.print("[red]not applied:[/red] reply hit the length limit mid-block")
                    return (
                        "ERROR: your inserted block was cut off (no closing ```) - replies are "
                        f"limited to about {self.client.max_tokens} tokens. The file was NOT changed. "
                        "Write it in parts: INSERT the first part now, then INSERT the rest after "
                        "its last line in the next command. For a web page, put the CSS in its own "
                        "file (style.css) so each part stays short.",
                        False,
                    )
                result = files.plan_insert(self.root, action.path, action.start, action.payload)
            else:
                result = files.plan_delete(self.root, action.path, action.start, action.end)
        except files.PathError as exc:
            return f"ERROR: {exc}", False

        # A no-op edit has nothing to review or approve. Without this, the
        # "identical retry gets applied" escape hatch below turns a model
        # resending the current file into an endless approve-nothing loop.
        if result.old_text == result.new_text:
            self.console.print(f"[yellow]no change:[/yellow] {result.rel} already has that content")
            return (
                f"No change: {result.rel} already contains exactly that. Do not resend it. "
                f"If the task needs new content, EDIT/INSERT only the new lines.",
                False,
            )

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

        if result.rel.lower().endswith((".html", ".htm")):
            unbalanced = validate.html_balance_change(result.old_text, result.new_text)
            if unbalanced:
                self.validation_failures += 1
                self.console.print(f"[bold red]rejected (would break the page):[/bold red] {unbalanced}")
                return (
                    f"ERROR: that {action.kind} would break {result.rel} - {unbalanced}. "
                    f"The file was NOT changed.",
                    False,
                )

        title = f"{action.kind} {result.rel}"
        render_diff(self.console, result.diff, title)

        if task_text is None:
            task_text = active_todo.text if active_todo else (self.progress.goal or "(ad hoc chat edit)")
        review_note = ""
        flagged = False
        edit_signature = f"{result.rel}:{action.start}-{action.end}:{action.payload or ''}"
        # FIX edits skip the model review: FIX's own steps chose the block,
        # and validation above checks the structure - which a small reviewer
        # misjudges (it rejected a correctly placed footer as "bad nesting").
        if self.self_review_enabled and not from_fix:
            span_start = action.start or 1
            span_end = action.end or span_start
            # The AFTER window must cover the lines that were actually
            # written, not the old range - a 3-line EDIT that writes 40 lines
            # otherwise shows the reviewer a fragment it calls "cut off".
            payload_lines = len((action.payload or "").splitlines())
            if action.kind == "EDIT":
                new_start, new_end = span_start, span_start + max(payload_lines, 1) - 1
            elif action.kind == "INSERT":
                new_start, new_end = span_start + 1, span_start + max(payload_lines, 1)
            else:
                new_start, new_end = span_start, span_start
            ok, reason = self._run_self_review(
                task_text,
                result.rel,
                review_snippet(result.old_text, span_start, span_end),
                review_snippet(result.new_text, new_start, new_end),
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
        self.progress.note_file(result.rel, len(result.new_text.splitlines()))
        self.console.print(f"[bold green]applied[/bold green] {action.kind} -> {result.rel}")
        incomplete_note = ""
        if result.rel.lower().endswith((".html", ".htm")):
            unclosed = validate.html_unclosed_tags(result.new_text)
            if unclosed:
                tags = ", ".join(f"<{t}>" for t in unclosed)
                line_count = len(result.new_text.splitlines())
                self.console.print(f"[yellow]note:[/yellow] {result.rel} is unfinished - unclosed {tags}")
                incomplete_note = (
                    f" WARNING: {result.rel} is unfinished - {tags} never closed, so the page will "
                    f"render blank. INSERT the rest after line {line_count} before saying it's done."
                )
        return f"Applied. {result.rel} now reflects the change.{review_note}{incomplete_note}", False

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
        self.progress.start(user_text)
        self.progress.set_project_files(files.project_files(self.root))

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
        self.progress.start(f"TODO #{todo.id}: {todo.text}")
        self.progress.set_project_files(files.project_files(self.root))
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

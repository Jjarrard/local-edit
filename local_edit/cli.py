"""Terminal chat app for local-edit."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import completion, files, shell, ui
from .agent import DEFAULT_CONTEXT_SIZE, Agent, ReviewAborted
from .client import JanClient, JanConnectionError
from .config import Config, config_dir, migrate_legacy_dir
from .review import render_diff
from .state import Session

HELP = """\
slash commands:
  /help                 show this
  /status               show models, toggles, and queue state
  /todo                 show the todo list
  /add <text>           add a todo item manually
  /work [goal]          plan (if goal given) and autonomously work the todo queue
  /model [name]         pick the code model - no name opens an arrow-key picker
  /fast [name|off]      pick the small model used for planning and review
  /run <command>        run a shell command yourself
  /auto [on|off]        toggle auto-apply (skip the human confirm on edits)
  /review [on|off]      toggle the model's self-review pass before edits
  /diff                 show the diff of the most recently applied edit
  /undo                 revert the most recently applied edit
  /history              list every edit applied this session (newest first)
  /reset                clear chat history (keeps todos and files)
  /quit, /exit          leave
anything else is sent to the model as a chat message.
"""

REASONING_HINTS = ("think", "reason", "r1", "qwq", "o1")
# Heuristics for guessing which loaded model is the small/fast one.
SMALL_HINTS = ("0.5b", "1b", "1.5b", "2b", "3b", "mini", "small", "tiny")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="local-edit", description=__doc__)
    p.add_argument("--base-url", default=None, help="Jan OpenAI-compatible API base URL")
    p.add_argument("--model", default=None, help="code model id (see /v1/models); remembered between runs")
    p.add_argument("--fast-model", default=None, help="small model for planning/review; defaults to the code model")
    p.add_argument(
        "--auto-fast",
        action="store_true",
        help="guess a small model for planning/review instead of leaving it on the code model",
    )
    p.add_argument("--project", default=".", help="project root to edit (defaults to the current directory)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=None, help="defaults to 512, or 2048 for reasoning models")
    p.add_argument("--max-steps", type=int, default=12, help="max actions per todo before giving up")
    p.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="model context window in tokens (default 4096; learned from the server on overflow). "
        "Note llama.cpp's --parallel N divides the context between slots.",
    )
    p.add_argument("--auto-apply", action="store_true", help="apply edits without asking for confirmation")
    p.add_argument("--no-self-review", action="store_true", help="skip the model's self-review pass")
    p.add_argument("--no-run", action="store_true", help="disable the RUN command entirely")
    p.add_argument("--confirm-all-commands", action="store_true", help="ask before even read-only commands")
    p.add_argument("--command-timeout", type=int, default=shell.DEFAULT_TIMEOUT, help="seconds before a RUN is killed")
    p.add_argument("--goal", default=None, help="start immediately in autonomous /work mode with this goal")
    return p


def _guess_fast_model(available: list[str], code_model: str) -> str | None:
    for name in available:
        if name != code_model and any(h in name.lower() for h in SMALL_HINTS):
            return name
    return None


def _pick_max_tokens(model: str, explicit: int | None, console: Console, context_size: int) -> int:
    # Never reserve more than a third of the window for the reply, or there's
    # no room left for the system prompt and the conversation.
    ceiling = max(256, context_size // 3)
    if explicit is not None:
        if explicit > ceiling:
            console.print(f"[yellow]--max-tokens {explicit} won't fit a {context_size}-token context; using {ceiling}[/yellow]")
            return ceiling
        return explicit
    if any(hint in model.lower() for hint in REASONING_HINTS):
        chosen = min(2048, ceiling)
        console.print(f"[dim]reasoning model detected, using --max-tokens {chosen}[/dim]")
        return chosen
    return min(512, ceiling)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    console = Console()
    root = Path(args.project).resolve()
    if not root.exists():
        console.print(f"[bold red]project path does not exist: {root}[/bold red]")
        return 1

    migrate_legacy_dir(config_dir())
    migrate_legacy_dir(root / ".local-edit")

    cfg = Config.load()
    base_url = args.base_url or cfg.base_url
    client = JanClient(base_url=base_url, model="", temperature=args.temperature)

    try:
        available = client.list_models()
    except JanConnectionError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        console.print("[dim]start Jan, load a model, and enable its local API server.[/dim]")
        return 1
    if not available:
        console.print("[bold red]Jan is running but has no models loaded.[/bold red]")
        return 1

    # Preference order: explicit flag > remembered choice > first available.
    model = args.model or (cfg.code_model if cfg.code_model in available else None) or available[0]
    if args.model and args.model not in available:
        console.print(f"[yellow]warning: '{args.model}' isn't loaded in Jan[/yellow]")
    client.model = model
    context_size = args.context_size or DEFAULT_CONTEXT_SIZE
    client.max_tokens = _pick_max_tokens(model, args.max_tokens, console, context_size)

    # Only ever route to a small model when the user actually chose one.
    # Guessing silently is how you end up with planning quietly handed to a
    # model that can't do it - the failure looks like the tool being broken.
    fast_model = args.fast_model or (cfg.fast_model if cfg.fast_model in available else None)
    if fast_model is None and args.auto_fast:
        fast_model = _guess_fast_model(available, model)
    if fast_model == model:
        fast_model = None

    cfg.base_url = base_url
    cfg.code_model = model
    cfg.fast_model = fast_model
    cfg.remember_model(model)
    cfg.save()

    session = Session(root)
    session.load()

    agent = Agent(
        client=client,
        root=root,
        session=session,
        console=console,
        auto_apply=args.auto_apply,
        self_review_enabled=not args.no_self_review,
        max_steps=args.max_steps,
        context_size=context_size,
        fast_model=fast_model,
        allow_run=not args.no_run,
        auto_run_safe=not args.confirm_all_commands,
        command_timeout=args.command_timeout,
    )

    console.print(f"[bold]local-edit[/bold]  {root}")
    _print_status(console, agent, cfg, available)
    console.print("[dim]/help for commands, /model to switch models, or just chat.[/dim]")
    if agent.fast_model is None and len(available) > 1:
        console.print("[dim]tip: /fast <model> sends planning and review to a smaller model.[/dim]")
    console.print()

    completion.install(get_models=lambda: available)

    if args.goal:
        try:
            agent.work(args.goal)
        except ReviewAborted:
            pass

    while True:
        try:
            text = ui.ask_line("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue

        if text.startswith("/"):
            try:
                if _handle_slash(text, agent, session, console, client, cfg, available):
                    break
            except ReviewAborted:
                console.print("[yellow]stopped[/yellow]")
            continue

        try:
            agent.chat_turn(text)
        except ReviewAborted:
            console.print("[yellow]stopped[/yellow]")

    session.save()
    console.print("bye.")
    return 0


def _print_status(console: Console, agent: Agent, cfg: Config, available: list[str]) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("code model", agent.code_model)
    table.add_row("fast model", agent.fast_model or "(same as code model)")
    table.add_row("server", cfg.base_url)
    table.add_row("context", f"{agent.context_size} tokens (reply up to {agent.client.max_tokens})")
    table.add_row("auto-apply", "on" if agent.auto_apply else "off (asks before each edit)")
    table.add_row("self-review", "on" if agent.self_review_enabled else "off")
    table.add_row("run commands", "on" if agent.allow_run else "off")
    pending = agent.session.todos.pending_count()
    table.add_row("todo queue", f"{pending} pending" if pending else "empty")
    total_attempts = agent.edits_applied + agent.validation_failures + agent.review_rejections
    if total_attempts:
        table.add_row(
            "edit outcomes",
            f"{agent.edits_applied} applied, {agent.validation_failures} failed validation, "
            f"{agent.review_rejections} rejected by review",
        )
    console.print(table)


def _choose_model(console: Console, title: str, available: list[str], cfg: Config, current: str | None) -> str | None:
    ordered = cfg.order_models(available)
    return ui.select(console, title, ordered, current=current)


def _handle_slash(
    text: str,
    agent: Agent,
    session: Session,
    console: Console,
    client: JanClient,
    cfg: Config,
    available: list[str],
) -> bool:
    """Returns True if the REPL should exit."""
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return True

    if cmd == "/help":
        console.print(HELP)
        return False

    if cmd == "/status":
        _print_status(console, agent, cfg, available)
        return False

    if cmd == "/todo":
        console.print(session.todos.render())
        return False

    if cmd == "/add":
        if not arg:
            console.print("[red]usage: /add <text>[/red]")
            return False
        item = session.todos.add(arg)
        if item is None:
            console.print("[red]todo queue is full; finish or drop existing todos first[/red]")
        else:
            console.print(f"[yellow]+ todo #{item.id}: {item.text}[/yellow]")
        session.save()
        return False

    if cmd == "/work":
        agent.work(arg or None)
        return False

    if cmd == "/run":
        if not arg:
            console.print("[red]usage: /run <command>[/red]")
            return False
        from .protocol import Action

        agent._handle_run(Action(kind="RUN", text=arg))
        return False

    if cmd == "/model":
        chosen = arg or _choose_model(console, "Select the code model", available, cfg, client.model)
        if not chosen:
            console.print("[dim](unchanged)[/dim]")
            return False
        client.model = chosen
        client.max_tokens = _pick_max_tokens(chosen, None, console, agent.context_size)
        if agent.fast_model == chosen:
            agent.fast_model = None
        cfg.code_model = chosen
        cfg.remember_model(chosen)
        cfg.save()
        console.print(f"code model: [bold]{chosen}[/bold]")
        return False

    if cmd == "/fast":
        if arg.lower() in ("off", "none"):
            agent.fast_model = None
            cfg.fast_model = None
            cfg.save()
            console.print("fast model: (same as code model)")
            return False
        chosen = arg or _choose_model(console, "Select the fast model", available, cfg, agent.fast_model)
        if not chosen:
            console.print("[dim](unchanged)[/dim]")
            return False
        agent.fast_model = None if chosen == client.model else chosen
        cfg.fast_model = agent.fast_model
        cfg.save()
        console.print(f"fast model: [bold]{agent.fast_model or '(same as code model)'}[/bold]")
        return False

    if cmd == "/auto":
        agent.auto_apply = arg.lower() in ("on", "true", "1", "yes") if arg else not agent.auto_apply
        cfg.auto_apply = agent.auto_apply
        cfg.save()
        console.print(f"auto-apply: {'on' if agent.auto_apply else 'off'}")
        return False

    if cmd == "/review":
        agent.self_review_enabled = arg.lower() in ("on", "true", "1", "yes") if arg else not agent.self_review_enabled
        cfg.self_review = agent.self_review_enabled
        cfg.save()
        console.print(f"self-review: {'on' if agent.self_review_enabled else 'off'}")
        return False

    if cmd == "/diff":
        edit = session.last_edit()
        if not edit:
            console.print("[dim](no edits applied yet)[/dim]")
            return False
        old_text = Path(edit["backup"]).read_text()
        current_path = agent.root / edit["rel"]
        new_text = current_path.read_text() if current_path.exists() else ""
        render_diff(console, files.diff_texts(edit["rel"], old_text, new_text), f"last edit: {edit['rel']}")
        return False

    if cmd == "/undo":
        edit = session.pop_last_edit()
        if not edit:
            console.print("[dim](nothing to undo)[/dim]")
            return False
        old_text = Path(edit["backup"]).read_text()
        target = agent.root / edit["rel"]
        if edit["is_new_file"] and old_text == "":
            target.unlink(missing_ok=True)
            console.print(f"removed {edit['rel']} (it was newly created)")
        else:
            target.write_text(old_text)
            console.print(f"reverted {edit['rel']}")
        session.save()
        return False

    if cmd == "/history":
        if not session.applied_edits:
            console.print("[dim](no edits applied yet)[/dim]")
            return False
        for i, edit in enumerate(reversed(session.applied_edits), start=1):
            tag = "new file" if edit["is_new_file"] else "edit"
            console.print(f"{i}. [{edit['time']}] {tag} {edit['rel']}")
        console.print("[dim]/undo reverts them one at a time, most recent first.[/dim]")
        return False

    if cmd == "/reset":
        session.history.clear()
        session.save()
        console.print("chat history cleared (todos kept)")
        return False

    console.print(f"[red]unknown command: {cmd}[/red] (try /help)")
    return False

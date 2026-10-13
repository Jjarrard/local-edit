# local-edit

A code-editing agent framework built for the dumbest local models you'd
actually run — 1-3B — hooked into [Jan](https://jan.ai)'s local
OpenAI-compatible API (`http://127.0.0.1:1338/v1` by default).

Small models can't reliably do JSON tool-calling, can't reproduce a whole file
without mangling it, and will confidently hallucinate that they finished work
they never did. So the design assumption here is **the model is unreliable and
the harness has to be the thing that's correct**.

## Run it

Setting up from scratch:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

Then run it via the wrapper script:

```bash
./local-edit-run --project /path/to/your/code
```

That's it — no venv activation, no flags. It picks up your last-used model,
talks to Jan, and edits files in place. Make it a bare command (run this from
inside the repo folder):

```bash
echo "alias local-edit=\"$(pwd)/local-edit-run\"" >> ~/.zshrc
```

## Switching models

`/model` with no argument opens an arrow-key picker: up/down to move, type to
filter, Enter to select, Esc to cancel. Most-recently-used models sort first,
and your choice is remembered in `~/.local-edit/config.json` between runs.

There are two model slots, because the jobs are different:

| slot                      | used for                 | wants                           |
| ------------------------- | ------------------------ | ------------------------------- |
| **code model** (`/model`) | writing and editing code | the most capable model you have |
| **fast model** (`/fast`)  | planning, diff review    | something small and quick       |

The fast model is opt-in — it's never guessed silently, because quietly routing
planning to a model that can't do it looks exactly like the tool being broken.
`/fast off` puts everything back on the code model.

## What the harness guarantees

The interesting part isn't the command grammar, it's the set of things a bad
model **cannot** do:

- **Can't land broken code.** Every edit is parsed and validated _before_ it
  touches disk — Python via `ast`, JSON via `json`, TOML via `tomllib`,
  braces for C-family languages. A syntactically broken edit is rejected
  with the specific error and the file is left untouched. Validators are a
  pluggable suffix->function registry (`validate.register_validator`), so
  wiring in a new language doesn't mean editing the dispatch logic. See
  [`validate.py`](local_edit/validate.py).
- **Can't leave a half-written file.** Writes go to a temp file in the same
  directory and are moved into place with an atomic rename, so a process
  killed mid-write can never leave a file with a mix of old and new content.
  See `files.write`.
- **Can't land code a reviewer rejected.** A second pass judges the change and
  a NO verdict blocks the write, even under `--auto-apply`. The reviewer sees
  before/after text rather than a unified diff, because small models misread
  diffs and reject correct fixes. If the model reconsiders and re-sends the
  _identical_ edit, it goes through — one wrong verdict must not deadlock the
  session. See [`review.py`](local_edit/review.py).
- **Can't fake having done the work.** A model that replies with an invented
  transcript (`You: INSERT ... Result: tests passed ... DONE`) gets everything
  after the first real command stripped, and it never enters history — so it
  can't read its own fiction next turn and double down. See
  `strip_hallucinated_continuation` in [`protocol.py`](local_edit/protocol.py).
- **Can't run anything catastrophic.** `RUN` classifies every command:
  a read-only allowlist runs freely, a denylist (`rm -rf`, `sudo`, `mkfs`,
  `curl | sh`, `git push`, …) never runs at all, and everything else needs a
  yes. Commands run with `cwd` pinned to the project, `stdin` closed so they
  can't hang, a wall-clock timeout, and truncated output. See
  [`shell.py`](local_edit/shell.py).
- **Can't loop forever.** Repeated identical replies, off-task `TODO ADD`
  spam, and mid-stream repetition are all cut off; the todo queue is capped;
  a single `/work` call processes a bounded number of todos.
- **Can't be approved by accident.** Buffered keystrokes are flushed before
  every prompt, so something you typed while the model was streaming is never
  consumed as an approval.

## UI

- The `you>` prompt is not editable — the prompt is passed to readline with
  `\001`/`\002` width markers so Backspace can't walk left and eat it.
- A spinner shows what's happening and which model is doing it
  (`thinking (Jan-code-4b)`, `reviewing the change`, `running: pytest -q`),
  and stops the instant real output starts streaming.
- `/status` shows both models, every toggle, and the queue at a glance.

## Commands

The model speaks a line-oriented grammar, one action per reply:

`SAY` · `LIST` · `READ` · `GREP` · `RUN` · `TODO ADD/DONE/LIST` · `EDIT` ·
`INSERT` · `DELETE` · `DONE`

Edits are always line-range based (`EDIT src/app.py 12-14` + a fenced block)
against a file the model was just shown with line numbers — never whole-file
rewrites.

## Slash commands

| command                | does                                              |
| ---------------------- | ------------------------------------------------- |
| `/status`              | models, toggles, queue state                      |
| `/model [name]`        | pick the code model (arrow-key picker if no name) |
| `/fast [name\|off]`    | pick the small planning/review model              |
| `/work [goal]`         | plan a todo queue and work it autonomously        |
| `/todo`, `/add <text>` | inspect / extend the queue                        |
| `/run <command>`       | run a shell command yourself                      |
| `/auto [on\|off]`      | auto-apply edits without confirming               |
| `/review [on\|off]`    | toggle the self-review pass                       |
| `/diff`, `/undo`       | inspect / revert the last applied edit            |
| `/history`             | list every edit applied this session              |
| `/reset`               | clear chat history (keeps todos and files)        |
| `/help`, `/quit`       |                                                   |

Not using Jan? See [docs/custom-api.md](docs/custom-api.md) for pointing
local-edit at another OpenAI-compatible API.

## Flags

`--project` `--model` `--fast-model` `--auto-fast` `--base-url` `--max-tokens`
`--max-steps` `--auto-apply` `--no-self-review` `--no-run`
`--confirm-all-commands` `--command-timeout` `--goal` `--temperature`

Reasoning models that emit `<think>` blocks get `--max-tokens 2048`
automatically; override if you need more.

### How `RUN` flags interact with the command classifier

Every `RUN` command is first classified `SAFE` / `NEEDS_APPROVAL` / `BLOCKED`
(see [`shell.py`](local_edit/shell.py)); these flags change what happens next,
not the classification itself - a `BLOCKED` command never runs no matter
what flags are set.

| verdict         | default                        | `--auto-apply`   | `--confirm-all-commands` | `--no-run`         |
| --------------- | ------------------------------- | ----------------- | ------------------------- | -------------------- |
| `SAFE`          | runs immediately                | runs immediately  | asks first                 | never offered at all |
| `NEEDS_APPROVAL`| asks first                      | runs immediately* | asks first                 | never offered at all |
| `BLOCKED`       | refused, always                 | refused, always   | refused, always            | never offered at all |

\* Commands that touch a path outside the project root are never
auto-run - not even under `--auto-apply` - because the project root is a
boundary the user agreed to, not something an "apply things without asking"
flag should be able to widen.

### Per-project command rules

Drop `<project>/.local-edit/command_rules.json` in a project to tune the
classifier without editing source:

```json
{
  "allow_programs": ["mvn"],
  "deny_programs": ["ssh"],
  "allow_subcommands": {"docker": ["build"]}
}
```

`allow_programs`/`allow_subcommands` only add new `SAFE` entries; they can
never re-enable something on the built-in `BLOCKED_PATTERNS` denylist (`rm
-rf`, `sudo`, `git push`, …). `deny_programs` escalates a program that would
otherwise be `SAFE` to `NEEDS_APPROVAL`.

## State

Per project, under `<project>/.local-edit/`: `session.json` (chat history + todo
queue, so sessions survive restarts) and `backups/` (pre-edit copies, used by
`/undo`). Model preferences are global, in `~/.local-edit/config.json`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

311 tests. The bulk of them encode failures actually observed against local
models — hallucinated transcripts, runaway `TODO ADD` loops, reviewers
rejecting correct fixes, chat templates that reject non-alternating roles,
edits that nest a function inside itself.

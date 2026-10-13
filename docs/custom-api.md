# Using a different API

local-edit talks to any OpenAI-compatible chat completions endpoint, not just Jan.
Point it elsewhere with `--base-url`:

```bash
./local-edit-run --project /path/to/your/code --base-url http://localhost:11434/v1
```

Or persist it so you don't have to pass the flag every run — set `base_url`
in `~/.local-edit/config.json`:

```json
{
  "base_url": "http://localhost:11434/v1"
}
```

Any server exposing `/chat/completions` in the OpenAI format works (Ollama,
LM Studio, vLLM, llama.cpp server, etc). `--model` / `--fast-model` should
match a model name that server serves.

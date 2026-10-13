from local_edit.completion import candidates

MODELS = [
    "Jan-code-4b-Q4_K_M",
    "Jan-v3.5-4B-Q4_K_XL",
    "Qwen3_5-9B-Claude-4_6-OS-AV-H-UNCENSORED-THINK-D_AU-IQ4_XS-imat",
    "bartowski/DeepSeek-R1-Distill-Qwen-14B-Q5_K_M",
]


def test_model_command_completes_from_model_list():
    result = candidates("/model ", "", MODELS)
    assert result == sorted(MODELS)


def test_model_command_filters_by_prefix():
    result = candidates("/model Jan", "Jan", MODELS)
    assert result == ["Jan-code-4b-Q4_K_M", "Jan-v3.5-4B-Q4_K_XL"]


def test_model_command_keeps_full_id_with_slash_intact():
    # model ids can contain "/" (e.g. "bartowski/DeepSeek-..."); the whole
    # id must be offered, not just the tail after the slash
    result = candidates("/model bartowski", "bartowski", MODELS)
    assert result == ["bartowski/DeepSeek-R1-Distill-Qwen-14B-Q5_K_M"]


def test_slash_prefix_completes_command_names():
    result = candidates("/mo", "/mo", MODELS)
    assert result == ["/model"]


def test_no_match_returns_empty():
    result = candidates("/model zzz", "zzz", MODELS)
    assert result == []


def test_plain_chat_text_gets_no_completions():
    result = candidates("what does this function do", "do", MODELS)
    assert result == []


def test_command_completion_ignores_models():
    result = candidates("/w", "/w", MODELS)
    assert result == ["/work"]

from local_edit import validate


def test_valid_python_passes():
    assert validate.validate("a.py", "def f():\n    return 1\n").ok


def test_broken_python_is_caught():
    result = validate.validate("a.py", "def f(:\n    return 1\n")
    assert not result.ok
    assert "syntax error" in result.message.lower()


def test_bad_indentation_is_caught():
    result = validate.validate("a.py", "def f():\nreturn 1\n")
    assert not result.ok


def test_unclosed_paren_in_python_is_caught():
    result = validate.validate("a.py", "print('hello'\n")
    assert not result.ok


def test_valid_json_passes():
    assert validate.validate("a.json", '{"a": 1}').ok


def test_broken_json_is_caught():
    result = validate.validate("a.json", '{"a": 1,}')
    assert not result.ok
    assert "json" in result.message.lower()


def test_empty_json_file_passes():
    assert validate.validate("a.json", "  ").ok


def test_balanced_js_passes():
    assert validate.validate("a.js", "function f() { return [1, 2]; }\n").ok


def test_unclosed_brace_in_js_is_caught():
    result = validate.validate("a.js", "function f() { return 1;\n")
    assert not result.ok
    assert "unclosed" in result.message.lower()


def test_mismatched_bracket_in_js_is_caught():
    result = validate.validate("a.js", "const a = [1, 2};\n")
    assert not result.ok


def test_braces_inside_js_strings_are_ignored():
    # a naive counter would flag this valid file
    assert validate.validate("a.js", 'const s = "a { unmatched";\nconst t = 1;\n').ok


def test_braces_inside_js_comments_are_ignored():
    assert validate.validate("a.js", "// a { comment\n/* another } here */\nconst a = 1;\n").ok


def test_escaped_quote_in_js_string_is_handled():
    assert validate.validate("a.js", 'const s = "he said \\"hi\\"";\n').ok


def test_valid_toml_passes():
    assert validate.validate("a.toml", '[section]\nkey = "value"\nnums = [1, 2, 3]\n').ok


def test_broken_toml_is_caught():
    result = validate.validate("a.toml", "key = \n[section\n")
    assert not result.ok


def test_empty_toml_file_passes():
    assert validate.validate("a.toml", "").ok


def test_valid_ini_passes():
    assert validate.validate("a.ini", "[section]\nkey = value\n").ok


def test_broken_ini_is_caught():
    result = validate.validate("a.ini", "not a valid ini [[[")
    assert not result.ok


def test_register_validator_adds_new_extension():
    calls = []

    def _fake(text):
        calls.append(text)
        return validate.ValidationResult.bad("nope") if "bad" in text else validate.ValidationResult.good()

    validate.register_validator(".fake", _fake)
    try:
        assert validate.validate("x.fake", "ok text").ok
        assert not validate.validate("x.fake", "bad text").ok
        assert calls == ["ok text", "bad text"]
    finally:
        del validate.VALIDATORS[".fake"]


def test_unknown_extension_always_passes():
    assert validate.validate("notes.md", "# hi {unbalanced").ok
    assert validate.validate("data.xyz", "((((").ok


def test_html_is_not_bracket_checked():
    # HTML legitimately contains unbalanced braces in inline CSS/JS snippets
    assert validate.validate("index.html", "<style>body { color: red }</style>").ok


def test_markup_inside_a_style_block_is_rejected():
    # Seen live: EDIT index.html 15-100 replaced CSS rules with the page's
    # header/main/footer, all inside <style>.
    from local_edit.validate import validate

    broken = "<html><head><style>\nbody { margin: 0; }\n        <footer class=\"f\">\n</style></head><body></body></html>"
    result = validate("index.html", broken)
    assert not result.ok and "line 3" in result.message

    fine = "<html><head><style>\nbody { margin: 0; }\n</style></head><body><footer>x</footer>\n" \
           "<script>el.innerHTML = `\n<div class=\"card\">`;</script></body></html>"
    assert validate("index.html", fine).ok


def test_content_after_closing_html_is_rejected():
    # Seen live: INSERT index.html 176 appended a new stall after </html>.
    from local_edit.validate import validate

    assert not validate("index.html", "<html><body></body></html>\n<div class=\"stall\">x</div>\n").ok
    assert validate("index.html", "<html><body></body></html>\n\n").ok


def test_js_syntax_errors_with_balanced_brackets_are_caught():
    import shutil

    import pytest

    from local_edit.validate import validate

    if shutil.which("node") is None:
        pytest.skip("node not installed - JS falls back to bracket balance")
    assert validate("app.js", "const x = 5;\n").ok
    assert validate("app.js", 'import a from "./b.js";\nexport const c = a;\n').ok
    bad = validate("app.js", "function f() {\n  if (x) {\n  }\n  else else {}\n}\n")
    assert not bad.ok and "line 4" in bad.message

import pytest

from local_edit import files


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n"
        "\n"
        "def main():\n"
        "    print(add(1, 2))\n"
    )
    return tmp_path


def test_resolve_blocks_path_traversal(project):
    with pytest.raises(files.PathError):
        files.resolve(project, "../outside.py")


def test_resolve_blocks_absolute_escape(project):
    with pytest.raises(files.PathError):
        files.resolve(project, "/etc/passwd")


def test_read_lines_numbers_output(project):
    out = files.read_lines(project, "src/app.py", 1, 2)
    assert "1: def add(a, b):" in out
    assert "2:     return a - b" in out


def test_read_lines_missing_file(project):
    with pytest.raises(files.PathError):
        files.read_lines(project, "src/nope.py")


def test_list_tree_ignores_dotdirs(project):
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref")
    out = files.list_tree(project)
    assert ".git" not in out
    assert "src/" in out
    assert "src/app.py" in out


def test_grep_finds_match(project):
    out = files.grep(project, "return a - b")
    assert "app.py:2" in out


def test_grep_no_match(project):
    out = files.grep(project, "nonexistent_symbol_xyz")
    assert "no matches" in out


def test_plan_replace_fixes_the_bug(project):
    result = files.plan_replace(project, "src/app.py", 2, 2, "    return a + b")
    assert "    return a + b" in result.new_text
    assert "-    return a - b" in result.diff
    assert "+    return a + b" in result.diff
    # file on disk is untouched until files.write() is called
    assert "a - b" in (project / "src" / "app.py").read_text()


def test_plan_replace_out_of_bounds(project):
    with pytest.raises(files.PathError):
        files.plan_replace(project, "src/app.py", 50, 51, "x")


def test_plan_insert_at_top(project):
    result = files.plan_insert(project, "src/app.py", 0, "import sys")
    assert result.new_text.startswith("import sys\ndef add")


def test_plan_insert_new_file(project):
    result = files.plan_insert(project, "src/new.py", 0, "x = 1")
    assert result.is_new_file
    assert result.new_text == "x = 1\n"


def test_plan_delete_range(project):
    result = files.plan_delete(project, "src/app.py", 3, 3)
    assert result.new_text.count("\n") == 4  # one blank line removed


def test_write_then_read_roundtrip(project):
    result = files.plan_replace(project, "src/app.py", 2, 2, "    return a + b")
    files.write(project, result.rel, result.new_text)
    assert "a + b" in (project / "src" / "app.py").read_text()


def test_write_no_leftover_temp_files(project):
    result = files.plan_replace(project, "src/app.py", 2, 2, "    return a + b")
    files.write(project, result.rel, result.new_text)
    leftovers = [p for p in (project / "src").iterdir() if p.name != "app.py"]
    assert leftovers == []


def test_write_creates_parent_dirs(project):
    files.write(project, "new/nested/file.py", "x = 1\n")
    assert (project / "new" / "nested" / "file.py").read_text() == "x = 1\n"


def test_write_replaces_existing_content_atomically(project):
    path = project / "src" / "app.py"
    before = path.read_text()
    files.write(project, "src/app.py", "totally new content\n")
    after = path.read_text()
    assert after == "totally new content\n"
    assert after != before


def test_plan_replace_on_single_line_file(tmp_path):
    (tmp_path / "one.py").write_text("x = 1\n")
    result = files.plan_replace(tmp_path, "one.py", 1, 1, "x = 2")
    assert result.new_text == "x = 2\n"


def test_plan_replace_no_trailing_newline_in_source(tmp_path):
    (tmp_path / "no_nl.py").write_text("x = 1")
    result = files.plan_replace(tmp_path, "no_nl.py", 1, 1, "x = 2")
    assert result.new_text == "x = 2\n"


def test_plan_insert_into_empty_file(tmp_path):
    (tmp_path / "empty.py").write_text("")
    result = files.plan_insert(tmp_path, "empty.py", 0, "x = 1")
    assert result.new_text == "x = 1\n"
    assert not result.is_new_file


def test_plan_insert_at_end_of_file(project):
    n = len((project / "src" / "app.py").read_text().splitlines())
    result = files.plan_insert(project, "src/app.py", n, "# trailing comment")
    assert result.new_text.rstrip("\n").endswith("# trailing comment")


def test_plan_insert_past_the_end_appends(project):
    # "after line 14" of a 13-line file: models count the next free line
    result = files.plan_insert(project, "src/app.py", 999, "x")
    assert result.new_text.splitlines()[-1] == "x"
    assert result.new_text.startswith(result.old_text)


def test_plan_insert_negative_line_out_of_bounds(project):
    with pytest.raises(files.PathError):
        files.plan_insert(project, "src/app.py", -1, "x")


def test_plan_delete_out_of_bounds(project):
    with pytest.raises(files.PathError):
        files.plan_delete(project, "src/app.py", 1, 999)


def test_plan_delete_entire_file_leaves_it_empty(tmp_path):
    (tmp_path / "f.py").write_text("a\nb\nc\n")
    result = files.plan_delete(tmp_path, "f.py", 1, 3)
    assert result.new_text == ""


def test_plan_delete_end_before_start_is_out_of_bounds(project):
    with pytest.raises(files.PathError):
        files.plan_delete(project, "src/app.py", 3, 2)


def test_plan_replace_missing_file(project):
    with pytest.raises(files.PathError):
        files.plan_replace(project, "src/nope.py", 1, 1, "x")


def test_plan_delete_missing_file(project):
    with pytest.raises(files.PathError):
        files.plan_delete(project, "src/nope.py", 1, 1)


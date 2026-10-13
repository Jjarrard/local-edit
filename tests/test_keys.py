"""Key decoding for the arrow-key picker.

Regression origin: pressing Down made the picker vanish and put a stray "B"
in the filter box. Down sends \\x1b[B; the old reader polled the OS fd with
select() but consumed bytes through the *buffered* sys.stdin, so the "[B"
tail was invisible to select(). It concluded "bare Escape" (cancelling the
picker) and the leftover bytes leaked in as literal text.
"""

import os

import pytest

from local_edit import ui


@pytest.mark.parametrize(
    "seq,expected",
    [
        (b"\x1b[A", "up"),
        (b"\x1b[B", "down"),
        (b"\x1b[C", "right"),
        (b"\x1b[D", "left"),
        # application cursor mode - some terminals send ESC O A instead
        (b"\x1bOA", "up"),
        (b"\x1bOB", "down"),
    ],
)
def test_arrow_sequences_decode_to_directions(seq, expected):
    assert ui.decode_key(seq) == expected


def test_bare_escape_is_escape():
    assert ui.decode_key(b"\x1b") == "esc"


def test_arrow_tail_never_decodes_as_a_printable_character():
    # the exact bug: "B" from ESC-[-B must never become filter text
    assert ui.decode_escape(b"[B") == "down"
    assert ui.decode_escape(b"[B") != "B"


@pytest.mark.parametrize(
    "data,expected",
    [
        (b"\r", "enter"),
        (b"\n", "enter"),
        (b"\x03", "ctrl-c"),
        (b"\x04", "esc"),
        (b"\x7f", "backspace"),
        (b"\b", "backspace"),
        (b"", "eof"),
        (b"a", "a"),
        (b"Z", "Z"),
        (b"/", "/"),
    ],
)
def test_ordinary_keys(data, expected):
    assert ui.decode_key(data) == expected


def test_unknown_escape_sequence_is_inert():
    # e.g. Home/End/F-keys: recognized as "not a character", so they can't
    # corrupt the filter
    assert ui.decode_key(b"\x1b[3~") == "other"
    assert ui.decode_key(b"\x1b[H") == "other"


def test_control_characters_are_not_treated_as_text():
    assert ui.decode_key(b"\x01") == "other"


def test_invalid_utf8_is_ignored():
    assert ui.decode_key(b"\xff") == "other"


class _FakeReader(ui.KeyReader):
    """Drives KeyReader.read_key over a real pipe, with no terminal involved.

    This exercises the actual os.read + select path - the code that was
    broken - rather than only the pure decoder.
    """

    def __init__(self, payload: bytes):
        super().__init__()
        self._r, w = os.pipe()
        os.write(w, payload)
        os.close(w)
        self._fd = self._r

    def close(self):
        os.close(self._r)


def test_read_key_consumes_a_whole_arrow_sequence_from_the_fd():
    reader = _FakeReader(b"\x1b[B")
    try:
        assert reader.read_key() == "down"
    finally:
        reader.close()


def test_read_key_handles_back_to_back_arrows_without_leaking_bytes():
    # holding Down delivers several sequences in one burst; each must decode
    # cleanly instead of spilling "[B" into the filter
    reader = _FakeReader(b"\x1b[B\x1b[B\x1b[A")
    try:
        assert [reader.read_key() for _ in range(3)] == ["down", "down", "up"]
    finally:
        reader.close()


def test_read_key_distinguishes_typed_text_from_arrows():
    reader = _FakeReader(b"q\x1b[Az")
    try:
        assert reader.read_key() == "q"
        assert reader.read_key() == "up"
        assert reader.read_key() == "z"
    finally:
        reader.close()


def test_read_key_reports_eof_at_end_of_stream():
    reader = _FakeReader(b"")
    try:
        assert reader.read_key() == "eof"
    finally:
        reader.close()


def test_read_key_handles_multibyte_characters():
    reader = _FakeReader("é".encode())
    try:
        assert reader.read_key() == "é"
    finally:
        reader.close()


# -- split_key: exactly one key per call, never over-consuming ---------------


def test_split_key_consumes_only_the_first_arrow():
    assert ui.split_key(b"\x1b[B\x1b[A") == ("down", 3)


def test_split_key_consumes_only_one_character():
    assert ui.split_key(b"abc") == ("a", 1)


def test_split_key_needs_more_data_for_a_partial_sequence():
    assert ui.split_key(b"\x1b[") is None
    assert ui.split_key(b"\x1b") is None


def test_split_key_reports_nothing_for_empty_input():
    assert ui.split_key(b"") is None


def test_split_key_handles_long_csi_sequences():
    key, consumed = ui.split_key(b"\x1b[3~rest")
    assert key == "other"
    assert consumed == 4, "must not swallow the bytes after the sequence"


def test_split_key_waits_for_a_full_multibyte_character():
    assert ui.split_key("é".encode()[:1]) is None

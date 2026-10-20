from local_edit.review import self_review


class FakeClient:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, stop=None, max_tokens=None, model=None):
        return self.reply


def test_self_review_plain_yes():
    ok, reason = self_review(FakeClient("YES, looks correct."), "task", "a.py", "before", "after")
    assert ok is True
    assert "YES" in reason


def test_self_review_plain_no():
    ok, reason = self_review(FakeClient("NO, this breaks main()."), "task", "a.py", "before", "after")
    assert ok is False


def test_self_review_reasoning_model_verdict_at_end():
    reply = (
        "<think>\n\nLet me check the diff carefully.\n\n"
        "It looks like it changes the return value.\n\n"
        "That matches the task.\n\n</think>\n\nYES"
    )
    ok, reason = self_review(FakeClient(reply), "task", "a.py", "before", "after")
    assert ok is True


def test_self_review_reasoning_model_says_no_at_end():
    reply = "<think>\n\nHmm, wait, this seems wrong.\n\n</think>\n\nNO"
    ok, reason = self_review(FakeClient(reply), "task", "a.py", "before", "after")
    assert ok is False


def test_self_review_no_verdict_defaults_to_allow():
    ok, reason = self_review(FakeClient("I'm not sure about this one."), "task", "a.py", "before", "after")
    assert ok is True


def test_self_review_client_error_defaults_to_allow():
    class BrokenClient:
        def chat(self, messages, stop=None, max_tokens=None, model=None):
            raise RuntimeError("connection refused")

    ok, reason = self_review(BrokenClient(), "task", "a.py", "before", "after")
    assert ok is True
    assert "unavailable" in reason


def test_verdict_line_wins_over_prose_in_the_reason():
    from local_edit.review import parse_review_verdict

    assert parse_review_verdict("YES\nNo syntax errors and no regressions.")[0] is True
    assert parse_review_verdict("**NO** - yes, it parses, but it nests the function.")[0] is False
    assert parse_review_verdict("<think>maybe NO... actually fine</think>\nYES, correct.")[0] is True
    assert parse_review_verdict("The answer is NO, it breaks the loop.")[0] is False

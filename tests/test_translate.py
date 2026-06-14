from polyglot.translate import build_messages, translate_with
from polyglot.segments import new_segment


def test_build_messages_includes_context():
    msgs = build_messages("SYSTEM", "current line", prev_text="previous line")
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert "previous line" in msgs[1]["content"]
    assert "current line" in msgs[1]["content"]


def test_build_messages_no_context():
    msgs = build_messages("SYSTEM", "only line", prev_text=None)
    assert msgs[1]["content"] == "only line"


def test_translate_with_fake_generator():
    segs = [
        new_segment(0, 0.0, 1.0, "Hello"),
        new_segment(1, 1.0, 2.0, "World"),
    ]

    def fake_generate(messages):
        user = messages[-1]["content"]
        return f"FR[{user.splitlines()[-1]}]"

    out = translate_with(segs, system="SYS", generate=fake_generate)
    assert out[0]["translation"] == "FR[Hello]"
    assert out[1]["translation"] == "FR[World]"

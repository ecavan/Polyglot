from polyglot.translate import build_messages, translate_with
from polyglot.segments import new_segment


def test_build_messages():
    msgs = build_messages("SYSTEM", "only line")
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert msgs[1] == {"role": "user", "content": "only line"}


def test_translate_with_fake_generator():
    segs = [
        new_segment(0, 0.0, 1.0, "Hello"),
        new_segment(1, 1.0, 2.0, "World"),
    ]

    def fake_generate(messages):
        return "FR[" + messages[-1]["content"] + "]"

    out = translate_with(segs, system="SYS", generate=fake_generate)
    assert out[0]["translation"] == "FR[Hello]"
    assert out[1]["translation"] == "FR[World]"


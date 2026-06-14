from polyglot.orpheus_tts import tokens_to_codes, redistribute


def test_tokens_to_codes_filters_and_aligns():
    # 6 leading invalid markers (small N -> tid <= 0) must be dropped; then a clean
    # 7-token frame at positions 0..6 with the per-position 4096 offset.
    markers = "".join(f"<custom_token_{n}>" for n in range(1, 7))  # tid <= 0, dropped
    frame = "".join(f"<custom_token_{10 + (i * 4096) + (i + 1)}>" for i in range(7))
    codes = tokens_to_codes(markers + frame)
    assert codes == [1, 2, 3, 4, 5, 6, 7]  # each recovered as its position value


def test_tokens_to_codes_truncates_to_whole_frames():
    frame = "".join(f"<custom_token_{10 + (i * 4096) + 1}>" for i in range(7))
    extra = "<custom_token_11><custom_token_4108>"  # 2 leftover tokens
    codes = tokens_to_codes(frame + extra)
    assert len(codes) == 7  # leftover partial frame dropped


def test_redistribute_layers():
    codes = [10, 11, 12, 13, 14, 15, 16]  # one frame
    l1, l2, l3 = redistribute(codes)
    assert l1 == [10]
    assert l2 == [11, 14]
    assert l3 == [12, 13, 15, 16]

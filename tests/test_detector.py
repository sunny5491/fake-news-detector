"""Smoke tests: run with `python3 -m pytest tests -q` after training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from explain import Detector  # noqa: E402

det = Detector()

WIRE = ("The Senate Judiciary Committee voted 12-10 on Thursday to advance the nomination, sending it to the "
        "full Senate, where a vote is expected next week, committee chairman Chuck Grassley said in a statement.")
CLICKBAIT = ("BREAKING: You Won't BELIEVE What This Senator Was Just Caught Doing (VIDEO). Share this before they "
             "take it down! Featured image via screenshot. Read more at the link below.")


def test_wire_style_is_real():
    r = det.predict(WIRE)
    assert r["label"] == "real"


def test_clickbait_is_fake():
    r = det.predict(CLICKBAIT)
    assert r["label"] == "fake"


def test_probabilities_sum_to_one():
    r = det.predict(WIRE)
    assert abs(r["p_real"] + r["p_fake"] - 1) < 1e-3


def test_attributions_cover_words_and_are_normalised():
    r = det.predict(WIRE)
    assert len(r["words"]) > 10
    assert max(abs(s) for _, s in r["words"]) <= 1.0


def test_too_short_input_is_rejected():
    assert "error" in det.predict("hello")

"""Checks for the judge helpers: ranking parsers, win rate, rubric split, and judge fingerprint.

Runs with pytest or standalone: python tests/test_cross_judge.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.cross_judge import parse_ranking, parse_ranking_prose, split_template, win_rate  # noqa: E402
from src.eval.judge_id import judge_id  # noqa: E402

PROMPT = REPO / "config/evaluation/judge_prompt.txt"
CONFIG = REPO / "config/evaluation/judge_config.yaml"


def test_parse_ranking_formats():
    assert parse_ranking("[{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}]") == "model_1"
    assert parse_ranking('```python\n[{"model": "model_2", "rank": 1}, {"model": "model_1", "rank": 2}]\n```') == "model_2"
    assert parse_ranking("[{'model': 'model_2', 'rank': 1}, {'model': 'model_1', 'rank': 2}]") == "model_2"
    assert parse_ranking("[{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 1}]") == "tie"
    assert parse_ranking("[{'model': 'model_1', 'rank': 1}]") is None
    assert parse_ranking("") is None


def test_parse_ranking_prose():
    assert parse_ranking_prose("Overall, model_2 is better because it keeps the numbers.") == "model_2"
    assert parse_ranking_prose("I prefer model 1.") == "model_1"
    assert parse_ranking_prose("Both are equally good.") == "tie"
    assert parse_ranking_prose("No preference stated.") is None


def test_win_rate_counts_ties_as_half():
    assert win_rate([2.0, 1.0, 1.5, 2.0]) == 62.5
    assert win_rate([]) == 0.0


def test_split_template_keeps_placeholders():
    system, user = split_template(PROMPT.read_text(encoding="utf-8"))
    assert system.startswith("You are an expert evaluator")
    assert all(k in user for k in ("{instruction}", "{output_1}", "{output_2}"))
    assert "<|im_start|>" not in system + user


def test_judge_id_tracks_verdict_settings_only():
    cwd = os.getcwd()
    os.chdir(REPO)  # the config names its prompt relative to the repository root
    try:
        base = judge_id("judge-a", CONFIG)
        assert base.startswith("judge-a-") and base == judge_id("judge-a", CONFIG)
        assert judge_id("judge-b", CONFIG) != base
        text = CONFIG.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            procs = Path(d) / "procs.yaml"
            procs.write_text(text.replace("num_procs: 2", "num_procs: 16"), encoding="utf-8")
            temp = Path(d) / "temp.yaml"
            temp.write_text(text.replace("temperature: 0", "temperature: 0.7"), encoding="utf-8")
            assert judge_id("judge-a", procs) == base
            assert judge_id("judge-a", temp) != base
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    os.chdir(REPO)
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"passed={len(tests) - failed} failed={failed}")
    raise SystemExit(1 if failed else 0)

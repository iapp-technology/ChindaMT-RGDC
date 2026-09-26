"""Span-alignment invariants for Phase 1C IFD scoring.

The IFD ratio is only meaningful if the token span recovered by
``data_by_IFD.get_loss_part_text`` really is the target span. The recovery
re-encodes a character prefix, which is a tokenizer-dependent assumption: a
vocabulary entry spanning the ``### Response:``/output boundary shifts the span.

This module re-implements the recovery locally rather than importing it, so the
Phase 1 code stays untouched, and checks the invariants the ratio depends on for
every scorer tokenizer in TOKENIZER_DIRS.
"""

import glob
import json
import os
import random

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # standalone fallback, see __main__ below
    class _Pytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda f: f

        class _Skip(Exception):
            pass

        @staticmethod
        def skip(msg=""):
            raise _Pytest._Skip(msg)

        @staticmethod
        def fixture(*a, **k):
            return lambda f: f

        class raises:
            def __init__(self, exc):
                self.exc = exc

            def __enter__(self):
                return self

            def __exit__(self, t, v, tb):
                assert t is not None and issubclass(t, self.exc), f"expected {self.exc.__name__}"
                return True

    pytest = _Pytest()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LENGTH = 1024

# Phase 1B checkpoint written by config/training/phase1b_pre_experience_qwen35_4b.yaml
TOKENIZER_DIRS = {
    "qwen35": f"{REPO}/training/LlamaFactory/saves/Qwen3.5-4B/freeze/pre_experience",
}

# Retyped from src/cherry/data_by_IFD.py:7-18 so a drift in the source is caught here.
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def build_texts(record, prompt_type="alpaca"):
    """Replicates data_by_IFD.py:59-73."""
    output_i = record["output"]
    direct_text = "### Response:" + output_i
    if prompt_type == "wiz":
        instruct_i = record["instruction"]
        whole_text = instruct_i + "\n\n### Response:" + output_i
    else:
        instruct_i = record["instruction"]
        input_i = record["input"]
        if input_i == "":
            temp = PROMPT_DICT["prompt_no_input"].format_map(record)
            whole_text = temp + output_i
            instruct_i = temp
        else:
            temp = PROMPT_DICT["prompt_input"].format_map(record)
            whole_text = temp + output_i
            instruct_i = temp
    return instruct_i, direct_text, whole_text


def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):
    """Replicates data_by_IFD.py:79-90 including the off-by-one arithmetic."""
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    start_index = text.rfind(target_span)
    text_temp = text[:start_index]
    token_id_temp = tokenizer.encode(text_temp)
    start_token = len(token_id_temp)
    end_token_real = input_ids.shape[1]
    loss_list = loss_list_[start_token - 1 : end_token_real - 1]
    return end_token_real - start_token, input_ids[0][start_token:end_token_real], np.array(loss_list), start_token, end_token_real


def synth_losses(tokenizer, text, max_length):
    """Loss vector with the producer's length/offset contract: element j predicts token j+1."""
    enc = tokenizer.encode(text, truncation=True, max_length=max_length)
    return [float(j) for j in range(len(enc) - 1)]


BASE_INSTRUCTION = "Task: Translate Thai -> Thai."

RECORDS = [
    ("thai_plain", "สวัสดีครับ วันนี้อากาศดีมาก", ""),
    ("thai_nospace_long", "การแปลภาษาไทยเป็นภาษาอังกฤษนั้นต้องอาศัยความเข้าใจบริบททางวัฒนธรรมอย่างลึกซึ้ง", ""),
    ("thai_leading_vowel", "ไม่เป็นไร", ""),
    ("thai_leading_combining_mark", "่ทดสอบ", ""),
    ("thai_zwsp", "2020 องค์การอนามัยโลก​ ไทย", ""),
    ("english_plain", "Hello, how are you today?", ""),
    ("english_leading_hyphen", "- TripAdvisor is proud to announce", ""),
    ("leading_newline", "\nnewline first", ""),
    ("leading_space", " leading space output", ""),
    ("leading_quote", '"quoted answer"', ""),
    ("whitespace_only_output", "   ", ""),
    ("output_contains_marker", "### Response: nested ### Response: twice", ""),
    ("emoji", "ผลลัพธ์ 😀 ok", ""),
    ("alpaca_with_input", "And in the day", "### Text: องค์ พระผู้เป็นเจ้า"),
]


@pytest.fixture(scope="module", params=sorted(TOKENIZER_DIRS))
def tok(request):
    path = TOKENIZER_DIRS[request.param]
    if not os.path.isdir(path):
        pytest.skip(f"tokenizer not on disk: {path}")
    t = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    t._arm_id = request.param
    return t


def test_p1_p3_tokenizer_preconditions(tok):
    """P1-P3: special-token layout must let the prefix count cancel."""
    n_special = len(tok.encode(""))
    assert n_special in (0, 1), f"{tok._arm_id}: encode('') has {n_special} specials; span arithmetic invalid"
    if n_special == 1:
        assert tok.encode("")[0] == tok.bos_token_id
        assert tok.encode("abc")[0] == tok.bos_token_id
    assert tok.encode("abc")[-1] != tok.eos_token_id, f"{tok._arm_id}: trailing EOS shifts every span"


@pytest.mark.parametrize("case_id,output,input_", RECORDS)
def test_span_invariants(tok, case_id, output, input_):
    """A2/A3/A4/A5/A7/A8/A9: the invariants the IFD ratio depends on."""
    record = {"instruction": BASE_INSTRUCTION, "input": input_, "output": output}
    instruct_i, direct_text, whole_text = build_texts(record)
    instruct_len = len(tok.encode(instruct_i, truncation=True, max_length=MAX_LENGTH))
    if MAX_LENGTH - instruct_len <= 0:
        pytest.skip("guard 1 rejects")

    losses = {}
    for name, text, ml in (
        ("direct", direct_text, MAX_LENGTH - instruct_len + 4),
        ("whole", whole_text, MAX_LENGTH),
    ):
        synth = synth_losses(tok, text, max(ml, 1))
        len_, ids, loss_slice, start_token, end_real = get_loss_part_text(tok, text, output, max(ml, 1), synth)
        if len_ <= 0:
            pytest.skip(f"guard 4 rejects ({name})")
        decoded = tok.decode(ids)

        # A2: span must be a suffix of the target, else prompt text leaked in.
        assert output.endswith(decoded), f"{tok._arm_id}/{case_id}/{name}: span not a suffix: {decoded!r}"
        # A3: at most one boundary token may be mis-attributed.
        widened = tok.decode(tok.encode(text, truncation=True, max_length=max(ml, 1))[start_token - 1 : end_real])
        assert widened.endswith(output), f"{tok._arm_id}/{case_id}/{name}: >1 boundary token lost"
        # A4: cardinality agreement.
        assert len_ == len(ids) == len(loss_slice) == end_real - start_token
        # A5: loss index j predicts token j+1.
        assert loss_slice.tolist() == [float(start_token - 1 + k) for k in range(len_)]
        # A9: no byte-fallback breakage.
        assert "�" not in decoded, f"{tok._arm_id}/{case_id}/{name}: replacement char in span"
        losses[name] = decoded

        # A7: rfind exactness even when the target repeats inside the prompt.
        if output:
            assert text.rfind(output) == len(text) - len(output)

    # A8: both passes must lose the identical number of characters, so the ratio is matched.
    assert len(losses["direct"]) == len(losses["whole"]), (
        f"{tok._arm_id}/{case_id}: direct/whole spans differ, IFD ratio is over mismatched spans"
    )


def test_a10_guards_reject_degenerate_records(tok):
    """A10: each degenerate record must be rejected by its specific guard."""
    # Empty output -> guard 4 (len_ <= 0).
    rec = {"instruction": BASE_INSTRUCTION, "input": "", "output": ""}
    instruct_i, direct_text, _ = build_texts(rec)
    instruct_len = len(tok.encode(instruct_i, truncation=True, max_length=MAX_LENGTH))
    ml = max(MAX_LENGTH - instruct_len + 4, 1)
    len_, _, _, _, _ = get_loss_part_text(tok, direct_text, "", ml, synth_losses(tok, direct_text, ml))
    assert len_ <= 0, "empty output must be rejected by guard 4"

    # Oversized instruction -> guard 1.
    rec = {"instruction": "คำสั่ง " * 5000, "input": "", "output": "สั้น"}
    instruct_i, _, _ = build_texts(rec)
    instruct_len = len(tok.encode(instruct_i, truncation=True, max_length=MAX_LENGTH))
    assert MAX_LENGTH - instruct_len <= 0, "oversized instruction must be rejected by guard 1"


def test_a11_ifd_ratio_boundary_is_strict():
    """A11: the keep/drop boundary is a strict >, so exactly 1.0 is retained."""
    assert not (1.0 > 1), "IFD exactly 1.0 must be KEPT (guard is strict >)"
    assert 1.0000001 > 1
    # NaN survives the guard, which is why a dtype that overflows silently corrupts selection.
    assert not (float("nan") > 1), "NaN passes the guard and would be selected with an arbitrary rank"


def test_a12_selection_takes_highest_rates():
    """A12: selection keeps the top sample_rate by IFD, ascending indices out."""

    def select(mean_rate_list, sample_rate):
        ordered = sorted(mean_rate_list)
        n = int(len(ordered) * sample_rate)
        chosen = ordered[-n:]
        return sorted(idx for _, idx in chosen), n

    nine = [(i / 10.0, i) for i in range(9)]
    picked, n = select(nine, 0.1)
    assert n == 0 and len(picked) == 9, "int(9*0.1)==0 makes [-0:] return the whole list"

    hundred = [(i / 100.0, i) for i in range(100)]
    picked, n = select(hundred, 0.1)
    assert n == 10 and len(picked) == 10
    assert picked == sorted(picked)
    assert picked == list(range(90, 100)), "must keep the ten HIGHEST rates"


def test_a13_missing_input_key_raises(tok):
    """A13: the alpaca branch indexes record['input'] directly."""
    with pytest.raises(KeyError):
        build_texts({"instruction": BASE_INSTRUCTION, "output": "x"})


CORPORA = sorted(
    glob.glob(f"{REPO}/data/source/pool/elrc_th2en_mid_zero-shot/*_sharegpt.json")
    + glob.glob(f"{REPO}/data/source/pool/tatoeba_en2th_mid_zero-shot/*_sharegpt.json")
)


def _load_real_records(n=300, seed=2):
    out = []
    for path in CORPORA:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for rec in data:
            convs = rec.get("conversations") or []
            if len(convs) < 2 or any(c.get("value") is None for c in convs):
                continue
            out.append({"instruction": convs[0]["value"], "input": "", "output": convs[1]["value"]})
    if not out:
        return []
    return random.Random(seed).sample(out, min(n, len(out)))


def test_a14_aggregate_regression_gate(tok):
    """A14: invariants hold on real corpus records; strict identity stays above the floor."""
    records = _load_real_records()
    if not records:
        pytest.skip("source corpora not on disk")

    strict_ok = considered = 0
    lead_chars = {}
    for rec in records:
        instruct_i, direct_text, whole_text = build_texts(rec)
        instruct_len = len(tok.encode(instruct_i, truncation=True, max_length=MAX_LENGTH))
        if MAX_LENGTH - instruct_len <= 0:
            continue
        spans = {}
        for name, text, ml in (
            ("direct", direct_text, MAX_LENGTH - instruct_len + 4),
            ("whole", whole_text, MAX_LENGTH),
        ):
            ml = max(ml, 1)
            synth = synth_losses(tok, text, ml)
            len_, ids, _, start_token, end_real = get_loss_part_text(tok, text, rec["output"], ml, synth)
            if len_ <= 0:
                spans = {}
                break
            decoded = tok.decode(ids)
            assert rec["output"].endswith(decoded), f"{tok._arm_id}: A2 violated"
            widened = tok.decode(tok.encode(text, truncation=True, max_length=ml)[start_token - 1 : end_real])
            assert widened.endswith(rec["output"]), f"{tok._arm_id}: A3 violated"
            spans[name] = decoded
        if not spans:
            continue
        considered += 1
        assert len(spans["direct"]) == len(spans["whole"]), f"{tok._arm_id}: A8 violated"
        if spans["whole"] == rec["output"]:
            strict_ok += 1
        else:
            lead = rec["output"][:1]
            lead_chars[lead] = lead_chars.get(lead, 0) + 1

    rate = strict_ok / considered if considered else 0.0
    print(f"\n[{tok._arm_id}] strict span identity {strict_ok}/{considered} = {rate:.1%}")
    if lead_chars:
        top = sorted(lead_chars.items(), key=lambda kv: -kv[1])[:8]
        print(f"[{tok._arm_id}] misaligned leading chars: {top}")
    assert rate >= 0.55, f"{tok._arm_id}: strict identity {rate:.1%} below the 55% floor"


def _run_standalone():
    """Runs the suite without pytest so no environment needs a new dependency."""
    failures, skipped, passed = [], 0, 0
    for arm, path in sorted(TOKENIZER_DIRS.items()):
        if not os.path.isdir(path):
            print(f"[{arm}] SKIP - tokenizer not on disk")
            continue
        t = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        t._arm_id = arm
        cases = [("preconditions", lambda: test_p1_p3_tokenizer_preconditions(t))]
        for case_id, output, input_ in RECORDS:
            cases.append((f"span[{case_id}]", lambda c=case_id, o=output, i=input_: test_span_invariants(t, c, o, i)))
        cases += [
            ("guards", lambda: test_a10_guards_reject_degenerate_records(t)),
            ("missing_input_key", lambda: test_a13_missing_input_key_raises(t)),
            ("aggregate_gate", lambda: test_a14_aggregate_regression_gate(t)),
        ]
        for name, fn in cases:
            try:
                fn()
                passed += 1
            except pytest._Skip:
                skipped += 1
            except AssertionError as e:
                failures.append(f"{arm}/{name}: {e}")
    for name, fn in (("ifd_boundary", test_a11_ifd_ratio_boundary_is_strict), ("selection", test_a12_selection_takes_highest_rates)):
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failures.append(f"{name}: {e}")

    print(f"\npassed={passed} skipped={skipped} failed={len(failures)}")
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())

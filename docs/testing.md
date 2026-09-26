# Testing

- [Unit tests](#unit-tests)
- [Smoke checks](#smoke-checks)

## Unit tests

[test_ifd_span_alignment.py](../tests/test_ifd_span_alignment.py) checks that Phase 1C scores IFD
on exactly the output tokens, across tokenizers, and that its guards and top-decile selection
behave as intended. [test_cross_judge.py](../tests/test_cross_judge.py) checks the judge helpers:
the ranking parsers, the win rate, the rubric split, and the judge fingerprint.

```bash
python tests/test_ifd_span_alignment.py     # runs without pytest
python tests/test_cross_judge.py
python -m pytest -q tests                   # with pytest
```

The tokenizer cases need the Phase 1B scorer at the path in `TOKENIZER_DIRS`, and the corpus case
needs the pool. Missing ones are skipped.

## Smoke checks

| Check | Command |
|---|---|
| Patches applied | `git -C alpaca_eval diff --stat && git -C training/LlamaFactory diff --stat` |
| Judge endpoint reachable | `set -a && source .env && set +a && curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"` |
| Phase 2 progress | `bash scripts/03_constraint_augmentation.sh config/rgdc_qwen35.yaml status` |
| Inference on five items | `bash scripts/04_evaluate.sh config/evaluation/default.yaml --n-samples 5 --target-only` |

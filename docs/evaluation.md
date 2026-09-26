# Evaluation

Judges a model against a reference model on [CoreEval](https://huggingface.co/datasets/iapp/ChindaMT-CoreEval)
and [BroadEval](https://huggingface.co/datasets/iapp/ChindaMT-BroadEval) with the length-controlled
win rate of [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), re-judges with cross-judges,
and scores the external metrics of the paper.

- [Setup](#setup)
- [Serve the judge](#serve-the-judge)
- [Run](#run)
- [Cross-judges](#cross-judges)
- [Change the judge](#change-the-judge)
- [Judge against ChindaMT](#judge-against-chindamt)
- [External metrics](#external-metrics)

## Setup

1. Download the suites to `data/eval/<suite>/<split>.jsonl`:

```bash
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
for suite in ("CoreEval", "BroadEval"):
    os.makedirs(f"data/eval/{suite.lower()}", exist_ok=True)
    for split in ("plain", "constrained"):
        shutil.copy(hf_hub_download(f"iapp/ChindaMT-{suite}", f"{split}.jsonl", repo_type="dataset"),
                    f"data/eval/{suite.lower()}/{split}.jsonl")
PY
```

2. Serve a judge ([below](#serve-the-judge)) and point `.env` at it ([.env.example](../.env.example)):
   `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `JUDGE_MODEL_NAME`.

## Serve the judge

The judge is any OpenAI-compatible endpoint, for example one served with [vLLM](https://github.com/vllm-project/vllm):

```bash
pip install vllm
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name qwen3.6-35b-a3b-fp8 --port 8000
```

Then set `OPENAI_BASE_URL=http://localhost:8000/v1` and `JUDGE_MODEL_NAME=qwen3.6-35b-a3b-fp8`. Add
`--api-key <your-api-key>` to require a key, and put the same key in `OPENAI_API_KEY`. The paper
judges with Qwen3.6-35B-A3B-FP8 on vLLM 0.19.0 and cross-checks with GPT-OSS-120B and
Llama-3.3-70B-Instruct; any model served this way can act as a judge or a cross-judge.

## Run

[default.yaml](../config/evaluation/default.yaml) sets the target, the references, and the output
folders. Each model has a file in [config/evaluation/models/](../config/evaluation/models/), where
`model_name_or_path` takes a local checkpoint or a Hub id such as `iapp/ChindaMT-4B`.

```bash
bash scripts/04_evaluate.sh config/evaluation/default.yaml --suite coreeval
bash scripts/04_evaluate.sh config/evaluation/default.yaml --suite broadeval
```

Useful options: `--splits plain`, `--n-samples N`, `--references a.yaml,b.yaml`, `--skip-inference`,
`--target-only`, `--force` to redo inference and judging from scratch.

| Output | Path |
|---|---|
| Predictions | `predictions/<generator>-<suite>-<split>/model_outputs.json` |
| Win rates | `output/<judge>/<target>-vs-<reference>/<suite>-<split>/leaderboard.csv` |

`<judge>` is the served judge name plus a fingerprint of the rubric and the judge settings. The
paper reports `length_controlled_winrate`. A cell counts as done only when every pair has a verdict;
if the judge fails part way, the driver says so, and rerunning the same command judges that cell again.
A reference with a `prompt_adapter` uses its own prompt template, which has no Rules block, so it
runs on the plain split only.

## Cross-judges

[cross_judge.py](../src/eval/cross_judge.py) re-judges a finished cell with another judge, using
the pairs and primary verdicts cached in that cell. It follows the paper protocol: the same rubric
sent as one user message, output order flipped per pair with seed 42, a format reminder then a
prose reading for replies that do not parse, and win rate with a tie as half a win.

```bash
J=output/<judge>
for cell in $J/*/*/; do
    python -m src.eval.cross_judge --cell "$cell" --base-url http://localhost:8001/v1 --model <cross-judge>
done
python -m src.eval.cross_judge --summary $J
```

Each cell gets `cross_judge/<judge>.csv` with per-pair verdicts and `<judge>.json` with agreement,
Cohen's kappa, and the win rate under both judges. The summary prints the win rates of every judge
and their mean.

## Change the judge

| To change | Edit |
|---|---|
| Judge model or endpoint | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `JUDGE_MODEL_NAME` in `.env` |
| Rubric | [judge_prompt.txt](../config/evaluation/judge_prompt.txt) |
| Decoding and parsing | [judge_config.yaml](../config/evaluation/judge_config.yaml) |

AlpacaEval caches verdicts by prompt and outputs only, so every judge writes to its own
`output/<judge>/` folder, and a changed rubric or setting gets a new one
([judge_id.py](../src/eval/judge_id.py)); concurrency and retry settings do not. The
`<|im_start|>` tags in the rubric are the AlpacaEval message format, which turns them into a system
and a user message, so the rubric works with any chat model. AlpacaEval also randomizes which output
comes first in each pair, with a fixed seed.

## Judge against ChindaMT

The suites carry ChindaMT predictions. Write them as reference outputs, and the driver skips
ChindaMT inference:

```bash
python - <<'PY'
import json, os
for suite in ("coreeval", "broadeval"):
    for split in ("plain", "constrained"):
        rows = [json.loads(l) for l in open(f"data/eval/{suite}/{split}.jsonl", encoding="utf-8")]
        gen = f"chindamt-4b-{suite}-{split}"
        os.makedirs(f"predictions/{gen}", exist_ok=True)
        json.dump([{"instruction": r["instruction"], "output": r["prediction_chindamt_4b"], "generator": gen} for r in rows],
                  open(f"predictions/{gen}/model_outputs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
bash scripts/04_evaluate.sh config/evaluation/default.yaml --suite coreeval \
    --references config/evaluation/models/chindamt_4b.yaml
```

Set `target_model` in [default.yaml](../config/evaluation/default.yaml) to your own model first.
For the smaller models, read `prediction_chindamt_2b` or `prediction_chindamt_08b`, name the
generator `chindamt-2b` or `chindamt-0.8b`, and pass the matching model file.

## External metrics

| Metric | Script | Environment |
|---|---|---|
| CometKiwi | [compute_cometkiwi.py](../src/eval/compute_cometkiwi.py) | `chindamt-comet` |
| GEMBA-DA, GEMBA-MQM | [compute_gemba.py](../src/eval/compute_gemba.py) | `chindamt` and the judge |
| MetricX-24 | [compute_metricx.py](../src/eval/compute_metricx.py) | `chindamt`, [metricx](https://github.com/google-research/metricx) cloned to `/tmp/metricx` |

```bash
python -m src.eval.compute_cometkiwi --dataset flores --models <generator>
python -m src.eval.compute_gemba --dataset flores --mode da --models <generator>
python -m src.eval.compute_metricx --dataset flores --models <generator>
```

The scripts read `data/eval/<flores or wmt24pp>/plain.jsonl`, whose rows carry `instruction`,
`direction`, `source_text`, and `reference_text`, and
`predictions/<generator>-<dataset>-plain/model_outputs.json`, and write `output/metrics/`. To produce
the predictions, run `python -m src.eval.run_inference --config <file> --suite flores` with a
configuration like:

```yaml
target_model: config/evaluation/models/chindamt_4b.yaml
suites:
  flores: {path: data/eval/flores, splits: [plain]}
predictions_dir: predictions
```

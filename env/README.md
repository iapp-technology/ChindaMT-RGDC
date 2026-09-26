# Environments

| Environment | Used for | Key pins |
|---|---|---|
| `chindamt` | Phase 1, Phase 2, evaluation, judging | Python 3.11, torch 2.11.0+cu128, transformers 5.5.4, scikit-learn 1.8.0, numpy 1.26.4, accelerate 1.12.0 |
| `chindamt-comet` | CometKiwi scoring | Python 3.12, torch 2.9.1, transformers 4.52.0, unbabel-comet 2.2.7 |
| `chindamt-train` | Fine-tuning and Phase 1B with LLaMA-Factory | Python 3.11, torch 2.10.0+cu126, transformers 5.2.0, deepspeed 0.19.1 |

`unbabel-comet` 2.2.7 requires `transformers` < 5, so CometKiwi runs in `chindamt-comet` while
everything else runs in `chindamt`. Running them in one environment will fail.

## Recreate

```bash
conda env create -f env/environment-chindamt.yml
conda env create -f env/environment-chindamt-comet.yml
```

Or with pip into an existing interpreter:

```bash
pip install -r env/requirements-chindamt.txt
pip install -r env/requirements-chindamt-comet.txt
```

The `environment-*.yml` files were exported with `--no-builds`, so they resolve across machines;
the `requirements-*.txt` files are exact `pip freeze` output from the machine the paper's results
were produced on.

## Training environment

[Training setup](../docs/training.md#setup) installs LLaMA-Factory from the `training/LlamaFactory`
submodule, pinned at upstream commit `436d26b`, into `chindamt-train`.
[environment-chindamt-train.yml](environment-chindamt-train.yml) and
[requirements-chindamt-train.txt](requirements-chindamt-train.txt) record the versions used in the paper.

## MetricX checkout

MetricX-24 scoring imports `google-research/metricx` from `/tmp/metricx`, which is not vendored here.
Clone it before running `src/eval/compute_metricx.py`:

```bash
git clone https://github.com/google-research/metricx /tmp/metricx
```

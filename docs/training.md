# Training

Fine-tunes ChindaMT on Grounded with [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory),
pinned as a submodule at `436d26b`.

- [Setup](#setup)
- [Data](#data)
- [Run](#run)
- [Recipe](#recipe)

## Setup

```bash
conda create -n chindamt-train python=3.11 -y && conda activate chindamt-train
pip install -e training/LlamaFactory -r training/LlamaFactory/requirements/deepspeed.txt
cp config/training/dataset_info.json training/LlamaFactory/data/dataset_info.json
ln -s ../../../data training/LlamaFactory/data/data     # registry paths start with data/
```

Apply the [submodule patch](../patches/README.md) first. Versions used in the paper:
[environment-chindamt-train.yml](../env/environment-chindamt-train.yml).

## Data

| Dataset name | File | Source |
|---|---|---|
| `chindamt_grounded` | `data/grounded/grounded.jsonl` | [iapp/ChindaMT-Grounded](https://huggingface.co/datasets/iapp/ChindaMT-Grounded) |
| `rgdc_output` | `data/rgdc_qwen35/phase2/augmented_sft_dataset_with_rules.jsonl` | [Data pipeline](data_pipeline.md) |

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('iapp/ChindaMT-Grounded', 'grounded.jsonl', repo_type='dataset', local_dir='data/grounded')"
```

## Run

```bash
cp config/training/llamaboard/*.yaml training/LlamaFactory/llamaboard_config/
cd training/LlamaFactory && CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli webui
```

In the web UI, load the saved configuration for [4B](../config/training/llamaboard/chindamt_4b.yaml),
[2B](../config/training/llamaboard/chindamt_2b.yaml), or
[0.8B](../config/training/llamaboard/chindamt_0.8b.yaml), choose the dataset, and
start. Checkpoints are written to `training/LlamaFactory/saves/`.

## Recipe

| Setting | Value |
|---|---|
| Base | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [2B](https://huggingface.co/Qwen/Qwen3.5-2B), [0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Method | Full-parameter SFT, one epoch |
| Learning rate | 2e-5, inverse-square-root schedule, 1% warmup |
| Optimizer | AdamW, weight decay 0.01 |
| Batch | 4 x 8 (4B), 8 x 4 (2B), 16 x 2 (0.8B) on two GPUs, effective 64 |
| Sequence | 1024 tokens, packing |
| Other | bf16, DeepSpeed ZeRO-2, NEFTune alpha 2, seed 42, template `qwen3_5` without thinking |
| Time | About 25 h (4B), 10 h (2B), 8 h (0.8B) on two H100s |

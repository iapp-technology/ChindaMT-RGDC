# Data pipeline

Builds Grounded from the ten [source corpora](corpora.md). To train on the released data instead,
see [Training](training.md).

- [Requirements](#requirements)
- [1. Pool](#1-pool)
- [2. Phase 1: IFD selection](#2-phase-1-ifd-selection)
- [3. Phase 2: constraint augmentation](#3-phase-2-constraint-augmentation)
- [Outputs](#outputs)

The run configuration sets every path: [rgdc_qwen35.yaml](../config/rgdc_qwen35.yaml)
is the paper's run and [example.yaml](../config/example.yaml) is a template.
[run_all.sh](../scripts/run_all.sh) runs 1A, 1C, and Phase 2 in sequence once the 1B checkpoint exists.

## Requirements

- `chindamt` environment ([env/README.md](../env/README.md)) and one GPU for Phase 1.
- LLaMA-Factory for Phase 1B ([Training setup](training.md#setup)).
- An OpenAI-compatible endpoint for the Phase 2 auxiliary LLM, set in `phase2.api_base_url`,
  `phase2.api_key`, and `phase2.llm_model`. The paper used Qwen3.5-35B-A3B-FP8 with thinking off.

## 1. Pool

```bash
python -m src.data.prepare_corpus --input wikimedia.csv --name wikimedia --en-col en --th-col th
python src/data/generate_mid_zero_shot.py \
    --input data/source/sharegpt/wikimedia_en2th_minimal_zero-shot/wikimedia_*_sharegpt.json \
    --output-dir data/source/pool/wikimedia_en2th_mid_zero-shot/ \
    --source-lang English --target-lang Thai --seed 42
```

Repeat for every corpus and for `th2en`. Code: [prepare_corpus.py](../src/data/prepare_corpus.py),
[generate_mid_zero_shot.py](../src/data/generate_mid_zero_shot.py).

## 2. Phase 1: IFD selection

```bash
bash scripts/01_pre_experience.sh config/rgdc_qwen35.yaml                         # 1A
(cd training/LlamaFactory && llamafactory-cli train \
    ../../config/training/phase1b_pre_experience_qwen35_4b.yaml)                  # 1B
bash scripts/02_cherry_selection.sh config/rgdc_qwen35.yaml                       # 1C
```

| Step | Code | What it does |
|---|---|---|
| 1A | [01_pre_experience.sh](../scripts/01_pre_experience.sh) | Clusters up to 20k records per corpus and direction, takes 10 per cluster from the mid-perplexity band |
| 1B | [phase1b_pre_experience_qwen35_4b.yaml](../config/training/phase1b_pre_experience_qwen35_4b.yaml) | Freeze-tunes the top 4 blocks of the base model for one epoch on the 1A sample |
| 1C | [02_cherry_selection.sh](../scripts/02_cherry_selection.sh) | Scores every record by IFD, drops IFD of 1 or above, keeps the top 10% |

1B writes its checkpoint to the path in `phase1c.pre_exp_model`.

## 3. Phase 2: constraint augmentation

```bash
bash scripts/03_constraint_augmentation.sh config/rgdc_qwen35.yaml all      # or one step, e.g. 4
bash scripts/03_constraint_augmentation.sh config/rgdc_qwen35.yaml status
```

| Step | Paper | Code |
|---|---|---|
| `1`, `1b` | 2A constraint extraction | [generate_constraint.py](../src/augment/generate_constraint.py) |
| `2`, `2b` | 2B yes/no questions | [generate_eval_ques.py](../src/augment/generate_eval_ques.py) |
| `3`, `4` | 2C regeneration under constraints | [pack_augmented_prompts.py](../src/augment/pack_augmented_prompts.py) |
| `5`, `6` | 2D judging each constraint | [pack_evaluation_prompts.py](../src/augment/pack_evaluation_prompts.py) |
| `7` | 2E all-pass filter | [merge_final_dataset.py](../src/augment/merge_final_dataset.py) |
| `8` | `Rules:` block in each instruction | [add_constraints_to_instruction.py](../src/augment/add_constraints_to_instruction.py) |

Steps `1b`, `2b`, `4`, and `6` call the endpoint through [call_endpoint.py](../src/augment/call_endpoint.py).
Finished steps are skipped, and `--force` reruns them.

## Outputs

| File in `phase2.output_dir` | Contents |
|---|---|
| `augmented_sft_dataset.jsonl` | Retained records with constraints and judge verdicts |
| `augmented_sft_dataset_with_rules.jsonl` | Training triples in the format of [Grounded](https://huggingface.co/datasets/iapp/ChindaMT-Grounded), registered as `rgdc_output` |

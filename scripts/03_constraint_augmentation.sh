#!/bin/bash
# Phase 2: Constraint Augmentation
# 8 steps: constraint gen -> eval ques gen -> pack -> infer -> eval -> merge -> add_rules
#
# Usage: bash scripts/03_constraint_augmentation.sh config/rgdc_qwen35.yaml [step] [--force]
# Steps: 1, 1b, 2, 2b, 3, 4, 5, 6, 7, 8, status, all (default)

set -e

CONFIG="${1:?Usage: $0 <config.yaml> [step] [--force]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

eval "$(python3 scripts/parse_config.py "$CONFIG" phase2)"

FORCE=false
STEP="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f) FORCE=true; shift ;;
        *) STEP="$1"; shift ;;
    esac
done

mkdir -p "$output_dir"

INPUT_DATA="$input_file"
CONSTRAINT_PROMPTS="$output_dir/constraint_prompts.jsonl"
CONSTRAINT_RESULTS="$output_dir/constraint_results.jsonl"
EVAL_QUES_PROMPTS="$output_dir/eval_ques_prompts.jsonl"
EVAL_QUES_RESULTS="$output_dir/eval_ques_results.jsonl"
PROMPTS_FILE="$output_dir/augmented_prompts.jsonl"
RESPONSES_FILE="$output_dir/augmented_responses.jsonl"
EVAL_PROMPTS="$output_dir/evaluation_prompts.jsonl"
EVAL_RESULTS="$output_dir/evaluation_results.jsonl"
FINAL_DATASET="$output_dir/augmented_sft_dataset.jsonl"
FINAL_WITH_RULES="$output_dir/augmented_sft_dataset_with_rules.jsonl"

# Use pre-computed constraints if available
if [ -n "$pre_computed_constraints" ] && [ -f "$pre_computed_constraints" ]; then
    CONSTRAINT_RESULTS="$pre_computed_constraints"
fi
if [ -n "$pre_computed_eval_questions" ] && [ -f "$pre_computed_eval_questions" ]; then
    EVAL_QUES_RESULTS="$pre_computed_eval_questions"
fi

check_exists() { [ -f "$1" ] && [ -s "$1" ]; }
check_complete() { [ "$(wc -l < "$1" 2>/dev/null || echo 0)" -eq "$(wc -l < "$2" 2>/dev/null || echo 0)" ]; }

run_step1() {
    echo "[Step 1] Generating constraint prompts..."
    if [ "$FORCE" = false ] && check_exists "$CONSTRAINT_PROMPTS"; then
        echo "[Step 1] SKIPPED (exists)"; return 0; fi
    [ "$FORCE" = true ] && rm -f "$CONSTRAINT_PROMPTS"
    python3 src/augment/generate_constraint.py \
        --input "$INPUT_DATA" --output "$CONSTRAINT_PROMPTS" --model "$llm_model"
}

run_step1b() {
    echo "[Step 1b] Running constraint inference..."
    if [ "$FORCE" = false ] && check_exists "$CONSTRAINT_RESULTS" && check_complete "$CONSTRAINT_PROMPTS" "$CONSTRAINT_RESULTS"; then
        echo "[Step 1b] SKIPPED (complete)"; return 0; fi
    python3 src/augment/call_endpoint.py \
        --input "$CONSTRAINT_PROMPTS" --output "$CONSTRAINT_RESULTS" \
        --base-url "$api_base_url" --api-key "$api_key" \
        --max-concurrent "$max_concurrent" --batch-size "$batch_size" --delay "$delay"
}

run_step2() {
    echo "[Step 2] Generating eval question prompts..."
    if [ "$FORCE" = false ] && check_exists "$EVAL_QUES_PROMPTS"; then
        echo "[Step 2] SKIPPED (exists)"; return 0; fi
    [ "$FORCE" = true ] && rm -f "$EVAL_QUES_PROMPTS"
    python3 src/augment/generate_eval_ques.py \
        --input "$INPUT_DATA" --constraints "$CONSTRAINT_RESULTS" \
        --output "$EVAL_QUES_PROMPTS" --model "$llm_model"
}

run_step2b() {
    echo "[Step 2b] Running eval question inference..."
    if [ "$FORCE" = false ] && check_exists "$EVAL_QUES_RESULTS" && check_complete "$EVAL_QUES_PROMPTS" "$EVAL_QUES_RESULTS"; then
        echo "[Step 2b] SKIPPED (complete)"; return 0; fi
    python3 src/augment/call_endpoint.py \
        --input "$EVAL_QUES_PROMPTS" --output "$EVAL_QUES_RESULTS" \
        --base-url "$api_base_url" --api-key "$api_key" \
        --max-concurrent "$max_concurrent" --batch-size "$batch_size" --delay "$delay"
}

run_step3() {
    echo "[Step 3] Packing augmented prompts..."
    if [ "$FORCE" = false ] && check_exists "$PROMPTS_FILE"; then
        echo "[Step 3] SKIPPED (exists)"; return 0; fi
    python3 src/augment/pack_augmented_prompts.py \
        --input "$INPUT_DATA" --constraints "$CONSTRAINT_RESULTS" \
        --eval-questions "$EVAL_QUES_RESULTS" --output "$PROMPTS_FILE" \
        --min-sample "$min_sample_constraints" --max-sample "$max_sample_constraints" \
        --seed "$seed" --model "$llm_model"
}

run_step4() {
    echo "[Step 4] Running augmented inference..."
    if [ "$FORCE" = false ] && check_exists "$RESPONSES_FILE" && check_complete "$PROMPTS_FILE" "$RESPONSES_FILE"; then
        echo "[Step 4] SKIPPED (complete)"; return 0; fi
    python3 src/augment/call_endpoint.py \
        --input "$PROMPTS_FILE" --output "$RESPONSES_FILE" \
        --base-url "$api_base_url" --api-key "$api_key" \
        --max-concurrent "$max_concurrent" --batch-size "$batch_size" --delay "$delay"
}

run_step5() {
    echo "[Step 5] Packing evaluation prompts..."
    if [ "$FORCE" = false ] && check_exists "$EVAL_PROMPTS"; then
        echo "[Step 5] SKIPPED (exists)"; return 0; fi
    python3 src/augment/pack_evaluation_prompts.py \
        --prompts "$PROMPTS_FILE" --responses "$RESPONSES_FILE" \
        --output "$EVAL_PROMPTS" --model "$llm_model"
}

run_step6() {
    echo "[Step 6] Running evaluation..."
    if [ "$FORCE" = false ] && check_exists "$EVAL_RESULTS" && check_complete "$EVAL_PROMPTS" "$EVAL_RESULTS"; then
        echo "[Step 6] SKIPPED (complete)"; return 0; fi
    python3 src/augment/call_endpoint.py \
        --input "$EVAL_PROMPTS" --output "$EVAL_RESULTS" \
        --base-url "$api_base_url" --api-key "$api_key" \
        --max-concurrent "$max_concurrent" --batch-size "$batch_size" --delay "$delay"
}

run_step7() {
    echo "[Step 7] Merging final dataset..."
    if [ "$FORCE" = false ] && check_exists "$FINAL_DATASET"; then
        echo "[Step 7] SKIPPED (exists)"; return 0; fi
    python3 src/augment/merge_final_dataset.py \
        --prompts "$PROMPTS_FILE" --responses "$RESPONSES_FILE" \
        --evaluations "$EVAL_RESULTS" --output "$FINAL_DATASET" \
        --filter-mode "$filter_mode" --include-failed
}

run_step8() {
    echo "[Step 8] Adding constraints to instruction..."
    if [ "$FORCE" = false ] && check_exists "$FINAL_WITH_RULES"; then
        echo "[Step 8] SKIPPED (exists)"; return 0; fi
    [ "$FORCE" = true ] && rm -f "$FINAL_WITH_RULES"
    python3 src/augment/add_constraints_to_instruction.py \
        --input "$FINAL_DATASET" --output "$FINAL_WITH_RULES"
}

show_status() {
    echo "=== Output Files ==="
    for f in "$CONSTRAINT_PROMPTS" "$CONSTRAINT_RESULTS" "$EVAL_QUES_PROMPTS" "$EVAL_QUES_RESULTS" \
             "$PROMPTS_FILE" "$RESPONSES_FILE" "$EVAL_PROMPTS" "$EVAL_RESULTS" "$FINAL_DATASET" "$FINAL_WITH_RULES"; do
        if [ -f "$f" ]; then
            COUNT=$(wc -l < "$f")
            SIZE=$(du -h "$f" | cut -f1)
            echo "  [ok] $f ($COUNT lines, $SIZE)"
        else
            echo "  [--] $f"
        fi
    done
}

case "$STEP" in
    1)      run_step1 ;;
    1b)     run_step1b ;;
    2)      run_step2 ;;
    2b)     run_step2b ;;
    3)      run_step3 ;;
    4)      run_step4 ;;
    5)      run_step5 ;;
    6)      run_step6 ;;
    7)      run_step7 ;;
    8)      run_step8 ;;
    status) show_status ;;
    all)
        show_status
        run_step1; run_step1b
        run_step2; run_step2b
        run_step3; run_step4
        run_step5; run_step6
        run_step7; run_step8
        echo "Pipeline completed: $FINAL_WITH_RULES"
        show_status
        ;;
    *)
        echo "Usage: $0 <config.yaml> [step] [--force]"
        echo "Steps: 1, 1b, 2, 2b, 3, 4, 5, 6, 7, 8, status, all"
        exit 1 ;;
esac

#!/bin/bash
# Phase 1C: IFD Cherry Selection
# Computes IFD scores using pre-experienced model, selects top 10%.
#
# Usage: bash scripts/02_cherry_selection.sh config/rgdc_qwen35.yaml

set -e

CONFIG="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

eval "$(python3 scripts/parse_config.py "$CONFIG" phase1c)"

mkdir -p "$output_dir"

# Count total corpora for progress tracking
total_corpora=0
done_corpora=0
skipped_corpora=0
while read -r sharegpt_file; do
    total_corpora=$((total_corpora + 1))
done < <(find -L "$source_data_dir" -name "*_sharegpt.json" 2>/dev/null | grep -E "_${template_filter}")

echo "=========================================="
echo "Phase 1C: IFD Cherry Selection"
echo "Total corpora to process: $total_corpora"
echo "=========================================="

OVERALL_START=$(date +%s)

find -L "$source_data_dir" -name "*_sharegpt.json" 2>/dev/null | grep -E "_${template_filter}" | while read -r sharegpt_file; do
    folder_path=$(dirname "$sharegpt_file")
    folder_name=$(basename "$folder_path")
    file_name=$(basename "$sharegpt_file" .json)
    dataset_id="${folder_name}__${file_name}"

    alpaca_file="$output_dir/${dataset_id}_alpaca.json"
    pt_file="$output_dir/${dataset_id}_cherry.pt"
    out_file="$output_dir/${dataset_id}_cherry.json"

    done_corpora=$((done_corpora + 1))

    if [ -f "$out_file" ]; then
        skipped_corpora=$((skipped_corpora + 1))
        echo "[$done_corpora/$total_corpora] Skipping $dataset_id (already processed)"
        continue
    fi

    CORPUS_START=$(date +%s)
    echo ""
    echo "[$done_corpora/$total_corpora] Processing: $dataset_id"

    python3 src/cherry/alpaca_converter.py "$sharegpt_file" \
        -o "$alpaca_file" --format json --seed 42

    python3 src/cherry/data_analysis.py \
        --data_path "$alpaca_file" \
        --save_path "$pt_file" \
        --model_name_or_path "$pre_exp_model" \
        --max_length "$max_length" \
        --prompt "$prompt" \
        --mod cherry \
        --batch_size "$batch_size"

    python3 src/cherry/data_by_IFD.py \
        --model_name_or_path "$pre_exp_model" \
        --pt_data_path "$pt_file" \
        --json_data_path "$alpaca_file" \
        --json_save_path "$out_file" \
        --max_length "$max_length" \
        --sample_rate "$sample_rate" \
        --prompt "$prompt"

    rm -f "$alpaca_file" "$pt_file"

    CORPUS_END=$(date +%s)
    CORPUS_ELAPSED=$((CORPUS_END - CORPUS_START))
    CORPUS_MIN=$((CORPUS_ELAPSED / 60))
    CORPUS_SEC=$((CORPUS_ELAPSED % 60))

    OVERALL_ELAPSED=$((CORPUS_END - OVERALL_START))
    PROCESSED=$((done_corpora - skipped_corpora))
    if [ $PROCESSED -gt 0 ]; then
        AVG_SEC=$((OVERALL_ELAPSED / PROCESSED))
        REMAINING=$(( (total_corpora - done_corpora) * AVG_SEC ))
        REM_H=$((REMAINING / 3600))
        REM_M=$(( (REMAINING % 3600) / 60 ))
        ETA="ETA ~${REM_H}h${REM_M}m"
    else
        ETA=""
    fi

    echo "[$done_corpora/$total_corpora] Completed in ${CORPUS_MIN}m${CORPUS_SEC}s | $ETA"
done

OVERALL_END=$(date +%s)
TOTAL_ELAPSED=$((OVERALL_END - OVERALL_START))
TOTAL_H=$((TOTAL_ELAPSED / 3600))
TOTAL_M=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_S=$((TOTAL_ELAPSED % 60))

echo ""
echo "=========================================="
echo "Concatenating cherry datasets..."
python3 src/cherry/concat_dataset.py \
    --root-dir "$output_dir" \
    --save-path "$output_concat" \
    --file-suffix "_cherry.json"

echo "=========================================="
echo "Phase 1C complete"
echo "Total time: ${TOTAL_H}h${TOTAL_M}m${TOTAL_S}s"
echo "Output: $output_concat"
echo "=========================================="

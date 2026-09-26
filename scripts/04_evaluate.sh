#!/bin/bash
# Pairwise evaluation with AlpacaEval on CoreEval or BroadEval.
#
# Usage:
#   bash scripts/04_evaluate.sh <config.yaml> [options]
#
# Options:
#   --suite NAME              coreeval or broadeval (default: coreeval)
#   --splits plain,constrained  splits to run (default: from config)
#   --references r1,r2        reference models (default: from config)
#   --n-samples N             limit to first N samples per split (smoke test)
#   --skip-inference          assume predictions already exist
#   --target-only             run inference for target model only, skip judge
#   --force                   rerun inference and judge from scratch
#
# Results go to <output_dir>/<judge>/<target>-vs-<reference>/<suite>-<split>/, where <judge> is the
# served judge name plus a fingerprint of the judge prompt and settings (src/eval/judge_id.py).
# References with a prompt_adapter use their own template, which has no Rules block, so they are
# run and judged on the plain split only.

set -e

CONFIG="${1:?Usage: $0 <config.yaml> [options]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Load .env if present (judge backend credentials)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_DIR/.env"
    set +a
fi

SUITE="coreeval"
SPLITS_OVERRIDE=""
REFS_OVERRIDE=""
N_SAMPLES=""
SKIP_INFERENCE="false"
TARGET_ONLY="false"
FORCE="false"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --suite) SUITE="$2"; shift 2 ;;
        --splits) SPLITS_OVERRIDE="$2"; shift 2 ;;
        --references) REFS_OVERRIDE="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --skip-inference) SKIP_INFERENCE="true"; shift ;;
        --target-only) TARGET_ONLY="true"; shift ;;
        --force|-f) FORCE="true"; shift ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --verbose|-v) LOG_LEVEL="INFO"; shift ;;
        --debug) LOG_LEVEL="DEBUG"; shift ;;
        --quiet|-q) LOG_LEVEL="ERROR"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Helper: check if all split outputs exist for a generator; usage: check_predictions_exist GEN SPLIT...
check_predictions_exist() {
    local GEN="$1"; shift
    for SPLIT in "$@"; do
        local OUT="$PREDICTIONS_DIR/${GEN}-${SUITE}-${SPLIT}/model_outputs.json"
        if [[ ! -f "$OUT" ]]; then return 1; fi
    done
    return 0
}

# Parse YAML with Python (simple, avoids yq dependency)
read_config() {
    python3 -c "
import yaml, sys, os
cfg = yaml.safe_load(open('$CONFIG'))
key = '$1'
val = cfg
for k in key.split('.'):
    val = val[k] if isinstance(val, dict) and k in val else val.get(k) if isinstance(val, dict) else None
    if val is None: break
if isinstance(val, list):
    print(','.join(str(x) for x in val))
elif val is not None:
    print(os.path.expandvars(str(val)))
"
}

TARGET_MODEL_CFG=$(read_config target_model)
PREDICTIONS_DIR=$(read_config predictions_dir)
OUTPUT_DIR=$(read_config output_dir)
JUDGE_CONFIG=$(read_config judge_config)

TARGET_NAME=$(python3 -c "import yaml,os; print(yaml.safe_load(open('$TARGET_MODEL_CFG'))['generator_name'])")

# Splits a reference is compared on: plain only when it uses its own prompt template
ref_splits() {
    local ADAPTER=""
    if [[ "$1" == *.yaml ]]; then
        ADAPTER=$(python3 -c "import yaml; print(yaml.safe_load(open('$1')).get('prompt_adapter') or '')")
    fi
    for SPLIT in "${SPLITS[@]}"; do
        if [[ -z "$ADAPTER" || "$SPLIT" == "plain" ]]; then echo "$SPLIT"; fi
    done
}

if [[ -z "$SPLITS_OVERRIDE" ]]; then
    SPLITS_OVERRIDE=$(read_config "suites.${SUITE}.splits")
fi
IFS=',' read -ra SPLITS <<< "$SPLITS_OVERRIDE"

# Build reference list
if [[ -n "$REFS_OVERRIDE" ]]; then
    IFS=',' read -ra REFS <<< "$REFS_OVERRIDE"
else
    REFS_RAW=$(read_config reference_models)
    IFS=',' read -ra REFS <<< "$REFS_RAW"
fi

if [[ "$TARGET_ONLY" == "false" ]]; then
    if [[ -z "${JUDGE_MODEL_NAME:-}" ]]; then
        echo "JUDGE_MODEL_NAME is not set; set the judge endpoint in .env (see .env.example)"; exit 1
    fi
    JUDGE_ID=$(python3 -m src.eval.judge_id --model "$JUDGE_MODEL_NAME" --config "$JUDGE_CONFIG")
    JUDGE_DIR="$OUTPUT_DIR/$JUDGE_ID"
fi

echo "=========================================="
echo "Target model config: $TARGET_MODEL_CFG"
echo "Target generator:    $TARGET_NAME"
echo "Suite:               $SUITE"
echo "Splits:              ${SPLITS[*]}"
echo "References:          ${REFS[*]}"
echo "Predictions dir:     $PREDICTIONS_DIR"
echo "Output dir:          $OUTPUT_DIR"
[[ -n "${JUDGE_ID:-}" ]] && echo "Judge:               $JUDGE_MODEL_NAME (results in $JUDGE_DIR)"
echo "=========================================="

# Step 1: target model inference
if [[ "$SKIP_INFERENCE" == "false" ]]; then
    if [[ "$FORCE" == "false" ]] && check_predictions_exist "$TARGET_NAME" "${SPLITS[@]}"; then
        echo "[Inference] Target SKIPPED (exists): $TARGET_NAME"
    else
        echo ""
        echo "[Inference] Target: $TARGET_NAME"
        python3 -m src.eval.run_inference \
            --config "$CONFIG" \
            --suite "$SUITE" \
            --splits "${SPLITS[@]}" \
            ${N_SAMPLES:+--limit $N_SAMPLES}
    fi
fi

# Step 2: reference model inference (only for references that are YAML configs)
if [[ "$SKIP_INFERENCE" == "false" && "$TARGET_ONLY" == "false" ]]; then
    for REF in "${REFS[@]}"; do
        if [[ "$REF" == *.yaml ]]; then
            REF_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$REF'))['generator_name'])")
            mapfile -t REF_SPLITS < <(ref_splits "$REF")
            [[ ${#REF_SPLITS[@]} -eq 0 ]] && continue
            if [[ "$FORCE" == "false" ]] && check_predictions_exist "$REF_NAME" "${REF_SPLITS[@]}"; then
                echo "[Inference] Reference SKIPPED (exists): $REF_NAME"
                continue
            fi
            echo ""
            echo "[Inference] Reference: $REF_NAME"
            python3 -m src.eval.run_inference \
                --config "$CONFIG" \
                --model-ref "$REF" \
                --suite "$SUITE" \
                --splits "${REF_SPLITS[@]}" \
                ${N_SAMPLES:+--limit $N_SAMPLES}
        fi
    done
fi

if [[ "$TARGET_ONLY" == "true" ]]; then
    echo "[--target-only] Done with target inference only."
    exit 0
fi

# Step 3: judge pairwise
mkdir -p "$JUDGE_DIR"
INCOMPLETE=0

for REF in "${REFS[@]}"; do
    # Resolve reference generator name
    if [[ "$REF" == *.yaml ]]; then
        REF_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$REF'))['generator_name'])")
    else
        REF_NAME="$REF"
    fi
    mapfile -t REF_SPLITS < <(ref_splits "$REF")

    for SPLIT in "${REF_SPLITS[@]}"; do
        TARGET_DIR="$PREDICTIONS_DIR/${TARGET_NAME}-${SUITE}-${SPLIT}"
        REF_DIR="$PREDICTIONS_DIR/${REF_NAME}-${SUITE}-${SPLIT}"

        TARGET_OUT="$TARGET_DIR/model_outputs.json"
        REF_OUT="$REF_DIR/model_outputs.json"

        if [[ ! -f "$TARGET_OUT" ]]; then
            echo "[Judge] Skipping ${SUITE}-${SPLIT}: target predictions missing ($TARGET_OUT)"
            continue
        fi
        if [[ ! -f "$REF_OUT" ]]; then
            echo "[Judge] Skipping ${SUITE}-${SPLIT} vs ${REF_NAME}: reference missing ($REF_OUT)"
            continue
        fi

        COMP_DIR="$JUDGE_DIR/${TARGET_NAME}-vs-${REF_NAME}/${SUITE}-${SPLIT}"
        LEADERBOARD="$COMP_DIR/leaderboard.csv"
        # The cached verdicts live in COMP_DIR, so --force clears it to judge afresh
        [[ "$FORCE" == "true" ]] && rm -rf "$COMP_DIR"

        EXPECTED=${N_SAMPLES:-$(python3 -c "import json; print(len(json.load(open('$TARGET_OUT'))))")}
        if [[ -f "$LEADERBOARD" ]]; then
            if python3 -m src.eval.check_cell "$COMP_DIR" --expected "$EXPECTED" > /dev/null; then
                echo "[Judge] SKIPPED (complete): ${TARGET_NAME} vs ${REF_NAME} on ${SUITE}-${SPLIT}"
                continue
            fi
            echo "[Judge] Incomplete result, judging again from scratch: ${TARGET_NAME} vs ${REF_NAME} on ${SUITE}-${SPLIT}"
            rm -rf "$COMP_DIR"
        fi

        mkdir -p "$COMP_DIR"
        echo ""
        echo "[Judge] ${TARGET_NAME} vs ${REF_NAME} on ${SUITE}-${SPLIT}"
        # Generate judge config with absolute paths (alpaca_eval resolves paths relative to its package)
        ABS_COMP_DIR="$(realpath "$COMP_DIR")"
        RESOLVED_JUDGE="$ABS_COMP_DIR/judge_config.resolved.yaml"
        python3 -c "
import yaml, os
cfg = yaml.safe_load(open('$PROJECT_DIR/$JUDGE_CONFIG'))
for name, annot in cfg.items():
    if isinstance(annot, dict) and 'prompt_template' in annot:
        pt = annot['prompt_template']
        if not os.path.isabs(pt):
            annot['prompt_template'] = os.path.abspath(os.path.join('$PROJECT_DIR', pt))
    if isinstance(annot, dict) and 'completions_kwargs' in annot:
        ck = annot['completions_kwargs']
        if isinstance(ck.get('model_name'), str):
            ck['model_name'] = os.path.expandvars(ck['model_name'])
yaml.safe_dump(cfg, open('$RESOLVED_JUDGE', 'w'), sort_keys=False)
"

        env PYTHONWARNINGS=ignore LOG_LEVEL="$LOG_LEVEL" \
            python3 -c "
import logging, warnings, os, sys
warnings.filterwarnings('ignore')
level = getattr(logging, os.environ.get('LOG_LEVEL','WARNING').upper(), logging.WARNING)

class _Filter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if level > logging.INFO:
            if 'Using OAI client' in msg: return False
            if 'openai_configs.yaml' in msg: return False
            if 'Unknown model' in msg and 'price per token' in msg: return False
        return True

logging.basicConfig(level=level, force=True)
for h in logging.getLogger().handlers:
    h.addFilter(_Filter())
for name in ('root','httpx','httpcore','openai','urllib3','alpaca_eval'):
    lg = logging.getLogger(name)
    lg.setLevel(level)
    for h in lg.handlers:
        h.addFilter(_Filter())

# Monkey-patch getLogger so newly created loggers inherit filter
_orig_getLogger = logging.getLogger
def _patched(name=None):
    lg = _orig_getLogger(name)
    if not any(isinstance(f, _Filter) for f in lg.filters):
        lg.addFilter(_Filter())
    return lg
logging.getLogger = _patched

from alpaca_eval.main import main as ae_main
sys.argv = ['alpaca_eval','evaluate',
  '--model_outputs','$(realpath "$TARGET_OUT")',
  '--reference_outputs','$(realpath "$REF_OUT")',
  '--annotators_config','$RESOLVED_JUDGE',
  '--output_path','$ABS_COMP_DIR'${N_SAMPLES:+,'--max_instances','$N_SAMPLES'}]
ae_main()
"

        # Clean up any annotator temp files
        rm -f "$COMP_DIR"/annotators_*.yaml 2>/dev/null || true
        if ! python3 -m src.eval.check_cell "$COMP_DIR" --expected "$EXPECTED" > /dev/null; then
            echo "[Judge] INCOMPLETE: some pairs have no verdict in $COMP_DIR; rerun to judge it again"
            INCOMPLETE=$((INCOMPLETE + 1))
        fi
    done
done

echo ""
echo "=========================================="
echo "Evaluation done. Results in: $JUDGE_DIR"
echo "=========================================="
[[ "$INCOMPLETE" -eq 0 ]] || { echo "$INCOMPLETE cell(s) incomplete; rerun the same command to judge them again"; exit 1; }

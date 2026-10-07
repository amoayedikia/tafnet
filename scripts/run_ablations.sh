#!/usr/bin/env bash
# Gate, residual and branch ablations of the Temporal Fusion Module (paper Section 5.4).
#
# Trains TAFNet-Full only, five folds per variant, each into its own directory
# under $ABL, reusing the pretrained encoder of the main run. Nothing is written
# to the main results directory. Resumable: a finished variant is skipped, and a
# finished fold is read from the per-directory cache.
#
#   export TAFNET_RESULTS=/path/to/results        # output_dir of the main run
#   export TAFNET_DATA_DIR=/path/to/preprocessed_v2
#   export TAFNET_PAIRS_CSV=/path/to/adni_pairs_ready_pair.csv
#   export TAFNET_PHASE4_CSV=/path/to/adni_phase4_labels.csv
#   bash scripts/run_ablations.sh 2>&1 | tee ablations.log
#
# Run from the repository root. All eight variants are evaluated on the same
# folds and held-out set as the main run, and all eight are reported.

set -u
: "${TAFNET_RESULTS:?set TAFNET_RESULTS}"; : "${TAFNET_DATA_DIR:?set TAFNET_DATA_DIR}"
: "${TAFNET_PAIRS_CSV:?set TAFNET_PAIRS_CSV}"; : "${TAFNET_PHASE4_CSV:?set TAFNET_PHASE4_CSV}"
PY=${PYTHON:-python}
ABL=${TAFNET_ABLATION_DIR:-$TAFNET_RESULTS/ablation}
ENC=$TAFNET_RESULTS/phase4_encoder_best.pth

if [ ! -f "$ENC" ]; then echo "[X] encoder not found at $ENC (run scripts/02_train.py first)"; exit 1; fi
mkdir -p "$ABL"

run () {
  name=$1; shift
  out="$ABL/$name"
  mkdir -p "$out"
  cp -n "$ENC" "$out/" 2>/dev/null
  if [ -f "$out/DONE" ]; then echo "[=] $name already complete, skipping"; return 0; fi
  echo
  echo "======================================================================"
  echo "  VARIANT: $name        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "======================================================================"
  PYTHONPATH=src "$PY" scripts/02_train.py \
    --config configs/default.yaml --no-drive-check --skip-phase4 \
    --override paths.output_dir="$out" \
    --override paths.data_dir="$TAFNET_DATA_DIR" \
    --override paths.pairs_csv="$TAFNET_PAIRS_CSV" \
    --override paths.phase4_csv="$TAFNET_PHASE4_CSV" \
    --override benchmarks.resnet18_single=false \
    --override benchmarks.densenet121_single=false \
    --override benchmarks.siamese_subtract=false \
    --override benchmarks.cnn_lstm=false \
    --override benchmarks.tafnet_initial_only=false \
    --override benchmarks.tafnet_full=true \
    --override phase56.batch_size=4 \
    --override training.accumulation_steps=1 \
    --override training.num_workers=2 \
    "$@" 2>&1 | tee "$out/run.log"
  if [ "$(ls "$out"/folds/TAFNet-Full_fold*.json 2>/dev/null | wc -l)" -eq 5 ]; then
    touch "$out/DONE"; echo "[=] $name complete (5/5 folds)"
  else
    echo "[!] $name did NOT finish 5 folds; re-run this script to resume"
  fi
}

# gate parameterisation and residual
run gate_position     --override architecture.gate_mode=position
run no_residual       --override architecture.baseline_residual=false
# single branch
run branch_diff       --override architecture.branches=difference
run branch_attn       --override architecture.branches=attention
run branch_concat     --override architecture.branches=concat
# leave one branch out
run branch_no_diff    --override architecture.branches=attention,concat
run branch_no_attn    --override architecture.branches=difference,concat
run branch_no_concat  --override architecture.branches=difference,attention

echo
echo "==================== SUMMARY ===================="
for d in "$ABL"/*/; do
  echo "$(basename "$d"): $(ls "$d"/folds/TAFNet-Full_fold*.json 2>/dev/null | wc -l)/5 folds"
done

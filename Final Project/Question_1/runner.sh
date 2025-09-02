#!/usr/bin/env bash
set -euo pipefail

OUTDIR="eval_outputs_sample200"
LOGDIR="logs"
mkdir -p "$OUTDIR" "$LOGDIR"

# تنظیمات مشترک خروجی کوتاه‌تر و بدون سمپلینگ (دلخواه ولی سریع‌تر)
COMMON="--n-samples 200 --max-new-tokens 200 --temperature 0.3 --top-p 0.9 --rep-penalty 1.05 --out-dir $OUTDIR"

echo "[1/3] Soft-Prompt ..."
python evaluate_sample.py --method soft-prompt  $COMMON --log-file $LOGDIR/soft200.log

echo "[2/3] LoRA ..."
python evaluate_sample.py --method lora         $COMMON --log-file $LOGDIR/lora200.log

echo "[3/3] Classic-FT ..."
python evaluate_sample.py --method classic-ft   $COMMON --log-file $LOGDIR/classic200.log

echo "Done. Metrics & predictions saved in $OUTDIR"

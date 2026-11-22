#!/bin/bash
#SBATCH --job-name=medvlm_eval
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   sbatch code/eval/run.sbatch <MODEL_ID> <DATASET> <N> <OUT_TAG>
# Example:
#   sbatch code/eval/run.sbatch Qwen/Qwen2.5-VL-3B-Instruct vqarad 50 smoke
#   sbatch code/eval/run.sbatch google/medgemma-4b-it pathvqa -1 medgemma_pathvqa

set -euo pipefail
cd /lustre/hdd/LAS/weile-lab/howlader/med-reasoner

MODEL="${1:?model id}"; DATASET="${2:?dataset}"; N="${3:--1}"; TAG="${4:?out tag}"
CONDITIONS="${5:-intact,blank,mismatch}"
VERBALIZED="${6:-0}"       # 1 => also elicit verbalized confidence
VFLAG=""; [ "$VERBALIZED" = "1" ] && VFLAG="--verbalized"

export HF_HOME=/lustre/hdd/LAS/weile-lab/howlader/med-reasoner/.hf_cache
export HF_HUB_OFFLINE=1          # compute nodes assumed offline; use pre-cached weights
export TOKENIZERS_PARALLELISM=false

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

echo "[job] model=$MODEL dataset=$DATASET n=$N tag=$TAG on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUT="results/${TAG}.jsonl"
python code/eval/run_eval.py --model "$MODEL" --dataset "$DATASET" \
    --n "$N" --conditions "$CONDITIONS" $VFLAG --out "$OUT"

python code/eval/score.py --pred "$OUT" --out "results/${TAG}_metrics.json" || true
echo "[job] done -> $OUT"

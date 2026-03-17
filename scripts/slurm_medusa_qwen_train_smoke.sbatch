#!/bin/bash
#SBATCH --job-name=medusa-qwen-train-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --partition=A100,ADA6000,L40S
#SBATCH --output=/home/fhshao/Medusa/logs/%x.%j.out
#SBATCH --error=/home/fhshao/Medusa/logs/%x.%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medusa

REPO_DIR="${REPO_DIR:-/home/fhshao/Medusa}"
BASE_MODEL="${BASE_MODEL:-/data1/public/hf/Qwen/Qwen2.5-3B-Instruct}"
DATA_JSON="${DATA_JSON:-data/qwen2p5_3b_medusa1_1k.json}"
OUT_PREFIX="${OUT_PREFIX:-outputs/medusa1_qwen2p5_3b_smoke}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

cd "${REPO_DIR}"
mkdir -p logs outputs

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model path not found: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${DATA_JSON}" ]]; then
  echo "Training data not found: ${DATA_JSON}" >&2
  exit 1
fi

echo "Validating training data before launch ..."
if ! python extension/medusa_qwen_smoke_data_ops.py validate --input "${DATA_JSON}"; then
  echo "Training data is invalid. Re-run bash scripts/submit_medusa_qwen_smoke.sh --force-regenerate" >&2
  exit 1
fi

echo "Starting Medusa-1 smoke training ..."
torchrun --nproc_per_node=4 medusa/train/train_legacy.py \
  --model_name_or_path "${BASE_MODEL}" \
  --data_path "${DATA_JSON}" \
  --bf16 True \
  --output_dir "${OUT_PREFIX}" \
  --max_steps 20 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 1024 \
  --lazy_preprocess True \
  --medusa_num_heads 3 \
  --medusa_num_layers 1 \
  --deepspeed deepspeed.json

echo "Training finished."

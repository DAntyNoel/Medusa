#!/bin/bash
#SBATCH --job-name=medusa-qwen-data-1k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=A100,ADA6000,L40S
#SBATCH --output=/home/fhshao/Medusa/logs/%x.%j.out
#SBATCH --error=/home/fhshao/Medusa/logs/%x.%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

REPO_DIR="${REPO_DIR:-/home/fhshao/Medusa}"
BASE_MODEL="${BASE_MODEL:-/data1/public/hf/Qwen/Qwen2.5-3B-Instruct}"
RAW_JSON="${RAW_JSON:-ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
SUBSET_JSON="${SUBSET_JSON:-data/sharegpt_1k_seed42.json}"
OUT_JSON="${OUT_JSON:-data/qwen2p5_3b_medusa1_1k.json}"
VLLM_PORT="${VLLM_PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-32}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${REPO_DIR}"
mkdir -p logs data

if [[ ! -f "${RAW_JSON}" ]]; then
  echo "Input dataset not found: ${RAW_JSON}" >&2
  exit 1
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model path not found: ${BASE_MODEL}" >&2
  exit 1
fi

echo "[1/4] Sampling 1000 examples from ShareGPT ..."
conda run -n medusa --no-capture-output python extension/medusa_qwen_smoke_data_ops.py \
  sample \
  --input "${RAW_JSON}" \
  --output "${SUBSET_JSON}" \
  --num-samples 1000 \
  --seed 42

echo "[2/4] Starting vLLM server on GPU ..."
VLLM_LOG="logs/vllm-${SLURM_JOB_ID:-local}.log"
conda run -n verl --no-capture-output python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --served-model-name tgi \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  if ps -p "${VLLM_PID}" >/dev/null 2>&1; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    echo "vLLM is ready"
    break
  fi
  sleep 5
done

curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null

echo "[3/4] Generating Qwen-style training data ..."
# create_data.py uses httpx env proxies at import time; clear all proxy vars first.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

conda run -n medusa --no-capture-output python create_data.py \
  --input-filename "${SUBSET_JSON}" \
  --output-filename "${OUT_JSON}" \
  --url "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
  --concurrency "${CONCURRENCY}"

echo "[4/4] Validating generated JSON ..."
conda run -n medusa --no-capture-output python extension/medusa_qwen_smoke_data_ops.py \
  validate \
  --input "${OUT_JSON}"

echo "Done. Output: ${OUT_JSON}"

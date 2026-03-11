#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_JOB_SCRIPT="${SCRIPT_DIR}/slurm_medusa_qwen_data_1k.sbatch"
TRAIN_JOB_SCRIPT="${SCRIPT_DIR}/slurm_medusa_qwen_train_smoke.sbatch"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH" >&2
  exit 1
fi

PARTITION="${1:-A100,ADA6000,L40S}"
REPO_DIR="${REPO_DIR:-/home/fhshao/Medusa}"
OUT_JSON="${OUT_JSON:-data/qwen2p5_3b_medusa1_1k.json}"

if [[ "${OUT_JSON}" = /* ]]; then
  target_dataset="${OUT_JSON}"
else
  target_dataset="${REPO_DIR}/${OUT_JSON}"
fi

if [[ ! -f "${DATA_JOB_SCRIPT}" || ! -f "${TRAIN_JOB_SCRIPT}" ]]; then
  echo "Missing sbatch scripts under ${SCRIPT_DIR}" >&2
  exit 1
fi

if [[ -f "${target_dataset}" ]]; then
  echo "Detected existing dataset: ${target_dataset}"
  echo "Skip data generation/inference stage and submit Medusa training directly."

  train_submit_output="$(DATA_JSON="${OUT_JSON}" sbatch -p "${PARTITION}" "${TRAIN_JOB_SCRIPT}")"
  train_job_id="$(awk '{print $4}' <<< "${train_submit_output}")"
  if [[ -z "${train_job_id}" ]]; then
    echo "Failed to parse train job id from: ${train_submit_output}" >&2
    exit 1
  fi

  echo "Submitted train job: ${train_job_id} (direct submit)"
  echo "Track status with: sacct -j ${train_job_id}"
else
  echo "Dataset not found: ${target_dataset}"
  echo "Submit data generation/inference stage first, then Medusa training."

  data_submit_output="$(OUT_JSON="${OUT_JSON}" sbatch -p "${PARTITION}" "${DATA_JOB_SCRIPT}")"
  data_job_id="$(awk '{print $4}' <<< "${data_submit_output}")"
  if [[ -z "${data_job_id}" ]]; then
    echo "Failed to parse data job id from: ${data_submit_output}" >&2
    exit 1
  fi

  train_submit_output="$(DATA_JSON="${OUT_JSON}" sbatch -p "${PARTITION}" --dependency=afterok:"${data_job_id}" "${TRAIN_JOB_SCRIPT}")"
  train_job_id="$(awk '{print $4}' <<< "${train_submit_output}")"
  if [[ -z "${train_job_id}" ]]; then
    echo "Failed to parse train job id from: ${train_submit_output}" >&2
    exit 1
  fi

  echo "Submitted data job:  ${data_job_id}"
  echo "Submitted train job: ${train_job_id} (afterok:${data_job_id})"
  echo "Track status with: sacct -j ${data_job_id},${train_job_id}"
fi

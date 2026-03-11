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

if [[ ! -f "${DATA_JOB_SCRIPT}" || ! -f "${TRAIN_JOB_SCRIPT}" ]]; then
  echo "Missing sbatch scripts under ${SCRIPT_DIR}" >&2
  exit 1
fi

data_submit_output="$(sbatch -p "${PARTITION}" "${DATA_JOB_SCRIPT}")"
data_job_id="$(awk '{print $4}' <<< "${data_submit_output}")"
if [[ -z "${data_job_id}" ]]; then
  echo "Failed to parse data job id from: ${data_submit_output}" >&2
  exit 1
fi

train_submit_output="$(sbatch -p "${PARTITION}" --dependency=afterok:"${data_job_id}" "${TRAIN_JOB_SCRIPT}")"
train_job_id="$(awk '{print $4}' <<< "${train_submit_output}")"
if [[ -z "${train_job_id}" ]]; then
  echo "Failed to parse train job id from: ${train_submit_output}" >&2
  exit 1
fi

echo "Submitted data job:  ${data_job_id}"
echo "Submitted train job: ${train_job_id} (afterok:${data_job_id})"
echo "Track status with: sacct -j ${data_job_id},${train_job_id}"

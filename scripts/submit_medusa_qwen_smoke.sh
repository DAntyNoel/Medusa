#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_JOB_SCRIPT="${SCRIPT_DIR}/slurm_medusa_qwen_data_1k.sbatch"
TRAIN_JOB_SCRIPT="${SCRIPT_DIR}/slurm_medusa_qwen_train_smoke.sbatch"
DEFAULT_PARTITION="A100,ADA6000,L40S"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/submit_medusa_qwen_smoke.sh [PARTITION] [--force-regenerate]

Examples:
  bash scripts/submit_medusa_qwen_smoke.sh
  bash scripts/submit_medusa_qwen_smoke.sh A100 --force-regenerate
EOF
}

PARTITION="${DEFAULT_PARTITION}"
FORCE_REGENERATE=0
POSITIONAL_PARTITION_SET=0
for arg in "$@"; do
  case "${arg}" in
    --force-regenerate)
      FORCE_REGENERATE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: ${arg}" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ "${POSITIONAL_PARTITION_SET}" -eq 1 ]]; then
        echo "Unexpected extra argument: ${arg}" >&2
        usage >&2
        exit 1
      fi
      PARTITION="${arg}"
      POSITIONAL_PARTITION_SET=1
      ;;
  esac
done

REPO_DIR="${REPO_DIR:-/home/fhshao/Medusa}"
OUT_JSON="${OUT_JSON:-data/qwen2p5_3b_medusa1_1k.json}"
VALIDATE_SCRIPT="${REPO_DIR}/extension/medusa_qwen_smoke_data_ops.py"

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${REPO_DIR}/${path}"
  fi
}

parse_job_id() {
  awk '{print $4}' <<< "$1"
}

validate_dataset() {
  local dataset_path="$1"
  local output
  if output="$(conda run -n medusa --no-capture-output python "${VALIDATE_SCRIPT}" validate --input "${dataset_path}" 2>&1)"; then
    printf '%s\n' "${output}"
    return 0
  fi

  printf '%s\n' "${output}" >&2
  return 1
}

submit_data_job() {
  REPO_DIR="${REPO_DIR}" OUT_JSON="${OUT_JSON}" sbatch -p "${PARTITION}" "${DATA_JOB_SCRIPT}"
}

submit_train_job() {
  local dependency="${1:-}"
  if [[ -n "${dependency}" ]]; then
    REPO_DIR="${REPO_DIR}" DATA_JSON="${OUT_JSON}" sbatch -p "${PARTITION}" --dependency=afterok:"${dependency}" "${TRAIN_JOB_SCRIPT}"
  else
    REPO_DIR="${REPO_DIR}" DATA_JSON="${OUT_JSON}" sbatch -p "${PARTITION}" "${TRAIN_JOB_SCRIPT}"
  fi
}

target_dataset="$(resolve_path "${OUT_JSON}")"

if [[ ! -f "${DATA_JOB_SCRIPT}" || ! -f "${TRAIN_JOB_SCRIPT}" ]]; then
  echo "Missing sbatch scripts under ${SCRIPT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${VALIDATE_SCRIPT}" ]]; then
  echo "Missing validation script: ${VALIDATE_SCRIPT}" >&2
  exit 1
fi

need_regenerate=0
if [[ "${FORCE_REGENERATE}" -eq 1 ]]; then
  echo "Force regenerate requested for dataset: ${target_dataset}"
  need_regenerate=1
elif [[ ! -f "${target_dataset}" ]]; then
  echo "Dataset not found: ${target_dataset}"
  need_regenerate=1
else
  echo "Validating existing dataset: ${target_dataset}"
  if validate_dataset "${target_dataset}"; then
    echo "Existing dataset passed validation. Submit Medusa training directly."
  else
    echo "Existing dataset is invalid. Regenerating before training."
    need_regenerate=1
  fi
fi

if [[ "${need_regenerate}" -eq 1 ]]; then
  data_submit_output="$(submit_data_job)"
  data_job_id="$(parse_job_id "${data_submit_output}")"
  if [[ -z "${data_job_id}" ]]; then
    echo "Failed to parse data job id from: ${data_submit_output}" >&2
    exit 1
  fi

  train_submit_output="$(submit_train_job "${data_job_id}")"
  train_job_id="$(parse_job_id "${train_submit_output}")"
  if [[ -z "${train_job_id}" ]]; then
    echo "Failed to parse train job id from: ${train_submit_output}" >&2
    exit 1
  fi

  echo "Submitted data job:  ${data_job_id}"
  echo "Submitted train job: ${train_job_id} (afterok:${data_job_id})"
  echo "Track status with: sacct -j ${data_job_id},${train_job_id}"
else
  train_submit_output="$(submit_train_job)"
  train_job_id="$(parse_job_id "${train_submit_output}")"
  if [[ -z "${train_job_id}" ]]; then
    echo "Failed to parse train job id from: ${train_submit_output}" >&2
    exit 1
  fi

  echo "Submitted train job: ${train_job_id} (direct submit)"
  echo "Track status with: sacct -j ${train_job_id}"
fi

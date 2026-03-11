# Medusa-1 Qwen2.5-3B-Instruct Smoke Reproduction (Slurm)

This folder contains a two-stage Slurm workflow:

1. `slurm_medusa_qwen_data_1k.sbatch`
Generates 1k Qwen-style chat data using vLLM (GPU) and `create_data.py`.

2. `slurm_medusa_qwen_train_smoke.sbatch`
Runs Medusa-1 smoke training for 20 steps on 4 GPUs.

3. `submit_medusa_qwen_smoke.sh`
Submits stage-1 first, then stage-2 with `afterok` dependency.

## Defaults

- `BASE_MODEL=/data1/public/hf/Qwen/Qwen2.5-3B-Instruct`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- Partition default: `A100,ADA6000,L40S`

## Important details

- `create_data.py` imports `httpx.AsyncClient` at module import time. If environment proxy vars contain unsupported schemes (for example `socks://...`), it fails immediately.
- The data job script clears all proxy variables before running `create_data.py`:
  - `unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy`
- Python helper logic used by the data job is extracted to:
  - `extension/medusa_qwen_smoke_data_ops.py`

## Usage

Submit with default partition:

```bash
bash scripts/submit_medusa_qwen_smoke.sh
```

Submit with an explicit partition:

```bash
bash scripts/submit_medusa_qwen_smoke.sh A100
```

## Optional overrides

You can override variables at submit time:

```bash
BASE_MODEL=/path/to/local/model \
OUT_JSON=data/custom_qwen_smoke.json \
sbatch scripts/slurm_medusa_qwen_data_1k.sbatch
```

```bash
DATA_JSON=data/custom_qwen_smoke.json \
OUT_PREFIX=outputs/custom_medusa_smoke \
sbatch scripts/slurm_medusa_qwen_train_smoke.sbatch
```

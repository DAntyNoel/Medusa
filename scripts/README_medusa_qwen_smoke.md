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

Force data regeneration even if an existing dataset is present:

```bash
bash scripts/submit_medusa_qwen_smoke.sh A100 --force-regenerate
```

## Direct inference

Run one-shot direct inference from a local smoke checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m medusa.inference.cli \
  --model outputs/medusa1_qwen2p5_3b_smoke_medusa_mlp_Qwen2.5-3B-Instruct_medusa_3_lr_0.001_layers_1 \
  --base-model /data1/public/hf/Qwen/Qwen2.5-3B-Instruct \
  --prompt "用一句话介绍 Medusa" \
  --max-steps 64
```

For local Qwen smoke checkpoints, the CLI currently falls back to base-model generation for one-shot direct inference when Medusa accelerated loading is unavailable.

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

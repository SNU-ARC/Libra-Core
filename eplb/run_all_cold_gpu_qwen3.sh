#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS="qwen3"
DATASETS=(bookcorpus codeforces deepseek_prover fineweb gsm8k hellaswag humanevalplus lmsys_chat_1m)
NUM_GPUS=8
RS=(8 16)

if [[ "$MODEL_ALIAS" == "qwen3" ]]; then
  START_LAYER=0
  NUM_LAYERS=94
  NUM_EXPERTS=128
elif [[ "$MODEL_ALIAS" == "glm45" ]]; then
  START_LAYER=3
  NUM_LAYERS=92
  NUM_EXPERTS=160
else
  echo "ERROR: unsupported MODEL '$MODEL_ALIAS'. Expected qwen3 or glm45." >&2
  exit 2
fi

for dataset in "${DATASETS[@]}"; do
  for r in "${RS[@]}"; do
    bash create_expert_placement_cold_gpu.sh ${MODEL_ALIAS} ${START_LAYER} ${NUM_LAYERS} ${NUM_GPUS} ${NUM_EXPERTS} "$r" "$dataset"
  done
done
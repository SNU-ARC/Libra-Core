#!/usr/bin/env bash
# One prefill throughput run of the Lina baseline with Qwen3-235B-A22B on 8 GPUs (paper Fig. 8 / Fig. 9).
# Run from the Libra-Core root (iclr2026_lina branch):
#   bash single_node_scripts/ep_test_qwen3.sh MODEL INPUT_LEN BS EP DATASET MODE START_PORTION END_PORTION \
#        START_IDX END_IDX N SEQ_LENS_SUM
#   MODEL: Qwen3-235B-A22B, loaded from ${LIBRA_MODEL_DIR:-$HOME}/MODEL
#   N: Lina's replicated experts per GPU; SEQ_LENS_SUM = INPUT_LEN * BS
#   Expert-selection-path tables (built by Libra-Internal lina/run_make_expert_path_table_qwen3.sh):
#     ${LIBRA_ARTIFACT_DIR:-$HOME/libra_artifacts}/lina_epts/qwen3_mixed_ept
# Results are appended to latency_throughput/${MODEL}_lina_${DATASET}_${N}_${SEQ_LENS_SUM}.txt
MODEL=$1
INPUT_LEN=$2
BS=$3
MOE_LAYER=$4
DATASET=$5
MODE=$6
START_PORTION=$7
END_PORTION=$8
START_IDX=$9
END_IDX=${10}
N=${11}
SEQ=${12}
export EXPERT_DUP_FACTOR_N=${11}
export SEQ_LENS_SUM=${12}
MODEL_ALIAS="qwen3"

# This branch runs the Lina code path of the Qwen3-MoE / GLM-4.5 decoder layers.
export LIBRA_SCHEME=lina

# Repository root (this script lives in <root>/single_node_scripts).
SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SGLANG_TORCH_PROFILER_DIR=$SRC_DIR/profile_logs/$MODEL/$MOE_LAYER

export SGLANG_TORCH_PROFILER_DIR=$SGLANG_TORCH_PROFILER_DIR
export SGL_ENABLE_JIT_DEEPGEMM=0

export TRITON_CACHE_DIR=$SRC_DIR/triton_cache
export TRITON_DISABLE_COMPILATION=1
export TRITON_DISABLE_AUTOTUNING=1
export TRITON_CACHE_USE_PID=0
export TORCHINDUCTOR_CACHE_DIR=$TRITON_CACHE_DIR

export TRITON_ABORT_ON_CACHE_MISS=1
export TRITON_ALWAYS_COMPILE=0

# Lina expert-selection-path tables (from Libra-Internal) and the cache of their preprocessed form.
# A leading ~ in LIBRA_ARTIFACT_DIR is expanded here (the cache directory is created by Python).
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR:-$HOME/libra_artifacts}
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR/#\~/$HOME}
export LINA_EPT_DIR=$LIBRA_ARTIFACT_DIR/lina_epts/${MODEL_ALIAS}_mixed_ept
export LINA_CACHE_DIR=$LIBRA_ARTIFACT_DIR/lina_epts/${MODEL_ALIAS}_${DATASET}_cache

mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p latency_throughput

MODEL_PATH=${LIBRA_MODEL_DIR:-$HOME}/$MODEL

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} python3 -m sglang.bench_one_batch_test \
    --model-path $MODEL_PATH \
    --mem-fraction-static 0.9 \
    --batch-size $BS \
    --input-len $INPUT_LEN \
    --output-len 0 \
    --trust-remote-code \
    --tp-size 8 \
    --enable-dp-attention \
    --dp-size 8 \
    --enable-ep-moe \
    --ep-size 8 \
    --attention-backend fa3 \
    --enable-torch-compile \
    --torch-compile-max-bs $BS \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --chunked-prefill-size -1 \
    --disable-cuda-graph \
    --dataset $DATASET \
    --mode $MODE \
    --start-portion $START_PORTION \
    --end-portion $END_PORTION \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --result-filename "latency_throughput/${MODEL}_lina_${DATASET}_${N}_${SEQ}.txt" \
    --is-real

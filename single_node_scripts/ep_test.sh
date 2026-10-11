#!/usr/bin/env bash
# One prefill throughput run of the SGLang baseline or Libra on 8 GPUs (paper Fig. 8 / Fig. 9).
# Run from the Libra-Core root:
#   bash single_node_scripts/ep_test.sh MODEL INPUT_LEN BS EP DATASET MODE START_PORTION END_PORTION \
#        START_IDX END_IDX N L SEQ_LENS_SUM SCHEME
#   MODEL: Qwen3-235B-A22B | GLM-4.5, loaded from ${LIBRA_MODEL_DIR:-$HOME}/MODEL
#   N, L: Libra's replicated / local-hot experts per GPU; SEQ_LENS_SUM = INPUT_LEN * BS
#   SCHEME: sglang | libra (exported as LIBRA_SCHEME)
# Results are appended to latency_throughput/${MODEL}_${SCHEME}_${DATASET}_${SEQ_LENS_SUM}.txt
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
L=${12}
SEQ=${13}
SCHEME=${14}
export EXPERT_DUP_FACTOR_N=${11}
export EXPERT_DUP_FACTOR_L=${12}
export SEQ_LENS_SUM=${13}

# SCHEME selects the code path of the Qwen3-MoE / GLM-4.5 decoder layers and names the result file.
case "$SCHEME" in
  sglang|libra) export LIBRA_SCHEME=$SCHEME ;;
  *) echo "ERROR: SCHEME must be 'sglang' or 'libra', got '$SCHEME' (use eplb_test.sh for EPLB)." >&2; exit 2 ;;
esac

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

mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p latency_throughput

MODEL_PATH=${LIBRA_MODEL_DIR:-$HOME}/$MODEL

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} python3 -m sglang.bench_one_batch_test \
    --model-path $MODEL_PATH \
    --mem-fraction-static 0.8 \
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
    --result-filename "latency_throughput/${MODEL}_${SCHEME}_${DATASET}_${SEQ}.txt" \
    --is-real

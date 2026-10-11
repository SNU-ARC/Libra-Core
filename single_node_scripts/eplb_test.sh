#!/usr/bin/env bash
# One prefill throughput run of the EPLB baseline on 8 GPUs (paper Fig. 8).
# Run from the Libra-Core root, after generating the placements with eplb/run_all_cold_gpu_*.sh:
#   bash single_node_scripts/eplb_test.sh MODEL INPUT_LEN BS EP DATASET MODE START_PORTION END_PORTION \
#        START_IDX END_IDX D SEQ_LENS_SUM
#   MODEL: Qwen3-235B-A22B | GLM-4.5, loaded from ${LIBRA_MODEL_DIR:-$HOME}/MODEL
#   D: number of redundant experts per layer; placement eplb/expert_placement/<alias>/D/DATASET.json
# Results are appended to latency_throughput/${MODEL}_eplb_${D}_${DATASET}_${SEQ_LENS_SUM}.txt
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
D=${11}
SEQ=${12}
export SEQ_LENS_SUM=${12}

# EPLB uses the baseline code path of the Qwen3-MoE / GLM-4.5 decoder layers.
export LIBRA_SCHEME=eplb

# Expert placements are generated per model alias under eplb/expert_placement/<alias>/.
case "$MODEL" in
  Qwen3-235B-A22B) MODEL_ALIAS=qwen3 ;;
  GLM-4.5) MODEL_ALIAS=glm45 ;;
  *) echo "ERROR: unsupported MODEL '$MODEL'. Expected Qwen3-235B-A22B or GLM-4.5." >&2; exit 2 ;;
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
    --init-expert-location ./eplb/expert_placement/${MODEL_ALIAS}/$D/${DATASET}.json \
    --ep-dispatch-algorithm dynamic \
    --ep-num-redundant-experts $D \
    --result-filename "latency_throughput/${MODEL}_eplb_${D}_${DATASET}_${SEQ}.txt" \
    --is-real

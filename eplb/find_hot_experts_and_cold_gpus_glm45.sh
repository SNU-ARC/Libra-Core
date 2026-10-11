MODEL_ALIAS=glm45
DATASET=mixed
if [[ "$MODEL_ALIAS" == "qwen3" ]]; then
  LAYER_IDXES=$(seq 0 93)
  NUM_EXPERTS=128
elif [[ "$MODEL_ALIAS" == "glm45" ]]; then
  LAYER_IDXES=$(seq 3 91)
  NUM_EXPERTS=160
else
  echo "ERROR: unsupported MODEL '$MODEL_ALIAS'. Expected qwen3 or glm45." >&2
  exit 2
fi
MAX_NUM_BATCHES=1000
# TEMT generated in Libra-Internal (generate_temt_*.sh), under $LIBRA_ARTIFACT_DIR (default ~/libra_artifacts)
LIBRA_ARTIFACT_DIR=${LIBRA_ARTIFACT_DIR:-$HOME/libra_artifacts}
TEMT_DIR=$LIBRA_ARTIFACT_DIR/expt_token2expert_results/${MODEL_ALIAS}_${DATASET}_train/${MAX_NUM_BATCHES}
SAVE_DIR=hot_experts/${MODEL_ALIAS}
START_IDX=0
END_IDX=40960
NUM_GPUS=8
SEQ_LEN=2048

python find_hot_experts_and_cold_gpus.py \
    --temt-dir $TEMT_DIR \
    --save-dir $SAVE_DIR \
    --model-alias $MODEL_ALIAS \
    --dataset $DATASET \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --layer-idxes $LAYER_IDXES \
    --num-experts $NUM_EXPERTS \
    --num-gpus $NUM_GPUS \
    --seq-len $SEQ_LEN \

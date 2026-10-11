MODEL=$1
START_LAYER=$2
NUM_LAYERS=$3
NUM_GPUS=$4
NUM_ROUTED_EXPERTS=$5
NUM_REDUNDANT_EXPERTS=$6
DATASET=$7
 
python ./create_expert_placement_with_r_cold_gpu.py \
    --start_layer $START_LAYER \
    --num_layers $NUM_LAYERS \
    --num_gpus $NUM_GPUS \
    --num_routed_experts $NUM_ROUTED_EXPERTS \
    --hot_experts_file ./hot_experts/$MODEL/${DATASET}_order_rates.json \
    --num_redundant_experts $NUM_REDUNDANT_EXPERTS \
    --output_filename ./expert_placement/$MODEL/$NUM_REDUNDANT_EXPERTS/$DATASET
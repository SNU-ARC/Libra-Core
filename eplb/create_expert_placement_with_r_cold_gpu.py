import torch
import argparse
import json
from pathlib import Path

def create_expert_placement_with_redundancy(
    start_layer: int,
    num_layers: int, 
    num_gpus: int = 8, 
    num_logical_experts: int = 128,
    hot_experts_data: dict = None,
    num_redundant_experts: int = 4
):
    """
    Build the EPLB physical-to-logical expert map for layers [start_layer, num_layers).

    Every GPU keeps its contiguous block of experts plus num_redundant_experts / num_gpus extra
    slots. Replica i goes to GPU cold_gpus[i % num_gpus] and holds the hottest remaining expert
    (from the `order` list of the hot-experts file) that is not resident on that GPU.
    """
    
    if num_logical_experts % num_gpus != 0:
        raise ValueError(
            f"num_logical_experts ({num_logical_experts}) must be divisible by "
            f"num_gpus ({num_gpus})"
        )
    
    experts_per_gpu = num_logical_experts // num_gpus  # e.g., 128/4 = 32
    
    # Calculate total slots needed per GPU
    assert num_redundant_experts % num_gpus == 0
    redundant_per_gpu = num_redundant_experts // num_gpus
    total_slots_per_gpu = experts_per_gpu + redundant_per_gpu
    num_physical_experts = num_gpus * total_slots_per_gpu
    
    # Initialize with -1 to indicate empty slots
    physical_to_logical_map = torch.full(
        (num_layers, num_physical_experts), -1, dtype=torch.long
    )
    
    # Process each layer
    for layer_idx in range(start_layer, num_layers):
        # Step 1: Place base experts (same for all layers)
        for gpu_idx in range(num_gpus):
            gpu_start_slot = gpu_idx * total_slots_per_gpu
            
            # Each GPU gets a consecutive range of experts
            for local_idx in range(experts_per_gpu):
                expert_id = gpu_idx * experts_per_gpu + local_idx
                physical_to_logical_map[layer_idx, gpu_start_slot + local_idx] = expert_id
        
        # Step 2: Place redundant experts if hot experts data is available
        if hot_experts_data:
            try:
                # Auto-detect the dataset name (first key in the JSON)
                dataset_name = list(hot_experts_data.keys())[0]
                layers_data = hot_experts_data[dataset_name]["layers"]
                
                if layer_idx - start_layer < len(layers_data):
                    layer_data = layers_data[layer_idx - start_layer]
                    
                    # Try to find the appropriate top-k key
                    key = f"order"
                    if key not in layer_data:
                        # Try other possible keys
                        available_keys = [k for k in layer_data.keys() if k.endswith("_ids")]
                        for k in available_keys:
                            if k.startswith("top"):
                                num_in_key = int(k[3:k.index("_")])
                                if num_in_key >= num_redundant_experts:
                                    hot_expert_ids = layer_data[k].copy()
                                    break
                    else:
                        hot_expert_ids = layer_data[key].copy()

                    # Track redundant placements per GPU for this layer
                    redundant_count = [0] * num_gpus
                    cold_gpus = layer_data["cold_gpus"]

                    duplicate_count = 0
                    for i in range(num_redundant_experts):
                        cold_gpu_idx = i % num_gpus
                        for hot_expert_idx in range(len(hot_expert_ids)):
                            hot_expert = hot_expert_ids[hot_expert_idx]
                            cold_gpu = cold_gpus[cold_gpu_idx]
                            if hot_expert == -1:
                                continue
                            if hot_expert // experts_per_gpu == cold_gpu:
                                continue
                            
                            gpu_start_slot = cold_gpu * total_slots_per_gpu
                            redundant_offset = experts_per_gpu + redundant_count[cold_gpu]
                            slot_idx = gpu_start_slot + redundant_offset

                            if slot_idx < (cold_gpu + 1) * total_slots_per_gpu:
                                physical_to_logical_map[layer_idx, slot_idx] = hot_expert
                                redundant_count[cold_gpu] += 1
                                hot_expert_ids[hot_expert_idx] = -1
                                duplicate_count += 1
                                break

            except Exception as e:
                print(f"Warning at layer {layer_idx}: {e}")
    
    return physical_to_logical_map

def visualize_pattern_with_redundancy(
    physical_to_logical_map, num_gpus: int = 2, num_redundant_experts: int = 4
):
    """
    Properly visualize the expert distribution pattern.
    """
    num_layers = physical_to_logical_map.shape[0]
    total_slots = physical_to_logical_map.shape[1]
    slots_per_gpu = total_slots // num_gpus
    
    # Calculate the actual number of base experts per GPU
    # Find where the first -1 appears or where expert IDs break sequence
    experts_per_gpu = 0
    for i in range(slots_per_gpu):
        if physical_to_logical_map[0, i] < 0:
            experts_per_gpu = i
            break
        if i > 0 and physical_to_logical_map[0, i] != physical_to_logical_map[0, i-1] + 1:
            experts_per_gpu = i
            break
    
    if experts_per_gpu == 0:  # No break found, use calculated value
        experts_per_gpu = 128 // num_gpus  # Use known total experts / num_gpus
    
    print(f"\nExpert Assignment Pattern - All {num_layers} Layers")
    print(f"Total redundant experts per layer: {num_redundant_experts}")
    print(f"Each GPU has {experts_per_gpu} base experts + space for redundant experts")
    print("=" * (42 * min(num_gpus, 8) + 10))
    
    # Print header
    header_parts = [f"{'Layer':^7}"]
    for gpu_id in range(min(num_gpus, 8)):
        header_parts.append(f"{'GPU ' + str(gpu_id) + ' Experts':^38}")
    print(" | ".join(header_parts))
    print("-" * (42 * min(num_gpus, 8) + 10))
    
    # Process and display each layer
    for layer_idx in range(num_layers):
        row_parts = [f"{layer_idx:^7}"]
        
        for gpu_idx in range(min(num_gpus, 8)):
            gpu_start = gpu_idx * slots_per_gpu
            
            # Extract base experts (first experts_per_gpu slots)
            base_experts = []
            for i in range(experts_per_gpu):
                val = physical_to_logical_map[layer_idx, gpu_start + i].item()
                if val >= 0:
                    base_experts.append(val)
            
            # Extract redundant experts (remaining slots)
            redundant_experts = []
            for i in range(experts_per_gpu, slots_per_gpu):
                val = physical_to_logical_map[layer_idx, gpu_start + i].item()
                if val >= 0:
                    redundant_experts.append(val)
            
            # Format the display string
            if base_experts:
                if len(base_experts) == experts_per_gpu and base_experts[-1] - base_experts[0] == experts_per_gpu - 1:
                    # Consecutive range
                    base_str = f"[{base_experts[0]}-{base_experts[-1]}]"
                else:
                    # Not consecutive or incomplete
                    base_str = str(base_experts)[:20]
            else:
                base_str = "[]"
            
            if redundant_experts:
                display_str = f"{base_str:20s} +R{redundant_experts}"
            else:
                display_str = f"{base_str:20s}"
            
            row_parts.append(f"{display_str:^38}")
        
        print(" | ".join(row_parts))
    
    # Statistics
    print("\n" + "=" * (42 * min(num_gpus, 8) + 10))
    print("Summary Statistics:")
    
    # Count total redundant experts per GPU
    gpu_redundant_totals = [0] * num_gpus
    
    for layer_idx in range(num_layers):
        for gpu_idx in range(num_gpus):
            gpu_start = gpu_idx * slots_per_gpu
            count = 0
            for i in range(experts_per_gpu, slots_per_gpu):
                if physical_to_logical_map[layer_idx, gpu_start + i] >= 0:
                    count += 1
            gpu_redundant_totals[gpu_idx] += count
    
    print(f"\nRedundant experts per GPU across all {num_layers} layers:")
    for gpu_idx in range(num_gpus):
        avg = gpu_redundant_totals[gpu_idx] / num_layers
        print(f"  GPU {gpu_idx}: {gpu_redundant_totals[gpu_idx]:4d} total, {avg:6.2f} average per layer")
    
    total_expected = num_redundant_experts * num_layers
    total_actual = sum(gpu_redundant_totals)
    print(f"\nExpected total: {total_expected}, Actual total: {total_actual}")
    
    if total_actual == total_expected:
        print("✓ All redundant experts successfully placed")
    else:
        print(f"⚠ Missing {total_expected - total_actual} redundant experts")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create expert placement with evenly distributed redundancy."
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        required=True,
        help="The start number of MoE layers for the model.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=True,
        help="The total number of layers for the model.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="The total number of GPUs",
    )
    parser.add_argument(
        "--num_routed_experts",
        type=int,
        required=True,
        help="The total number of routed experts",
    )
    parser.add_argument(
        "--hot_experts_file",
        type=str,
        default=None,
        help="Path to JSON file containing hot experts data",
    )
    parser.add_argument(
        "--num_redundant_experts",
        type=int,
        default=4,
        help="Number of hot experts to duplicate per layer (default: 4)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="The base filename for the output .json and .pt files.",
    )
    args = parser.parse_args()
    
    # Load hot experts data if provided
    hot_experts_data = None
    if args.hot_experts_file:
        with open(args.hot_experts_file, 'r') as f:
            hot_experts_data = json.load(f)
        print(f"Loaded hot experts data from {args.hot_experts_file}")
    
    # Create placement with redundancy
    physical_to_logical_map = create_expert_placement_with_redundancy(
        start_layer=args.start_layer,
        num_layers=args.num_layers,
        num_gpus=args.num_gpus,
        num_logical_experts=args.num_routed_experts,
        hot_experts_data=hot_experts_data,
        num_redundant_experts=args.num_redundant_experts
    )
    
    # Visualize the pattern
    visualize_pattern_with_redundancy(
        physical_to_logical_map, 
        num_gpus=args.num_gpus,
        num_redundant_experts=args.num_redundant_experts
    )
    
    # Save to files
    json_path = Path(f"{args.output_filename}.json")
    pt_path = Path(f"{args.output_filename}.pt")

    num_physical_experts = args.num_routed_experts + args.num_redundant_experts
    return_physical_to_logical_map = torch.full(
        (args.num_layers, num_physical_experts), -1, dtype=torch.long
    )

    for layer_idx in range(args.start_layer, args.num_layers):
        return_physical_to_logical_map[layer_idx] = torch.LongTensor([i for i in physical_to_logical_map[layer_idx] if i != -1])
    
    mapping_config = {"physical_to_logical_map": return_physical_to_logical_map.tolist()}
    
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(mapping_config, f)
    
    torch.save(
        {"physical_to_logical_map": return_physical_to_logical_map},
        pt_path,
    )
    
    print(f"\nExpert mapping with evenly distributed redundancy created successfully!")
    print(f"  - JSON config saved to: {json_path}")
    print(f"  - PyTorch tensor saved to: {pt_path}")
    print(f"  - Redundant experts per layer: {args.num_redundant_experts}")
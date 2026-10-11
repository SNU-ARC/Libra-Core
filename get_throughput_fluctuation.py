#!/usr/bin/env python3
"""Combine per-dataset prefill throughput into the dynamic-workload series of paper Fig. 9 (top row).

Reads the JSON-lines files written by the launcher scripts into latency_throughput/ (next to this
script):
  {model}_sglang_{dataset}_{seq_len}.txt      single_node_scripts/ep_test.sh ... sglang
  {model}_libra_{dataset}_{seq_len}.txt       single_node_scripts/ep_test.sh ... libra
  {model}_lina_{dataset}_{N}_{seq_len}.txt    iclr2026_lina branch (N = 6 for 2048, 8 for 4096)
For every model, sequence length and scheme, the per-step prefill throughputs of the 8 datasets
are concatenated in alphabetical dataset order, shuffled with a fixed permutation, and printed as
a list.
"""
import json
import os
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "latency_throughput"


def build_dataset_matrix(root, model, dataset, seq_len, scheme):
    """Return the list of `prefill_throughput` values of one result file."""
    base_dir = root
    if scheme[:4] == 'eplb':
        assert(len(scheme) > 4)
        path = Path(base_dir) /  Path(f'{model}_{scheme[:4]}_{scheme[4:]}_{dataset}_{seq_len}.txt')
    elif scheme == 'lina':
        if seq_len == 2048:
            N = 6
        elif seq_len == 4096:
            N = 8
        else:
            raise ValueError
        path = Path(base_dir) /  Path(f'{model}_{scheme}_{dataset}_{N}_{seq_len}.txt')
    else:
        path = Path(base_dir) /  Path(f'{model}_{scheme}_{dataset}_{seq_len}.txt')
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Result directory not found: {base_dir}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Result file not found: {path} (run the {scheme} throughput script for {model} first)"
        )

    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                vals.append(json.loads(line)["prefill_throughput"])

    return vals


def main():
    random_array_2048 = [13, 14, 134, 169, 186, 29, 54, 196, 75, 30, 84, 173, 45, 71, 176, 77, 95, 15, 70, 142, 152, 171, 105, 140, 181, 190, 124, 22, 170, 172, 195, 80, 154, 82, 120, 3, 88, 137, 83, 159, 63, 157, 2, 115, 133, 73, 61, 178, 91, 198, 131, 101, 87, 136, 100, 161, 38, 188, 189, 34, 19, 153, 127, 148, 112, 37, 28, 65, 21, 162, 102, 143, 119, 103, 184, 163, 145, 146, 6, 47, 104, 68, 193, 86, 158, 64, 166, 132, 39, 125, 164, 1, 56, 16, 138, 31, 81, 32, 128, 174, 114, 49, 180, 8, 149, 93, 144, 96, 72, 107, 187, 50, 12, 46, 130, 147, 141, 94, 67, 5, 135, 79, 53, 185, 117, 55, 109, 139, 177, 51, 90, 192, 92, 89, 108, 27, 122, 48, 58, 0, 17, 167, 25, 168, 9, 36, 69, 197, 76, 85, 123, 44, 110, 42, 62, 155, 99, 4, 106, 18, 191, 66, 20, 116, 121, 129, 165, 33, 194, 24, 156, 59, 151, 175, 98, 160, 97, 182, 179, 35, 78, 118, 111, 150, 41, 57, 40, 7, 60, 74, 199, 52, 11, 26, 43, 183, 10, 113, 126, 23]
    random_array_4096 = [i for i in random_array_2048 if i < 96]
    for model in ['Qwen3-235B-A22B', 'GLM-4.5']:
        ds_list = ['bookcorpus', 'codeforces', 'deepseek_prover', 'fineweb', 'gsm8k', 'hellaswag', 'humanevalplus', 'lmsys_chat_1m']

        for seq_len in [2048, 4096]:
            for scheme in ['sglang', 'libra', 'lina']:
                result = []
                for ds in ds_list:
                    values = build_dataset_matrix(RESULT_DIR, model, ds, seq_len, scheme)
                    result.extend(values)

                if len(result) == 0:
                    print(f"# {model} | seq_len={seq_len} | scheme={scheme}")
                    print("[]")  # nothing to show
                    continue

                if seq_len == 2048:
                    final_result = [result[i] for i in random_array_2048]
                else:
                    final_result = [result[i] for i in random_array_4096]

                print(f"# {model} | seq_len={seq_len} | scheme={scheme}")
                print("[" + ", ".join(f"{v:.9f}" for v in final_result) + "]")

if __name__ == "__main__":
    main()

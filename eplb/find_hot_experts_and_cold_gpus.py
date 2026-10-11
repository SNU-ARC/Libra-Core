#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

def topk_from_slice(feather_path: Path, start_idx: int, end_idx: int,
                    num_experts: int, num_gpus: int):
    """
    Return {"order": [...], "cold_gpus": [...]} (expert ids sorted by activation count and
    the cold-GPU ranking) for one layer's feather file, or None if the token slice is empty.
    """
    # Load and slice
    df = pd.read_feather(feather_path)
    # Drop helper columns if present
    for col in ("req_idx", "token_idx"):
        if col in df.columns:
            df = df.drop(columns=[col])

    arr = df.values  # shape: (num_tokens, experts_per_token)
    print(df.shape)
    arr = arr[start_idx:end_idx, :]

    if arr.size == 0:
        # Nothing in range → return empty tops
        return None

    # Flatten to 1D and count occurrences of expert IDs
    flat = arr.reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(flat, minlength=num_experts)  # shape: (num_experts,)
    gpu_load = counts.reshape(-1, num_gpus).sum(0)
    cold_gpus = np.argsort(gpu_load)

    total = flat.size
    if total == 0:
        return None

    # Descending sort by count; 'stable' keeps lower-ID tie order predictable
    order = np.argsort(-counts, kind="stable")

    entry = {}

    entry["order"] = order.tolist()
    entry["cold_gpus"] = cold_gpus.tolist()

    return entry

def dump_json_compact_arrays(obj, fp, indent=2, ensure_ascii=False):
    def is_scalar(x): return isinstance(x, (str, int, float, bool)) or x is None
    def encode(o, level=0):
        pad  = ' ' * (indent * level)
        spad = ' ' * (indent * (level + 1))
        if isinstance(o, dict):
            body = ",\n".join(f"{spad}{json.dumps(k, ensure_ascii=ensure_ascii)}: {encode(v, level+1)}"
                              for k, v in o.items())
            return "{\n" + body + "\n" + pad + "}" if body else "{}"
        if isinstance(o, list):
            if all(is_scalar(x) for x in o):
                return "[" + ", ".join(json.dumps(x, ensure_ascii=ensure_ascii) for x in o) + "]"
            body = ",\n".join(f"{spad}{encode(x, level+1)}" for x in o)
            return "[\n" + body + "\n" + pad + "]"
        return json.dumps(o, ensure_ascii=ensure_ascii)
    fp.write(encode(obj))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temt-dir", required=True, help="Directory that has real/layer_*.feather")
    ap.add_argument("--save-dir", required=True, help="Output dir")
    ap.add_argument("--model-alias", required=True, help="e.g., glm45")
    ap.add_argument("--dataset", required=True, help="e.g., bookcorpus")
    ap.add_argument("--layer-idxes", type=int, nargs="+", required=True)
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs')
    ap.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    temt_dir = Path(args.temt_dir)
    real_dir = temt_dir / "real"

    layers_payload = []
    for li in args.layer_idxes:
        feather_path = real_dir / f"layer_{li}.feather"
        if not feather_path.exists():
            print(f"# missing: {feather_path}")
            continue
        tops = topk_from_slice(
            feather_path,
            args.start_idx,
            args.end_idx,
            args.num_experts,
            args.num_gpus,
        )
        if tops is None:
            print(f"# empty slice for layer {li}")
            continue
        layers_payload.append({"layer": li, **tops})

    for dataset in ["bookcorpus", "codeforces", "deepseek_prover", "fineweb", "gsm8k", "hellaswag", "humanevalplus", "lmsys_chat_1m", "mixed"]:
        
        out_obj = {
            f"{args.model_alias}_{dataset}": {
                "layers": layers_payload
            }
        }

        out_dir = Path(args.save_dir)
        out_path = out_dir / f'{dataset}_order_rates.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as jf:
            dump_json_compact_arrays(out_obj, jf, indent=2, ensure_ascii=False)

        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
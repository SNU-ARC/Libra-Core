# Adapted from SGLang v0.4.10 python/sglang/bench_one_batch.py (https://github.com/sgl-project/sglang) for Libra (ICLR 2026).
"""
Prefill throughput benchmark for the Lina baseline of the Libra experiments (paper Fig. 8 and Fig. 9).

Like sglang.bench_one_batch it runs static batches without a server, but the input tokens come
from one of the evaluation datasets instead of synthetic ids. Every DP rank prefills its own
batch of `batch_size * input_len` tokens. Batches [start_idx, end_idx) of the dataset are
measured `tp_size` at a time, after 4 warm-up runs on the first `tp_size` batches, and rank 0
appends one JSON line per measured step (prefill latency / throughput of rank 0) to
--result-filename.

Environment variables (exported by single_node_scripts/ep_test_qwen3.sh and ep_test_glm45.sh):
  LIBRA_SCHEME            load-balancing scheme, "lina" on this branch (required by the Qwen3-MoE /
                          GLM-4.5 decoder layers)
  EXPERT_DUP_FACTOR_N     Lina's N (replicated experts per GPU)
  SEQ_LENS_SUM            number of tokens per rank (batch_size * input_len)
  LINA_EPT_DIR            Lina expert-selection-path tables (required)
  LINA_CACHE_DIR          cache of the preprocessed path tables (optional)
  LIBRA_DATA_DIR          local copies of the datasets (default: ~/hf_datasets)

Benchmark arguments added on top of ServerArgs:
  --dataset         bookcorpus | codeforces | deepseek_prover | fineweb | gsm8k | hellaswag |
                    humanevalplus | lmsys_chat_1m
  --mode            HF split to read (default: train; HumanEvalPlus always reads "test")
  --start-portion   start of the split slice in percent, i.e. "<mode>[start%:end%]"
  --end-portion     end of the split slice in percent (GSM8K ignores both and reads the whole split)
  --start-idx       first batch to measure (one batch = the input of one DP rank)
  --end-idx         end of the measured batch range (exclusive)
  --is-real         accepted for compatibility with the launcher scripts; has no effect

Example (normally run through single_node_scripts/ep_test_qwen3.sh, which sets more flags):
LIBRA_SCHEME=lina EXPERT_DUP_FACTOR_N=6 SEQ_LENS_SUM=2048 \
LINA_EPT_DIR=$HOME/libra_artifacts/lina_epts/qwen3_mixed_ept \
python -m sglang.bench_one_batch_test --model-path ~/Qwen3-235B-A22B --trust-remote-code \
    --tp-size 8 --dp-size 8 --enable-dp-attention --enable-ep-moe --ep-size 8 \
    --batch-size 2 --input-len 1024 --output-len 0 --dataset bookcorpus --mode train \
    --start-portion 0 --end-portion 1 --start-idx 800 --end-idx 1000 \
    --result-filename latency_throughput/Qwen3-235B-A22B_lina_bookcorpus_6_2048.txt
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)


from sglang.srt.utils import DeepEPMode
from typing import Iterable, List
from pathlib import Path
from datasets import load_dataset, load_from_disk


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_filename_prefix: str = "profile"
    ######################################################
    
    dataset: str = "gsm8k"
    mode: str = "test"
    is_real: bool = False
    start_portion: int = 25
    end_portion: int = 25
    start_idx: int = 0
    end_idx: int = 0
    ######################################################

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument(
            "--profile", action="store_true", help="Use Torch Profiler."
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        ######################################################
        
        parser.add_argument(
            "--dataset",
            type=str,
            choices=[
                "hellaswag", 
                "gsm8k", 
                "bookcorpus", 
                "lmsys_chat_1m", 
                "fineweb", 
                "codeforces", 
                "humanevalplus", 
                "deepseek_prover",
            ],
            default="gsm8k",
            help="Evaluation dataset that provides the input tokens.",
        )
        parser.add_argument("--mode", type=str, default="train")
        parser.add_argument(
            "--is-real",
            action="store_true",
            help="Accepted for compatibility with the launcher scripts; has no effect.",
        )
        parser.add_argument(
            "--start-portion",
            type=int,
            default=BenchArgs.start_portion,
            help="Start of the split slice, in percent.",
        )
        parser.add_argument(
            "--end-portion",
            type=int,
            default=BenchArgs.end_portion,
            help="End of the split slice, in percent.",
        )
        parser.add_argument(
            "--start-idx",
            type=int,
            default=BenchArgs.start_idx,
            help="First batch to measure.",
        )
        parser.add_argument(
            "--end-idx",
            type=int,
            default=BenchArgs.end_idx,
            help="End of the measured batch range (exclusive).",
        )
        ######################################################

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    model_runner.max_total_num_tokens = 90000
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer):
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs

#################################################
# Dataset loaders (the 8 evaluation datasets)
def _get_hf_root(bench_args) -> Path:
    """Return the root directory of the local dataset copies: $LIBRA_DATA_DIR (default ~/hf_datasets)."""
    return Path(os.environ.get("LIBRA_DATA_DIR", "~/hf_datasets")).expanduser()


def _split_expr(bench_args) -> str:
    """Compose split slice like "train[0%:10%]" from bench_args.

    Requires bench_args.mode, bench_args.start_portion, bench_args.end_portion.
    """
    return f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]"


def _batchify_stream(
    tokenizer,
    texts: Iterable[str],
    batch_size: int,
    input_len: int,
    tp_size: int,
    *,
    carry_over: bool = True,
) -> List[List[int]]:
    """Tokenize `texts` and pack the token ids into consecutive batches.

    Each batch is a flat list of exactly `batch_size * input_len` token ids. With `carry_over`
    (the default) the leftover tokens of one text continue into the next batch; otherwise each
    text contributes at most one batch and its remainder is dropped. `tp_size` is unused.
    """
    cap = batch_size * input_len
    total_batches: List[List[int]] = []
    buf: List[int] = []

    def _flush_if_ready() -> None:
        nonlocal buf
        while len(buf) >= cap:
            total_batches.append(buf[:cap])
            buf = buf[cap:]

    for text in texts:
        encoded = tokenizer.encode(text)
        if carry_over:
            buf.extend(encoded)
            _flush_if_ready()
        else:
            buf.extend(encoded)
            if len(buf) >= cap:
                total_batches.append(buf[:cap])
                buf.clear()
            else:
                # drop remainder
                buf.clear()

    return total_batches


def _load_local_dataset(cache_root: Path, name: str, bench_args):
    """Load the HF Hub dataset `name` (cached under `cache_root / name`), sliced as in _split_expr."""
    last_err = None
    cache_path = cache_root / name
    try:
        ds = load_dataset(
            name,
            split=_split_expr(bench_args),
            trust_remote_code=True,
            cache_dir=str(cache_path),
        )
        return ds
    except Exception as e:
        last_err = e
        raise last_err

def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret

def load_gsm8k_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    """Pack 2-shot GSM8K examples ("Question: <question>\nAnswer: <answer>") into batches.

    Reads a `save_to_disk` copy from $LIBRA_DATA_DIR/gsm8k if present, otherwise the HF Hub
    dataset "gsm8k" (config "main") cached there. The whole --mode split is used.
    """
    cache_root = _get_hf_root(bench_args)

    cache_path = cache_root / "gsm8k"
    # Use a dataset saved with `save_to_disk` if it exists (same check as for bookcorpus)
    is_saved_dataset = (cache_path / "dataset_info.json").exists() or (
        cache_path / "state.json"
    ).exists() or (cache_path / "dataset_dict.json").exists()

    if is_saved_dataset:
        # Load from local disk if saved dataset exists
        dataset = load_from_disk(str(cache_path))
        # If the dataset is a DatasetDict, get the appropriate split
        if hasattr(dataset, bench_args.mode):
            dataset = dataset[bench_args.mode]
    else:
        dataset = load_dataset(
            "gsm8k",
            "main",
            split=bench_args.mode,
            cache_dir=cache_path,
            trust_remote_code=True,
        )

    dataset_lines = [
        {"question": item["question"], "answer": item["answer"]} for item in dataset
    ]

    # Create prompts
    num_shots = 2
    total_batches = []
    input_ids = []
    max_num_questions = len(dataset_lines)

    for i in range(0, max_num_questions, num_shots):
        if len(total_batches) >= end_idx:
            break
        if i + num_shots >= max_num_questions:
            excess_num = i + num_shots - max_num_questions
            few_shot_examples = get_few_shot_examples(
                dataset_lines[i:], num_shots - excess_num
            )
        else:
            few_shot_examples = get_few_shot_examples(dataset_lines[i:], num_shots)
        for text in few_shot_examples:
            if len(total_batches) >= end_idx:
                break
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * batch_size:
                total_batches.append(input_ids[: input_len * batch_size])
                input_ids.clear()

    return total_batches[start_idx:end_idx]

def load_bookcorpus_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    """Pack the `text` field of BookCorpus into batches.

    Reads a `save_to_disk` copy from $LIBRA_DATA_DIR/bookcorpus if present, otherwise the HF Hub
    dataset "bookcorpus" cached there.
    """
    cache_root = _get_hf_root(bench_args)
    cache_path = cache_root / "bookcorpus"
    
    is_saved_dataset = (cache_path / "dataset_info.json").exists() or (
        cache_path / "state.json"
    ).exists() or (cache_path / "dataset_dict.json").exists()

    if is_saved_dataset:
        dataset = load_from_disk(str(cache_path))
    else:
        dataset = _load_local_dataset(
            cache_root=cache_root,
            name="bookcorpus",
            bench_args=bench_args,
        )

    def _texts():
        for sample in dataset:
            txt = sample.get("text") or ""
            if txt:
                yield txt

    return _batchify_stream(tokenizer, _texts(), batch_size, input_len, tp_size, carry_over=True)[start_idx:end_idx]

def load_lmsyschat_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    hf_dataset_dir = _get_hf_root(bench_args)
    cache_path = hf_dataset_dir / "lmsys-chat-1m"

    is_saved_dataset = (cache_path / "dataset_info.json").exists() or (
        cache_path / "state.json"
    ).exists() or (cache_path / "dataset_dict.json").exists()

    if is_saved_dataset:
        dataset = load_from_disk(str(cache_path))
    else:
        dataset = load_dataset(
            str(cache_path),
            split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]",
            trust_remote_code=True,
        )

    total_batches = []
    input_ids = []
    for _, sample in enumerate(dataset):
        if len(total_batches) >= end_idx:
            break
        for turn in sample["conversation"]:
            text = f"{turn['role']}: {turn['content']}"
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * batch_size:
                total_batches.append(input_ids[: input_len * batch_size])
                input_ids.clear()
    return total_batches[start_idx:end_idx]

def load_fineweb_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    cache_root = _get_hf_root(bench_args)
    cache_path = cache_root / "fineweb"

    dataset = load_dataset(
        "parquet", 
        data_files={"train":f"{cache_path}/sample/10BT/000_00000.parquet"}, 
        split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", 
        trust_remote_code=True
    )

    total_batches = []
    input_ids = []
    for _, sample in enumerate(dataset):
        if len(total_batches) >= end_idx:
            break
        for text in sample["text"]:
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * batch_size:
                total_batches.append(input_ids[: input_len * batch_size])
                input_ids.clear()
    
    return total_batches[start_idx:end_idx]

def load_hellaswag_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    cache_root = _get_hf_root(bench_args)
    cache_path = cache_root / "hellaswag"

    dataset = load_dataset(
        "parquet", 
        data_files={"train":f"{cache_path}/data/train-00000-of-00001.parquet"}, 
        split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", 
        trust_remote_code=True
    )

    total_batches = []
    input_ids = []
    for _, sample in enumerate(dataset):
        if len(total_batches) >= end_idx:
            break
        for text in sample["ctx"]:
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * batch_size:
                total_batches.append(input_ids[: input_len * batch_size])
                input_ids.clear()
    
    return total_batches[start_idx:end_idx]

def load_codeforces_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    cache_root = _get_hf_root(bench_args)
    cache_path = cache_root / "codeforces"

    dataset = load_dataset(str(cache_path), split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)

    texts = [sample['description'] for sample in dataset if sample['description'] is not None]

    total_batches = []
    input_ids = []
    for _, sample in enumerate(dataset):
        if len(total_batches) >= end_idx:
            break
        if sample.get("description") is None:
            continue
        for text in sample["description"]:
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * batch_size:
                total_batches.append(input_ids[: input_len * batch_size])
                input_ids.clear()
    
    return total_batches[start_idx:end_idx]

def load_humanevalplus_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    cache_root = _get_hf_root(bench_args)
    cache_path = cache_root / "humanevalplus"

    dataset = load_dataset(str(cache_path), split=f"test[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)

    total_batches = []
    input_ids = []
    for _, sample in enumerate(dataset):
        if len(total_batches) >= end_idx:
            break
        for text in f"prompt: {sample['prompt']}, test: {sample['test']}":
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * batch_size:
                total_batches.append(input_ids[: input_len * batch_size])
                input_ids.clear()
    
    return total_batches[start_idx:end_idx]

def load_deepseek_prover_dataset(bench_args, tokenizer, batch_size, input_len, tp_size, start_idx, end_idx):
    cache_root = _get_hf_root(bench_args)
    cache_path = cache_root / "DeepSeek-Prover-V1"

    dataset = load_dataset(str(cache_path), 'default', split=f"{bench_args.mode}[{bench_args.start_portion}%:{bench_args.end_portion}%]", trust_remote_code=True)

    total_batches = []
    input_ids = []
    for _, sample in enumerate(dataset):
        if len(total_batches) >= end_idx:
            break
        for text in sample["formal_statement"]:
            encoded = tokenizer.encode(text)
            input_ids.extend(encoded)
            if len(input_ids) > input_len * batch_size:
                total_batches.append(input_ids[: input_len * batch_size])
                input_ids.clear()
    
    return total_batches[start_idx:end_idx]
#################################################


def prepare_eval_inputs_for_latency_test(total_batches, batch_size, input_len, tp_rank):
    # Each DP rank takes its own batch
    input_ids = np.array(total_batches[tp_rank]).reshape(batch_size, input_len)

    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs

@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        Scheduler.prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
            
            enable_two_batch_overlap=model_runner.server_args.enable_two_batch_overlap,
            enable_deepep_moe=model_runner.server_args.enable_deepep_moe,
            deepep_mode=DeepEPMode[model_runner.server_args.deepep_mode],
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
        )

def correctness_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    device,
    log_decode_step,
    profile,
    profile_filename_prefix,
    tp_rank,
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        profiler.start()

    # Prefill
    synchronize(device)
    tic = time.perf_counter()
    next_token_ids, _, batch = extend(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.perf_counter() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    
    if profile:
        profiler.stop()

    # Decode
    decode_latencies = []
    for i in range(output_len - 1):
        synchronize(device)
        tic = time.perf_counter()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.perf_counter() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )
    
    if profile:
        base_dir = Path(os.environ.get("SGLANG_TORCH_PROFILER_DIR"))
        base_dir = base_dir / f"batch{batch_size}_input{input_len}_output{output_len}"
        profile_filename = (
            base_dir
            / f"TP{tp_rank}_batch{batch_size}_input{input_len}_output{output_len}.trace.json.gz"
        )
        parent_dir = os.path.dirname(os.path.abspath(profile_filename))
        os.makedirs(parent_dir, exist_ok=True)
        profiler.export_chrome_trace(str(profile_filename))
        rank_print(f"torch profiler chrome trace saved to {profile_filename}")

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)


    ######################################################
    
    rank_print(f"Loading {bench_args.dataset}...")
    if bench_args.dataset == "gsm8k":
        total_batches = load_gsm8k_dataset(
            bench_args,
            tokenizer,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx,
        )
    elif bench_args.dataset == "bookcorpus":
        total_batches = load_bookcorpus_dataset(
            bench_args, 
            tokenizer, 
            bench_args.batch_size[0], 
            bench_args.input_len[0], 
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx
        )
    elif bench_args.dataset == "lmsys_chat_1m":
        total_batches = load_lmsyschat_dataset(
            bench_args,
            tokenizer,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx,
        )
    elif bench_args.dataset == "fineweb":
        total_batches = load_fineweb_dataset(
            bench_args,
            tokenizer,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx,
        )
    elif bench_args.dataset == "hellaswag":
        total_batches = load_hellaswag_dataset(
            bench_args,
            tokenizer,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx,
        )
    elif bench_args.dataset == "codeforces":
        total_batches = load_codeforces_dataset(
            bench_args,
            tokenizer,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx,
        )
    elif bench_args.dataset == "humanevalplus":
        total_batches = load_humanevalplus_dataset(
            bench_args,
            tokenizer,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx,
        )
    elif bench_args.dataset == "deepseek_prover":
        total_batches = load_deepseek_prover_dataset(
            bench_args,
            tokenizer,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            server_args.tp_size,
            bench_args.start_idx,
            bench_args.end_idx,
        )
    rank_print(f"Number of prompts: {len(total_batches)}")
    ######################################################

    # Warm up: 4 runs on the first tp_size batches
    for i in range(4):
        rank_print(f"Warmup ...{i}")
        for bs, il, ol in itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        ):
            reqs = prepare_eval_inputs_for_latency_test(total_batches[:server_args.tp_size], bs, il, tp_rank)
            ret = latency_test_run_once(
                bench_args.run_name,
                model_runner,
                rank_print,
                reqs,
                bs,
                il,
                ol,
                server_args.device,
                bench_args.log_decode_step,
                False,
                bench_args.profile_filename_prefix,
                tp_rank=tp_rank,
            )
    rank_print("Benchmark ...")

    
    torch.cuda.cudart().cudaProfilerStart()

    # Run the sweep: batches [start_idx, end_idx), tp_size batches (one per DP rank) per step
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        total_batch_size = bench_args.end_idx - bench_args.start_idx
        for i in range(0, total_batch_size, server_args.tp_size):
            if i + server_args.tp_size > total_batch_size:
                break
            reqs = prepare_eval_inputs_for_latency_test(total_batches[i:i + server_args.tp_size], bs, il, tp_rank)
            ret = latency_test_run_once(
                bench_args.run_name,
                model_runner,
                rank_print,
                reqs,
                bs,
                il,
                ol,
                server_args.device,
                bench_args.log_decode_step,
                bench_args.profile,
                bench_args.profile_filename_prefix,
                tp_rank=tp_rank,
            )
            if ret is not None:
                result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()
    
    torch.cuda.cudart().cudaProfilerStop()

def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    os.environ["TRITON_DISABLE_COMPILATION"] = "1"
    os.environ["TRITON_ABORT_ON_CACHE_MISS"] = "1"
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_SHM_DISABLE'] = '0'
    os.environ['NCCL_BUFFSIZE'] = '1048576'
    torch.set_printoptions(precision=10, sci_mode=False)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)

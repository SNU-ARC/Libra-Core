import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import xxhash


# Only supports L=3
CONFIGS = [
    triton.Config({'BLOCK_SIZE': 256 },  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512 },  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024},  num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024},  num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE': 2048},  num_warps=8, num_stages=3),
]
@triton.autotune(configs=CONFIGS, key=['seq_len', 'num_gpus'])
@triton.jit
def compute_num_devices_kernel_v2(
    # inputs
    paths_ptr,
    ept_keys_ptr,
    ept_expert_ids_ptr,
    ept_values_ptr, 
    # outputs
    num_devices_float_ptr,
    # scalars
    seq_len: tl.constexpr,
    num_gpus: tl.constexpr,
    max_dup_per_gpu,
    ept_values_len,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_id = tl.program_id(0)
    i = tl.program_id(1) * BLOCK_SIZE

    total_seq_len = seq_len * num_gpus
    s = 0.0

    base_offsets = tl.arange(0, BLOCK_SIZE)

    # For all tokens, binary search expert path table and sum probabilities for this expert
    offsets = i + base_offsets
    mask = offsets < total_seq_len

    # Load expert path for each token
    e0 = tl.load(paths_ptr + offsets, mask=mask, other=0)
    e1 = tl.load(paths_ptr + total_seq_len + offsets, mask=mask, other=0)
    e2 = tl.load(paths_ptr + total_seq_len * 2 + offsets, mask=mask, other=0)
    e3 = tl.load(paths_ptr + total_seq_len * 3 + offsets, mask=mask, other=0)

    # Pack path for lookup
    packed_path = (e0 << 48) | (e1 << 32) | (e2 << 16) | e3
    low = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    high = tl.full((BLOCK_SIZE,), ept_values_len-1, dtype=tl.int64)

    # Binary search to find matching path in expert path table
    for _ in range(30):
        mid = (low + high) // 2
        idx = tl.load(ept_keys_ptr + mid, mask=mask)
        cond = packed_path > idx
        low = tl.where(cond, mid + 1, low)
        high = tl.where(cond, high, mid)

    # For each top-k expert, accumulate probability if matches expert_id
    for k in range(topk):
        expert_ids = tl.load(ept_expert_ids_ptr + low * topk + k, mask=mask, other=-1)
        expert_mask = expert_ids == expert_id
        value = tl.load(ept_values_ptr + low * topk + k, mask=mask & expert_mask, other=0.0)
        s += tl.sum(value)
    
    # Normalize and store result
    num_devices_float = s / seq_len * max_dup_per_gpu
    tl.atomic_add(num_devices_float_ptr + expert_id, num_devices_float)


@triton.jit
def prefetch_kernel_v2(
    # inputs
    num_devices_ptr,
    # outputs
    next_expert_gpu_mapping_ptr,
    # scalars
    num_gpus: tl.constexpr,
    num_experts: tl.constexpr,
    max_dup_per_gpu,
):
    experts_per_gpu = num_experts // num_gpus
    n_dups = tl.zeros([num_gpus], dtype=tl.int64)
    idx = tl.arange(0, num_gpus)
    # For each expert, assign to GPUs in round-robin fashion up to n times
    for e in range(num_experts):
        n = tl.load(num_devices_ptr + e)
        for i in range(num_gpus):
            g = (e + i) % num_gpus
            # Assign expert to GPU if not original and under duplication limit
            if n > 0 and (g != e // experts_per_gpu and tl.sum(tl.where(idx == g, n_dups, 0)) < max_dup_per_gpu):
                tl.store(next_expert_gpu_mapping_ptr + e * num_gpus + g, True)
                n_dups += tl.cast(idx == g, tl.int64)
                n -= 1
        # Always assign expert to its original GPU
        tl.store(next_expert_gpu_mapping_ptr + e * num_gpus + e // experts_per_gpu, True)


@triton.jit
def prefetch_postprocess_kernel(
    # inputs
    next_expert_gpu_mapping_ptr,
    # outputs
    send_map_ptr,
    send_len_ptr,
    recv_len_ptr,
    recv_experts_ptr,
    recv_num_ptr,
    # scalars
    num_gpus,
    cur_rank,
    experts_per_gpu,
    max_dup_per_gpu,
):
    # expert e is prefetched to GPU g_dst
    e = tl.program_id(0)
    g_dst = tl.program_id(1)

    # expert e is originally on GPU g_src
    g_src = e // experts_per_gpu

    # valid only if g_src != g_dst AND expert e is prefetched on GPU g_dst
    map_val = tl.load(next_expert_gpu_mapping_ptr + e * num_gpus + g_dst)
    cond = (g_src != g_dst) & map_val

    # send_map[g_src][g_dst][send_len[g_src][g_dst]] = e % experts_per_gpu
    # send_len[g_src][g_dst] += 1
    sidx = tl.atomic_add(send_len_ptr + g_src * num_gpus + g_dst, 1, mask=cond)
    tl.store(send_map_ptr + ((g_src * num_gpus + g_dst) * max_dup_per_gpu) + sidx, e % experts_per_gpu, mask=cond)

    # recv_len[g_dst][g_src] += 1
    tl.atomic_add(recv_len_ptr + g_dst * num_gpus + g_src, 1, mask=cond)

    # cond2: current rank is receiving expert e
    cond2 = cond & (g_dst == cur_rank)

    # recv_experts[recv_num] = e
    # recv_num += 1
    ridx = tl.atomic_add(recv_num_ptr, 1, mask=cond2)
    tl.store(recv_experts_ptr + ridx, e, mask=cond2)


CONFIGS = [
    triton.Config({'BLOCK_SIZE': 256 },  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512 },  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024},  num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024},  num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE': 2048},  num_warps=8, num_stages=3),
]
@triton.autotune(configs=CONFIGS, key=['M'])
@triton.jit
def compute_expert_gpu_load_kernel_v2(
    # inputs
    topk_ids_ptr,
    # outputs
    expert_gpu_load_ptr,
    # scalars
    num_gpus,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    gpu_id = tl.program_id(0)
    base_offsets = tl.arange(0, BLOCK_SIZE)
    # For each token, count how many times each expert is used on each GPU
    for i in range(0, M, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < M
        expert_ids = tl.load(topk_ids_ptr + gpu_id * M + offsets, mask=mask)
        tl.atomic_add(expert_gpu_load_ptr + expert_ids * num_gpus + gpu_id, 1, mask=mask)


@triton.jit
def route_kernel(
    # inputs
    expert_gpu_mapping_ptr,
    expert_gpu_load_ptr,
    expert_gpu_count_ptr,
    # outputs
    cur_rank_start_idx_ptr,
    cur_rank_end_idx_ptr,
    # scalars
    cur_rank,
    num_gpus: tl.constexpr,
    num_experts,
):
    e = tl.program_id(0)
    g_src = tl.program_id(1)
    offsets = tl.arange(0, num_gpus)

    # mask of GPUs that have expert e
    g_dst_mask = tl.load(expert_gpu_mapping_ptr + e * num_gpus + offsets)

    # number of tokens that GPU g_src have and go to expert e
    l = tl.load(expert_gpu_load_ptr + e * num_gpus + g_src)

    # number of GPUs that have expert e
    n = tl.load(expert_gpu_count_ptr + e)

    # Compute how many tokens to send to each GPU (evenly)
    cnt = tl.where(g_dst_mask, l // n, 0)
    g_dst_mask_psum = tl.cumsum(tl.cast(g_dst_mask, tl.int64))
    cnt += g_dst_mask & (g_dst_mask_psum <= (l % n))

    # Compute indices of tokens to send to each GPU
    idx = tl.cumsum(cnt) - cnt

    # Consider only cur_rank
    mask = (cnt > 0) & (offsets == cur_rank)
    cur_idx = tl.sum(tl.where(mask, idx, 0))
    cur_cnt = tl.sum(tl.where(mask, cnt, 0))
    valid = tl.max(mask) == 1

    # Write routing info [start_idx, end_idx)
    tl.store(cur_rank_start_idx_ptr + g_src * num_experts + e, cur_idx, mask=valid)
    tl.store(cur_rank_end_idx_ptr + g_src * num_experts + e, cur_idx + cur_cnt, mask=valid)


@triton.jit
def route_postprocess_kernel(
    # inputs
    cur_rank_start_idx_ptr,
    cur_rank_end_idx_ptr,
    topk_ids_ptr,
    # outputs
    topk_ids_mask_ptr,
    # scalars
    num_experts,
    M: tl.constexpr,
):
    e = tl.program_id(0)
    g = tl.program_id(1)

    # Sequential equivalent of the vectorized code below:
    # idx = 0
    # for offset in range(M):
    #     if topk_ids[g, offset] == e:                                      mask1
    #         if cur_rank_start_idx[g, e] <= idx < cur_rank_end_idx[g, e]:  mask2
    #             topk_ids_mask[g, offset] = 1
    #         idx += 1

    offsets = tl.arange(0, M)
    expert_ids = tl.load(topk_ids_ptr + g * M + offsets)

    mask1 = expert_ids == e
    idx = tl.cumsum(tl.cast(mask1, tl.int64)) - 1

    cur_rank_start_idx = tl.load(cur_rank_start_idx_ptr + g * num_experts + e)
    cur_rank_end_idx = tl.load(cur_rank_end_idx_ptr + g * num_experts + e)
    mask2 = (cur_rank_start_idx <= idx) & (idx < cur_rank_end_idx)

    tl.store(topk_ids_mask_ptr + g * M + offsets, 1, mask=mask1 & mask2)


class Lina:
    def __init__(
        self,
        num_layers: int,
        L: int = 3,
        num_gpus: int = 8,
        topk: int = 8,
        max_dup_per_gpu: int = 6,
        device='cuda',
        verbose: bool = False,
    ):
        self.num_layers = num_layers
        self.L = L
        self.num_gpus = num_gpus
        self.topk = topk
        self.max_dup_per_gpu = max_dup_per_gpu
        self.device = device
        self.verbose = verbose

        assert L == 3

        self.expert_path_tables = None
        self.paths = None
        self.num_experts = None

        self.fields = [f'f{i}' for i in range(L+1)]
        self.dtype_key = np.dtype([(f, np.int64) for f in self.fields])
        self.gpu_indices = torch.arange(num_gpus, dtype=torch.int64, device=device)

    def load_expert_path_tables(
        self,
        expert_path_table_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        start_layer: int=0,
        step=1,
    ):
        # Load the expert path tables and precompute, per path, the top-k next-layer experts and
        # their probabilities, sorted by the packed path key (for binary search).
        expert_path_table_dir = Path(expert_path_table_dir)
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        if type(expert_path_table_dir) is str:
            expert_path_table_dir = Path(expert_path_table_dir)
        
        self.expert_path_tables = [None] * self.num_layers

        for i in tqdm(range(self.L + start_layer, self.num_layers-1), disable=not self.verbose):
            ept_data = pd.read_feather(expert_path_table_dir / f'layer_{i*step+step-1}.feather').values

            cache_key = xxhash.xxh128(ept_data.T.data, seed=0).hexdigest()

            keys, expert_ids, values = None, None, None
            if cache_dir is not None:
                keys_path = cache_dir / f'{cache_key}_key'
                expert_ids_path = cache_dir / f'{cache_key}_ids'
                values_path = cache_dir / f'{cache_key}_value'
                if keys_path.is_file():
                    keys = torch.load(keys_path, map_location=self.device)
                if expert_ids_path.is_file():
                    expert_ids = torch.load(expert_ids_path, map_location=self.device)
                if values_path.is_file():
                    values = torch.load(values_path, map_location=self.device)
            
            if keys is not None and values is not None:
                self.expert_path_tables[i] = (keys, expert_ids, values)
                continue

            ept = torch.tensor(ept_data, device=self.device)
            assert ept.dtype == torch.int64
            assert len(ept) == len(set(map(tuple, ept[:, :self.L+1])))

            keys = ept[:, :self.L+1]
            values = ept[:, self.L+1:].float()
            values /= values.sum(axis=1, keepdims=True)
            
            packed_keys = (keys[:, 0] << 48) | (keys[:, 1] << 32) | (keys[:, 2] << 16) | keys[:, 3]
            values, expert_ids = torch.topk(values, self.topk, dim=1)
            
            sorted_indices = torch.argsort(packed_keys)
            keys = packed_keys[sorted_indices]
            expert_ids = expert_ids[sorted_indices]
            values = values[sorted_indices]
            self.expert_path_tables[i] = (keys, expert_ids, values)

            if cache_dir is not None:
                torch.save(keys, keys_path)
                torch.save(expert_ids, expert_ids_path)
                torch.save(values, values_path)
        
        self.num_experts = ept_data.shape[1] - self.L - 1

    def new_batch(self, tot_seq_len: int):
        assert tot_seq_len % self.num_gpus == 0
        self.seq_len = tot_seq_len // self.num_gpus
        self.paths = torch.empty((self.num_layers, tot_seq_len), dtype=torch.int64, device=self.device)

    def record(self, layer_id: int, topk_ids: torch.Tensor):
        assert self.paths is not None
        assert self.paths.shape[1] == topk_ids.shape[0], (self.paths.shape, topk_ids.shape)
        self.paths[layer_id, :] = topk_ids[:, 0]
    
    def prefetch_v2(
        self,
        layer_id: int,
        cur_rank: int,
        send_map_local_out: torch.Tensor,
        send_map_local_len_out: torch.Tensor,
        recv_map_local_len_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.expert_path_tables is not None, 'Call load_expert_path_tables first'
        assert self.num_experts is not None
        assert self.paths is not None
        assert layer_id >= self.L
        experts_per_gpu = self.num_experts // self.num_gpus

        paths = self.paths[layer_id-self.L:layer_id+1, :]
        ept_keys, ept_expert_ids, ept_values = self.expert_path_tables[layer_id]

        assert paths.is_contiguous()
        assert ept_keys.is_contiguous()
        assert ept_values.is_contiguous()

        # 1. Compute num_devices_float (n_e in the Lina paper)
        num_devices_float = torch.zeros((self.num_experts,), dtype=torch.float32, device=self.device)
        grid = lambda META: (self.num_experts, triton.cdiv(self.seq_len * self.num_gpus, META['BLOCK_SIZE']))
        compute_num_devices_kernel_v2[grid](
            paths,
            ept_keys,
            ept_expert_ids,
            ept_values,
            num_devices_float,
            self.seq_len,
            self.num_gpus,
            self.max_dup_per_gpu,
            len(ept_values),
            self.topk,
        )

        # 2. Compute num_devices (integer number of replicas per expert)
        num_devices = torch.floor(num_devices_float).int()
        sorted_num_devices_float = torch.argsort(torch.fmod(num_devices_float, 1), descending=True, stable=True)
        mask = sorted_num_devices_float < self.num_gpus*self.max_dup_per_gpu - num_devices.sum()
        num_devices.add_(mask.int())
        
        # 3. prefetch
        next_expert_gpu_mapping = torch.zeros((self.num_experts, self.num_gpus), dtype=torch.bool, device=self.device)
        prefetch_kernel_v2[(1,)](
            num_devices,
            next_expert_gpu_mapping,
            self.num_gpus,
            self.num_experts,
            self.max_dup_per_gpu,
        )

        # 4. Postprocess: build the send/recv maps on the GPU (no host round-trip)
        recv_experts_next_cur_rank = torch.full((self.max_dup_per_gpu + 1,), 0x3f3f3f3f, dtype=torch.int64, device=self.device)
        recv_num_experts_next = torch.zeros((1,), dtype=torch.int64, device=self.device)
        send_map_local_out.fill_(0x3f3f3f3f)
        send_map_local_len_out.fill_(0)
        recv_map_local_len_out.fill_(0)

        grid = (self.num_experts, self.num_gpus)
        prefetch_postprocess_kernel[grid](
            next_expert_gpu_mapping,
            send_map_local_out,
            send_map_local_len_out,
            recv_map_local_len_out,
            recv_experts_next_cur_rank,
            recv_num_experts_next,
            self.num_gpus,
            cur_rank,
            experts_per_gpu,
            self.max_dup_per_gpu,
        )

        res, _ = send_map_local_out.sort()
        send_map_local_out.copy_(res)

        res, _ = recv_experts_next_cur_rank.sort()
        recv_experts_next_cur_rank.copy_(res)

        return next_expert_gpu_mapping, recv_experts_next_cur_rank, recv_num_experts_next


    def route_v2(
        self,
        topk_ids: torch.Tensor,             # Shape: (num_gpus * seq_len, experts_per_token) - true expert IDs for each token
        expert_gpu_mapping: torch.Tensor,   # Shape: (num_experts, num_gpus) - expert placement matrix
        cur_rank: int,                      # rank of current GPU
        topk_ids_mask_out: torch.Tensor,    # Shape: (num_gpus * seq_len, experts_per_token) - mask of tokens processed on cur_rank
    ) -> None:
        assert self.num_experts is not None
        assert expert_gpu_mapping.is_contiguous()
        topk_ids = topk_ids.reshape(self.num_gpus, -1).contiguous()

        topk_ids_mask_out.fill_(0)

        # 1. Compute expert_gpu_load
        expert_gpu_load = torch.zeros((self.num_experts, self.num_gpus), dtype=torch.int64, device=self.device)
        grid = (self.num_gpus,)
        compute_expert_gpu_load_kernel_v2[grid](
            topk_ids,
            expert_gpu_load,
            self.num_gpus,
            topk_ids.size(1),
        )

        # 2. route
        expert_gpu_count = expert_gpu_mapping.count_nonzero(dim=1)
        cur_rank_start_idx = torch.full((self.num_gpus, self.num_experts), -1, dtype=torch.int64, device=self.device)
        cur_rank_end_idx = torch.full((self.num_gpus, self.num_experts), -1, dtype=torch.int64, device=self.device)

        grid = (self.num_experts, self.num_gpus)
        route_kernel[grid](
            expert_gpu_mapping,
            expert_gpu_load,
            expert_gpu_count,
            cur_rank_start_idx,
            cur_rank_end_idx,
            cur_rank,
            self.num_gpus,
            self.num_experts,
        )

        route_postprocess_kernel[grid](
            cur_rank_start_idx,
            cur_rank_end_idx,
            topk_ids,
            topk_ids_mask_out,
            self.num_experts,
            topk_ids.size(1),
        )

import torch
from sglang.load_balance.prefetch_rebalance_cython import prefetch_impl, rebalance_impl


def prefetch(
    next_topk_ids: torch.Tensor,    # Shape: (num_gpus * seq_len, experts_per_token) - predicted expert IDs for each token
    num_experts: int,
    seq_len: int,
    num_gpus: int,
    cur_rank: int,
    send_map_local_out: torch.Tensor,
    send_map_local_len_out: torch.Tensor,
    recv_map_local_len_out: torch.Tensor,
    max_dup_per_gpu: int = 6,       # (N) maximum number of duplicated experts per GPU
    n_local_dup_per_gpu: int = 4,   # (L) local hot experts to duplicate
    max_send_len: int = 6,          # (P) maximum number of destination GPUs a GPU sends experts to
    force_max_dup: bool = False,    # force each GPU to have max_dup_per_gpu experts
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_topk_ids_numpy = next_topk_ids.reshape(num_gpus, seq_len, -1).numpy()
    next_expert_gpu_mapping, remote_gpu_expert_load, recv_experts_next_cur_rank = prefetch_impl(
        next_topk_ids_numpy,
        num_experts,
        seq_len,
        num_gpus,
        cur_rank,
        send_map_local_out.numpy(),
        send_map_local_len_out.numpy(),
        recv_map_local_len_out.numpy(),
        max_dup_per_gpu,
        n_local_dup_per_gpu,
        max_send_len,
        force_max_dup,
    )
    return torch.tensor(next_expert_gpu_mapping), torch.tensor(remote_gpu_expert_load), torch.tensor(recv_experts_next_cur_rank)


def rebalance(
    topk_ids: torch.Tensor,             # Shape: (num_gpus * seq_len, experts_per_token) - true expert IDs for each token
    expert_gpu_mapping: torch.Tensor,   # Shape: (num_experts, num_gpus) - expert placement matrix
    init_remote_gpu_expert_load: torch.Tensor,
    num_experts: int,
    seq_len: int,
    num_gpus: int,
    cur_rank: int,
    topk_ids_mask_out: torch.Tensor,    # Shape: (num_gpus * seq_len, experts_per_token) - mask of remote tokens processed on cur_rank
    min_dup_experts_tokens: int = 0,    # (Q) minimum limit of tokens that duplicated experts process on a GPU
    verbose: bool = False,
) -> None:
    assert topk_ids_mask_out.is_contiguous()
    topk_ids_numpy = topk_ids.reshape(num_gpus, seq_len, -1).numpy()
    topk_ids_mask_numpy = topk_ids_mask_out.reshape(num_gpus, seq_len, -1).numpy()
    rebalance_impl(
        topk_ids_numpy,
        expert_gpu_mapping.numpy(),
        init_remote_gpu_expert_load.numpy(),
        num_experts,
        seq_len,
        num_gpus,
        cur_rank,
        topk_ids_mask_numpy,
        min_dup_experts_tokens,
    )

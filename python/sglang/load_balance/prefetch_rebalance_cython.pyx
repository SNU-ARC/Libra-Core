# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.string cimport memset

DEF MAX_GPUS = 8
DEF MAX_EXPERTS = 160
DEF MAX_LOCAL_DUP_PER_GPU  = 16

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

def prefetch_impl(
    long [:, :, :] next_topk_ids not None,
    long num_experts,
    long seq_len,
    long num_gpus,
    long cur_rank,
    long [:, :, ::1] send_map_local not None,
    long [:, ::1] send_map_local_len not None,
    long [:, ::1] recv_map_local_len not None,
    long max_dup_per_gpu,
    long n_local_dup_per_gpu,
    long max_send_len,
    bint force_max_dup,
):
    """Greedy expert duplication / prefetch planner.

    Parameters
    ----------
    n_local_dup_per_gpu : int
        How many additional *local* duplicates are allowed per GPU.  The original
        C implementation restricted this to <=4 because it used hard-coded
        4-element scratch buffers.  We generalise the algorithm by introducing
        :py:data:`MAX_LOCAL_DUP_PER_GPU` and replacing the fixed-width logic with
        small insertion sorts so the planner works for any value up to that
        compile-time constant.
    """

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    assert n_local_dup_per_gpu <= MAX_LOCAL_DUP_PER_GPU, (
        f"n_local_dup_per_gpu={n_local_dup_per_gpu} exceeds hard limit "
        f"MAX_LOCAL_DUP_PER_GPU={MAX_LOCAL_DUP_PER_GPU}.  Increase the constant "
        "and re-compile if you really need more." )
    assert n_local_dup_per_gpu <= max_dup_per_gpu
    assert num_gpus <= MAX_GPUS
    assert num_experts <= MAX_EXPERTS

    cdef int experts_per_token = next_topk_ids.shape[2]
    cdef int experts_per_gpu   = num_experts // num_gpus
    cdef int balanced_load     = seq_len * experts_per_token

    # ------------------------------------------------------------------
    # Scratch buffers / accumulators
    # ------------------------------------------------------------------
    cdef np.ndarray[np.npy_bool, ndim=2] next_expert_gpu_mapping_np = \
        np.zeros((num_experts, num_gpus), dtype=np.bool_)
    cdef unsigned char [:, ::1] next_expert_gpu_mapping = next_expert_gpu_mapping_np

    cdef np.ndarray[DTYPE_t, ndim=2] remote_gpu_expert_load_np = \
        np.zeros((num_gpus, num_experts), dtype=DTYPE)
    cdef long [:, ::1] remote_gpu_expert_load = remote_gpu_expert_load_np

    cdef int [MAX_GPUS]                        gpu_load
    cdef int [MAX_EXPERTS]                     expert_load
    cdef int [MAX_GPUS][MAX_EXPERTS]           gpu_expert_load
    cdef int [MAX_EXPERTS][MAX_GPUS][MAX_GPUS] remote_expert_src_dst_load
    cdef bint[MAX_GPUS][MAX_GPUS]              expert_send
    cdef int [MAX_GPUS]                        expert_send_cnt

    memset(gpu_load,                   0, sizeof(gpu_load))
    memset(expert_load,                0, sizeof(expert_load))
    memset(gpu_expert_load,            0, sizeof(gpu_expert_load))
    memset(remote_expert_src_dst_load, 0, sizeof(remote_expert_src_dst_load))
    memset(expert_send,                0, sizeof(expert_send))
    memset(expert_send_cnt,            0, sizeof(expert_send_cnt))

    cdef long* p_send_map_local_len = &send_map_local_len[0, 0]
    memset(p_send_map_local_len, 0, num_gpus * num_gpus * sizeof(long))
    cdef long* p_recv_map_local_len = &recv_map_local_len[0, 0]
    memset(p_recv_map_local_len, 0, num_gpus * num_gpus * sizeof(long))

    cdef int [MAX_LOCAL_DUP_PER_GPU] max_local_load
    cdef int [MAX_LOCAL_DUP_PER_GPU] local_hot_experts

    cdef np.ndarray[DTYPE_t, ndim=1] recv_experts_next_cur_rank_np
    cdef long [MAX_EXPERTS] recv_experts_next_cur_rank_arr
    cdef long [::1] recv_experts_next_cur_rank = recv_experts_next_cur_rank_arr
    cdef int recv_num_experts_next = 0

    cdef int i, j, e, g_src, g_dst, g, l

    # ------------------------------------------------------------------
    # 1) Count #tokens hitting each (gpu, expert)
    # ------------------------------------------------------------------
    for g_src in range(num_gpus):
        for i in range(seq_len):
            for j in range(experts_per_token):
                e = next_topk_ids[g_src, i, j]
                gpu_expert_load[g_src][e] += 1
                expert_load[e] += 1

    # ------------------------------------------------------------------
    # 2) Local duplication pass (phase 1) – select the `n_local_dup_per_gpu` hottest
    #    non-resident experts for each gpu.
    # ------------------------------------------------------------------
    for g_src in range(num_gpus):
        # Mark experts that are already resident on g_src
        next_expert_gpu_mapping[g_src*experts_per_gpu:(g_src+1)*experts_per_gpu, g_src] = 1

        # Move the load of local (resident) experts into gpu_load and reset per‑expert
        for e in range(num_experts):
            if e // experts_per_gpu == g_src:
                gpu_load[g_src] += gpu_expert_load[g_src][e]
                gpu_expert_load[g_src][e] = 0

        # reset scratch arrays
        for i in range(n_local_dup_per_gpu):
            max_local_load[i]     = -1
            local_hot_experts[i]  = -1

        # single‑pass insertion sort to keep top‑k experts by demand
        for e in range(num_experts):
            g = e // experts_per_gpu
            if g == g_src or (expert_send_cnt[g] >= max_send_len and not expert_send[g][g_src]):
                continue
            l = gpu_expert_load[g_src][e]
            if l <= max_local_load[n_local_dup_per_gpu-1]:
                continue  # not hot enough even for the worst in current top‑k

            # Find insertion point (descending order)
            for i in range(n_local_dup_per_gpu):
                if l > max_local_load[i]:
                    # shift right to make room
                    for j in range(n_local_dup_per_gpu-1, i, -1):
                        max_local_load[j]    = max_local_load[j-1]
                        local_hot_experts[j] = local_hot_experts[j-1]
                    max_local_load[i]     = l
                    local_hot_experts[i]  = e
                    break

        # actually duplicate chosen experts
        for i in range(n_local_dup_per_gpu):
            e = local_hot_experts[i]
            if e == -1:
                break
            next_expert_gpu_mapping[e, g_src] = 1
            gpu_load[g_src] += gpu_expert_load[g_src][e]
            gpu_expert_load[g_src][e] = 0
            g = e // experts_per_gpu
            if not expert_send[g][g_src]:
                expert_send[g][g_src] = True
                expert_send_cnt[g] += 1

    for g_src in range(num_gpus):
        for e in range(num_experts):
            g_dst = e // experts_per_gpu
            l = gpu_expert_load[g_src][e]
            gpu_load[g_dst] += l
            remote_gpu_expert_load[g_dst, e] += l
            remote_expert_src_dst_load[e][g_src][g_dst] += l
    
    if max_dup_per_gpu == n_local_dup_per_gpu:
        for e in range(num_experts):
            g_src = e // experts_per_gpu
            for g_dst in range(num_gpus):
                if g_src == g_dst or not next_expert_gpu_mapping[e, g_dst]:
                    continue
                send_map_local[g_src, g_dst, send_map_local_len[g_src, g_dst]] = e % experts_per_gpu
                send_map_local_len[g_src, g_dst] += 1
                recv_map_local_len[g_dst, g_src] += 1
                if g_dst == cur_rank:
                    recv_experts_next_cur_rank[recv_num_experts_next] = e
                    recv_num_experts_next += 1
        recv_experts_next_cur_rank_np = np.asarray(recv_experts_next_cur_rank[:recv_num_experts_next])

        return next_expert_gpu_mapping_np, remote_gpu_expert_load_np, recv_experts_next_cur_rank_np

    cdef bint found
    cdef int [MAX_GPUS] n_dups_gpu
    cdef int [MAX_GPUS] sorted_g_src
    cdef int [MAX_GPUS][MAX_EXPERTS] sorted_experts
    cdef int* cur_sorted_experts
    cdef int shifted_load, num_new_local_tokens
    cdef int cur_val, cur_idx, min_gpu_load, g_src_idx, e_idx

    memset(n_dups_gpu, 0, sizeof(n_dups_gpu))

    for i in range(num_gpus):
        sorted_g_src[i] = i

    for i in range(num_gpus):
        for j in range(num_experts):
            sorted_experts[i][j] = j

    for _ in range((max_dup_per_gpu - n_local_dup_per_gpu) * num_gpus):
        found = False
        
        for i in range(1, num_gpus):
            cur_val = gpu_load[sorted_g_src[i]]
            cur_idx = sorted_g_src[i]
            j = i - 1
            while j >= 0 and gpu_load[sorted_g_src[j]] < cur_val:
                sorted_g_src[j + 1] = sorted_g_src[j]
                j = j - 1
            sorted_g_src[j + 1] = cur_idx

        for g_src_idx in range(num_gpus):
            g_src = sorted_g_src[g_src_idx]
            if gpu_load[g_src] <= balanced_load:
                break
            
            cur_sorted_experts = sorted_experts[g_src]
            
            for i in range(1, num_experts):
                cur_val = remote_gpu_expert_load[g_src, cur_sorted_experts[i]]
                cur_idx = cur_sorted_experts[i]
                j = i - 1
                while j >= 0 and remote_gpu_expert_load[g_src, cur_sorted_experts[j]] < cur_val:
                    cur_sorted_experts[j + 1] = cur_sorted_experts[j]
                    j = j - 1
                cur_sorted_experts[j + 1] = cur_idx
            
            for e_idx in range(num_experts):
                e = cur_sorted_experts[e_idx]
                if remote_gpu_expert_load[g_src, e] <= 0:
                    break
                
                min_gpu_load = 987654321
                g_dst = -1
                for g in range(num_gpus):
                    if expert_send_cnt[g_src] >= max_send_len and not expert_send[g_src][g]:
                        continue
                    if not next_expert_gpu_mapping[e, g] and gpu_load[g] < min_gpu_load and n_dups_gpu[g] + n_local_dup_per_gpu < max_dup_per_gpu:
                        min_gpu_load = gpu_load[g]
                        g_dst = g
                
                if g_dst == -1:
                    continue

                num_new_local_tokens = 0
                for i in range(num_gpus):
                    num_new_local_tokens += remote_expert_src_dst_load[e][g_dst][i]

                if gpu_load[g_dst] + num_new_local_tokens <= balanced_load:
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            break


        next_expert_gpu_mapping[e, g_dst] = True
        n_dups_gpu[g_dst] += 1
        if not expert_send[g_src][g_dst]:
            expert_send[g_src][g_dst] = True
            expert_send_cnt[g_src] += 1

        for g in range(num_gpus):
            l = remote_expert_src_dst_load[e][g_dst][g]
            remote_expert_src_dst_load[e][g_dst][g] = 0
            gpu_load[g] -= l
            gpu_load[g_dst] += l
            remote_gpu_expert_load[g, e] -= l

        shifted_load = min(remote_gpu_expert_load[g_src, e], balanced_load - gpu_load[g_dst])

        gpu_load[g_src] -= shifted_load
        gpu_load[g_dst] += shifted_load
        remote_gpu_expert_load[g_src, e] -= shifted_load
        remote_gpu_expert_load[g_dst, e] += shifted_load
        
        for g in range(num_gpus):
            if shifted_load <= 0: break
            if g == g_src or g == g_dst: continue
            l = min(remote_expert_src_dst_load[e][g][g_src], shifted_load)
            shifted_load -= l
            remote_expert_src_dst_load[e][g][g_src] -= l
            remote_expert_src_dst_load[e][g][g_dst] += l

    cdef int cnt_full

    if force_max_dup:
        cur_sorted_experts = sorted_experts[0]

        for i in range(1, num_experts):
            cur_val = expert_load[cur_sorted_experts[i]]
            cur_idx = cur_sorted_experts[i]
            j = i - 1
            while j >= 0 and expert_load[cur_sorted_experts[j]] < cur_val:
                cur_sorted_experts[j + 1] = cur_sorted_experts[j]
                j = j - 1
            cur_sorted_experts[j + 1] = cur_idx
        

        g = 0
        for e_idx in range(num_experts):
            e = cur_sorted_experts[e_idx]
            g_src = e // experts_per_gpu

            cnt_full = 0
            for i in range(num_gpus):
                g_dst = (g + i) % num_gpus
                if n_dups_gpu[g_dst] + n_local_dup_per_gpu >= max_dup_per_gpu:
                    cnt_full += 1
                elif expert_send_cnt[g_src] >= max_send_len and not expert_send[g_src][g_dst]:
                    continue
                elif not next_expert_gpu_mapping[e, g_dst]:
                    next_expert_gpu_mapping[e, g_dst] = True
                    n_dups_gpu[g_dst] += 1
                    if not expert_send[g_src][g_dst]:
                        expert_send[g_src][g_dst] = True
                        expert_send_cnt[g_src] += 1
                    break
            if cnt_full >= num_gpus:
                break
            g = (g_dst + 1) % num_gpus

    for e in range(num_experts):
        g_src = e // experts_per_gpu
        for g_dst in range(num_gpus):
            if g_src == g_dst or not next_expert_gpu_mapping[e, g_dst]:
                continue
            send_map_local[g_src, g_dst, send_map_local_len[g_src, g_dst]] = e % experts_per_gpu
            send_map_local_len[g_src, g_dst] += 1
            recv_map_local_len[g_dst, g_src] += 1
            if g_dst == cur_rank:
                recv_experts_next_cur_rank[recv_num_experts_next] = e
                recv_num_experts_next += 1
    recv_experts_next_cur_rank_np = np.asarray(recv_experts_next_cur_rank[:recv_num_experts_next])

    return next_expert_gpu_mapping_np, remote_gpu_expert_load_np, recv_experts_next_cur_rank_np


def rebalance_impl(
    long [:, :, :] topk_ids not None,
    unsigned char [:, ::1] expert_gpu_mapping not None,
    long [:, ::1] remote_gpu_expert_load not None,
    long num_experts,
    long seq_len,
    long num_gpus,
    long cur_rank,
    long [:, :, ::1] topk_ids_mask not None,
    long min_dup_experts_tokens,
):
    assert num_gpus <= MAX_GPUS
    assert num_experts <= MAX_EXPERTS

    cdef int experts_per_token = topk_ids.shape[2]
    cdef int experts_per_gpu = num_experts // num_gpus
    cdef int balanced_load = seq_len * experts_per_token

    cdef int [MAX_GPUS][MAX_EXPERTS] expert_load
    cdef int [MAX_EXPERTS] remote_expert_load
    cdef int [MAX_GPUS] gpu_load

    memset(expert_load, 0, sizeof(expert_load))
    memset(remote_expert_load, 0, sizeof(remote_expert_load))
    memset(gpu_load, 0, sizeof(gpu_load))

    cdef int i, j, e, g_src, g_dst, g, cur_sum, target_sum, remainder, n

    for g_src in range(num_gpus):
        for i in range(seq_len):
            for j in range(experts_per_token):
                e = topk_ids[g_src, i, j]
                expert_load[g_src][e] += 1

    for g_src in range(num_gpus):
        for e in range(num_experts):
            if expert_gpu_mapping[e, g_src]:
                gpu_load[g_src] += expert_load[g_src][e]
            else:
                remote_expert_load[e] += expert_load[g_src][e]

    for e in range(num_experts):
        target_sum = remote_expert_load[e]
        cur_sum = 0
        for g in range(num_gpus):
            cur_sum += remote_gpu_expert_load[g, e]
        if target_sum == cur_sum:
            continue
        
        remainder = target_sum
        if cur_sum > 0:
            for g in range(num_gpus):
                if expert_gpu_mapping[e, g]:
                    remote_gpu_expert_load[g, e] = remote_gpu_expert_load[g, e] * target_sum / cur_sum
                    remainder -= remote_gpu_expert_load[g, e]
        else:
            n = 0
            for g in range(num_gpus):
                n += expert_gpu_mapping[e, g]
            for g in range(num_gpus):
                if expert_gpu_mapping[e, g]:
                    remote_gpu_expert_load[g, e] = target_sum / n
                    remainder -= remote_gpu_expert_load[g, e]
        
        for g in range(num_gpus):
            if remainder <= 0:
                break
            if expert_gpu_mapping[e, g]:
                remote_gpu_expert_load[g, e] += 1
                remainder -= 1
    
    for g in range(num_gpus):
        for e in range(num_experts):
            gpu_load[g] += remote_gpu_expert_load[g, e]

    cdef bint found
    cdef int [MAX_GPUS] sorted_g_src
    cdef int [MAX_GPUS][MAX_EXPERTS] sorted_experts
    cdef int* cur_sorted_experts
    cdef int shifted_load, cur_val, cur_idx, min_gpu_load, g_src_idx, e_idx

    for i in range(num_gpus):
        sorted_g_src[i] = i

    for i in range(num_gpus):
        for j in range(num_experts):
            sorted_experts[i][j] = j

    while True:
        found = False

        for i in range(1, num_gpus):
            cur_val = gpu_load[sorted_g_src[i]]
            cur_idx = sorted_g_src[i]
            j = i - 1
            while j >= 0 and gpu_load[sorted_g_src[j]] < cur_val:
                sorted_g_src[j + 1] = sorted_g_src[j]
                j = j - 1
            sorted_g_src[j + 1] = cur_idx
        
        if gpu_load[sorted_g_src[0]] <= balanced_load * 1.01:
            break

        for g_src_idx in range(num_gpus):
            g_src = sorted_g_src[g_src_idx]
            if gpu_load[g_src] <= balanced_load:
                break

            cur_sorted_experts = sorted_experts[g_src]

            for i in range(1, num_experts):
                cur_val = remote_gpu_expert_load[g_src, cur_sorted_experts[i]]
                cur_idx = cur_sorted_experts[i]
                j = i - 1
                while j >= 0 and remote_gpu_expert_load[g_src, cur_sorted_experts[j]] < cur_val:
                    cur_sorted_experts[j + 1] = cur_sorted_experts[j]
                    j = j - 1
                cur_sorted_experts[j + 1] = cur_idx
            
            for e_idx in range(num_experts):
                e = cur_sorted_experts[e_idx]
                if remote_gpu_expert_load[g_src, e] <= 0:
                    break

                min_gpu_load = 987654321
                g_dst = -1
                for g in range(num_gpus):
                    if g != g_src and expert_gpu_mapping[e, g] and gpu_load[g] < min_gpu_load:
                        min_gpu_load = gpu_load[g]
                        g_dst = g
                
                if g_dst == -1:
                    continue

                if gpu_load[g_dst] < balanced_load:
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            break

        shifted_load = min(remote_gpu_expert_load[g_src, e], balanced_load - gpu_load[g_dst])

        gpu_load[g_src] -= shifted_load
        gpu_load[g_dst] += shifted_load
        remote_gpu_expert_load[g_src, e] -= shifted_load
        remote_gpu_expert_load[g_dst, e] += shifted_load

    cdef int [MAX_GPUS] dup_experts_tokens

    if min_dup_experts_tokens > 0:
        while True:
            g_src = -1

            for g in range(num_gpus):
                dup_experts_tokens[g] = 0
                for e in range(num_experts):
                    if e // experts_per_gpu != g and expert_gpu_mapping[e, g]:
                        dup_experts_tokens[g] += remote_gpu_expert_load[g, e]
                if g_src == -1 and 0 < dup_experts_tokens[g] and dup_experts_tokens[g] < min_dup_experts_tokens:
                    g_src = g

            if g_src == -1:
                break

            for e in range(num_experts):
                if e // experts_per_gpu == g_src or not expert_gpu_mapping[e, g_src]:
                    continue

                shifted_load = remote_gpu_expert_load[g_src, e]

                min_gpu_load = 987654321
                g_dst = -1
                for g in range(num_gpus):
                    if g != g_src and expert_gpu_mapping[e, g] and gpu_load[g] < min_gpu_load \
                        and (e // experts_per_gpu == g or dup_experts_tokens[g] + shifted_load >= min_dup_experts_tokens):
                        min_gpu_load = gpu_load[g]
                        g_dst = g
                assert g_dst != -1

                gpu_load[g_src] -= shifted_load
                gpu_load[g_dst] += shifted_load
                remote_gpu_expert_load[g_src, e] = 0
                remote_gpu_expert_load[g_dst, e] += shifted_load
                if e // experts_per_gpu != g_dst:
                    dup_experts_tokens[g_dst] += shifted_load

    cdef int last_idx, recv_cnt, l

    cdef int [MAX_GPUS][MAX_EXPERTS] cur_rank_start_idx
    cdef int [MAX_GPUS][MAX_EXPERTS] cur_rank_end_idx
    memset(cur_rank_start_idx, -1, sizeof(cur_rank_start_idx))
    memset(cur_rank_end_idx, -1, sizeof(cur_rank_end_idx))

    i = 0
    for e in range(num_experts):
        last_idx = 0
        g_src = 0
        for g_dst in range(num_gpus):
            recv_cnt = remote_gpu_expert_load[g_dst, e]
            if not expert_gpu_mapping[e, g_dst]:
                continue

            while recv_cnt > 0:
                if expert_gpu_mapping[e, g_src]:
                    last_idx = 0
                    g_src += 1
                    continue
                l = min(expert_load[g_src][e] - last_idx, recv_cnt)
                if l > 0:
                    if g_dst == cur_rank:
                        cur_rank_start_idx[g_src][e] = last_idx
                        cur_rank_end_idx[g_src][e] = last_idx + l
                    last_idx += l
                    recv_cnt -= l
                if last_idx == expert_load[g_src][e]:
                    last_idx = 0
                    g_src += 1
    
    memset(expert_load, 0, sizeof(expert_load))

    for g_src in range(num_gpus):
        for i in range(seq_len):
            for j in range(experts_per_token):
                e = topk_ids[g_src, i, j]
                l = expert_load[g_src][e]
                if cur_rank_start_idx[g_src][e] <= l and l < cur_rank_end_idx[g_src][e]:
                    topk_ids_mask[g_src, i, j] = 1
                expert_load[g_src][e] += 1
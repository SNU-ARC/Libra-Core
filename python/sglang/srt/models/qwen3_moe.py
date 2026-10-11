# Adapted from qwen2_moe.py

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modified from SGLang v0.4.10 (https://github.com/sgl-project/sglang) for Libra (ICLR 2026).


"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
    get_local_attention_dp_rank,
    get_dp_local_info
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP as Qwen3MoeMLP
from sglang.srt.models.qwen2_moe import Qwen2MoeModel
from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.utils import DeepEPMode, add_prefix, is_cuda, is_non_idle_and_non_empty


from torch.cuda import nvtx
from sglang.srt.utils import (
    find_rows_with_large_diff, 
    EXPERT_DUP_FACTOR_N,
    EXPERT_DUP_FACTOR_L,
    EXPERT_DUP_MAX,
    MIN_TOKEN_LIMIT,
    broadcast_topk,
    broadcast_topk_with_next,
    compute_cum_recv_map_len,
    P2PPlan,
    get_libra_scheme,
)
from sglang.load_balance import prefetch, rebalance
from sglang.srt.layers.moe.ep_moe.kernels import fused_add_without_zero_init_v2
from sglang.srt.layers.moe.topk import StandardTopKOutput

Qwen3MoeConfig = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
        )

        self.experts = get_moe_impl_class()(
            num_experts=config.num_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            **(
                dict(deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
            # Additional args for FusedMoE
            **(
                dict(
                    enable_flashinfer_cutlass_moe=True,
                    enable_ep_moe=global_server_args_dict["enable_ep_moe"],
                )
                if global_server_args_dict["enable_flashinfer_cutlass_moe"]
                else {}
            ),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = (
                config.num_experts + global_server_args_dict["ep_num_redundant_experts"]
            )
            self.top_k = config.num_experts_per_tok
        
        
        self.local_dp_rank = get_local_attention_dp_rank()

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        forward_batch: Optional[ForwardBatch] = None, 
        is_ours: bool = False, 
        topk_ids_mask: torch.Tensor = None,
        w1_buf: torch.Tensor = None,
        w2_buf: torch.Tensor = None,
        topk_output: StandardTopKOutput = None,
    ) -> torch.Tensor:

        if not global_server_args_dict["enable_deepep_moe"]:
            if is_ours:
                return self.forward_normal_ours(
                    hidden_states, 
                    topk_ids_mask=topk_ids_mask,
                    w1_buf=w1_buf,
                    w2_buf=w2_buf,
                    topk_output=topk_output,
                )
            else:
                return self.forward_normal_ori(hidden_states)
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward_normal_ori(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        ###############################################################
        # NOTE: Gate/Top-k
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        expert_location_dispatch_info = ExpertLocationDispatchInfo.init_new(
                layer_id=self.layer_id,
            )
        topk_output = self.topk(
            hidden_states, 
            router_logits,
            expert_location_dispatch_info=expert_location_dispatch_info
        )
        ###############################################################

        ###############################################################
        # NOTE: MoE Computation
        final_hidden_states = self.experts(hidden_states, topk_output)
        ###############################################################

        ###############################################################
        # NOTE: Combine
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        ###############################################################

        return final_hidden_states.view(num_tokens, hidden_dim)
    
    def forward_normal_ours(
            self, 
            hidden_states: torch.Tensor, 
            topk_ids_mask: torch.Tensor = None,
            w1_buf: torch.Tensor = None,
            w2_buf: torch.Tensor = None,
            topk_output: StandardTopKOutput = None,
        ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        ###############################################################
        # NOTE: Gate/Top-k
        if topk_output is None:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            expert_location_dispatch_info = ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                )
            topk_output = self.topk(
                hidden_states, 
                router_logits,
                expert_location_dispatch_info=expert_location_dispatch_info
            )
        ###############################################################

        ###############################################################
        # NOTE: MoE Computation
        final_hidden_states = self.experts(
            hidden_states, 
            topk_output, 
            topk_ids_mask=topk_ids_mask,
            w1_buf=w1_buf,
            w2_buf=w2_buf,
        )
        ###############################################################

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            topk_weights, topk_idx, _ = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )
        return final_hidden_states

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits, _ = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_weights_local, state.topk_idx_local, _ = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_idx_local = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            state.topk_weights_local = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),
                topk_idx=state.pop("topk_idx_local"),
                topk_weights=state.pop("topk_weights_local"),
                forward_batch=state.forward_batch,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.deepep_dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.hidden_states_experts_output = self.experts.moe_impl(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.combine_a(
                hidden_states=state.pop("hidden_states_experts_output"),
                topk_idx=state.dispatch_output.topk_idx,
                topk_weights=state.dispatch_output.topk_weights,
                forward_batch=state.forward_batch,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = (
                self.experts.deepep_dispatcher.combine_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )
            )

    def op_output(self, state):
        state.hidden_states_mlp_output = state.pop("hidden_states_after_combine")


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        event: Optional[torch.cuda.Event] = None,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        
        if event is not None:
            event.record()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(*inner_state)
        output, _ = self.o_proj(attn_output)
        return output

    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        event: Optional[torch.cuda.Event] = None,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            event=event,
        )
        return self.forward_core(s)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        
        self.is_correctness = False
        self.is_ori = get_libra_scheme() in ("sglang", "eplb")
        self.imbalance_ratio = False
        self.local_dp_size = get_local_attention_dp_size()
        self.local_dp_rank = get_local_attention_dp_rank()

        # Qwen3MoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        # CUDA Stream
        rebalance_stream: torch.Stream, 
        prefetch_original_stream: torch.Stream, 
        prefetch_build_stream: torch.Stream, 
        prefetch_exec_w1_stream: torch.Stream, 
        prefetch_exec_w2_stream: torch.Stream, 
        dispatch_stream: torch.Stream, 
        output_stream: torch.Stream, 
        # Prefetcher
        next_gate: Optional[torch.nn.Module],
        next_w13_weight: Optional[torch.Tensor], 
        next_w2_weight: Optional[torch.Tensor], 
        # MoE computation buffers
        w13_weight_buf_odd: Optional[torch.Tensor], 
        w2_weight_buf_odd: Optional[torch.Tensor], 
        w13_weight_buf_even: Optional[torch.Tensor], 
        w2_weight_buf_even: Optional[torch.Tensor],
        # MoE computation handlers
        w13_weight_hdl_odd: Any, 
        w2_weight_hdl_odd: Any, 
        w13_weight_hdl_even: Any, 
        w2_weight_hdl_even: Any,
        next_topk_ids_cpu: Optional[torch.Tensor],
        topk_ids_cpu: Optional[torch.Tensor],
        topk_ids_mask: Optional[torch.Tensor],
        topk_ids_mask_cpu: Optional[torch.Tensor],
        send_map_local: Optional[torch.Tensor],
        send_map_local_gpu: Optional[torch.Tensor],
        send_map_local_len: Optional[torch.Tensor],
        recv_map_local_len: Optional[torch.Tensor],
        recv_experts_cur_rank_tensor: Optional[torch.Tensor],
        recv_experts_cur_rank_tensor_cpu: Optional[torch.Tensor],
        recv_num_experts: Optional[torch.Tensor],
        expert_gpu_mapping: Optional[torch.Tensor],
        remote_gpu_expert_load: Optional[torch.Tensor],
        topk_ids_buf: torch.Tensor,
        topk_weights_buf: torch.Tensor,
        next_topk_ids_buf: torch.Tensor,
        topk_ids_hdl: Any,
        topk_weights_hdl: Any,
        next_topk_ids_hdl: Any,
        hidden_states_for_prefetch_rebalance: Optional[torch.Tensor],
        q_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_ori:
            # NOTE: Original forward (SGLang / EPLB baselines)
            hidden_states, residual, _ = self.forward_ori(
                positions=positions, 
                hidden_states=hidden_states, 
                forward_batch=forward_batch, 
                residual=residual
            )
            return (
                hidden_states,
                residual,
                None,
                None,
                None,
                None,
                None
            )
        if not self.is_correctness:
            # NOTE: Libra forward
            if self.layer_id < q_len:
                (
                    hidden_states, 
                    residual, 
                    recv_experts_next_cur_rank_tensor, 
                    recv_experts_next_cur_rank_tensor_cpu, 
                    recv_num_experts_next,
                    next_expert_gpu_mapping,
                    next_remote_gpu_expert_load,
                ) = self.forward_mid_first(
                    positions=positions, 
                    hidden_states=hidden_states, 
                    forward_batch=forward_batch, 
                    residual=residual,
                    hidden_states_ori_all=None,
                    rebalance_stream=rebalance_stream,
                    prefetch_original_stream=prefetch_original_stream,
                    prefetch_build_stream=prefetch_build_stream,
                    prefetch_exec_w1_stream=prefetch_exec_w1_stream,
                    prefetch_exec_w2_stream=prefetch_exec_w2_stream,
                    dispatch_stream=dispatch_stream,
                    output_stream=output_stream,
                    next_gate=next_gate,
                    next_w13_weight=next_w13_weight,
                    next_w2_weight=next_w2_weight,
                    w13_weight_buf_odd=w13_weight_buf_odd,
                    w2_weight_buf_odd=w2_weight_buf_odd,
                    w13_weight_buf_even=w13_weight_buf_even,
                    w2_weight_buf_even=w2_weight_buf_even,
                    w13_weight_hdl_odd=w13_weight_hdl_odd,
                    w2_weight_hdl_odd=w2_weight_hdl_odd,
                    w13_weight_hdl_even=w13_weight_hdl_even,
                    w2_weight_hdl_even=w2_weight_hdl_even,
                    next_topk_ids_cpu=next_topk_ids_cpu,
                    topk_ids_cpu=topk_ids_cpu,
                    topk_ids_mask=topk_ids_mask,
                    topk_ids_mask_cpu=topk_ids_mask_cpu,
                    send_map_local=send_map_local,
                    send_map_local_gpu=send_map_local_gpu,
                    send_map_local_len=send_map_local_len,
                    recv_map_local_len=recv_map_local_len,
                    recv_experts_cur_rank_tensor=recv_experts_cur_rank_tensor,
                    recv_experts_cur_rank_tensor_cpu=recv_experts_cur_rank_tensor_cpu,
                    recv_num_experts=recv_num_experts,
                    expert_gpu_mapping=expert_gpu_mapping,
                    remote_gpu_expert_load=remote_gpu_expert_load,
                    hidden_states_for_prefetch_rebalance=hidden_states_for_prefetch_rebalance,
                )
            else:
                (
                    hidden_states, 
                    residual, 
                    recv_experts_next_cur_rank_tensor, 
                    recv_experts_next_cur_rank_tensor_cpu, 
                    recv_num_experts_next,
                    next_expert_gpu_mapping,
                    next_remote_gpu_expert_load,
                ) = self.forward_mid(
                    positions=positions, 
                    hidden_states=hidden_states, 
                    forward_batch=forward_batch, 
                    residual=residual,
                    hidden_states_ori_all=None,
                    rebalance_stream=rebalance_stream,
                    prefetch_original_stream=prefetch_original_stream,
                    prefetch_build_stream=prefetch_build_stream,
                    prefetch_exec_w1_stream=prefetch_exec_w1_stream,
                    prefetch_exec_w2_stream=prefetch_exec_w2_stream,
                    dispatch_stream=dispatch_stream,
                    output_stream=output_stream,
                    next_gate=next_gate,
                    next_w13_weight=next_w13_weight,
                    next_w2_weight=next_w2_weight,
                    w13_weight_buf_odd=w13_weight_buf_odd,
                    w2_weight_buf_odd=w2_weight_buf_odd,
                    w13_weight_buf_even=w13_weight_buf_even,
                    w2_weight_buf_even=w2_weight_buf_even,
                    w13_weight_hdl_odd=w13_weight_hdl_odd,
                    w2_weight_hdl_odd=w2_weight_hdl_odd,
                    w13_weight_hdl_even=w13_weight_hdl_even,
                    w2_weight_hdl_even=w2_weight_hdl_even,
                    next_topk_ids_cpu=next_topk_ids_cpu,
                    topk_ids_cpu=topk_ids_cpu,
                    topk_ids_mask=topk_ids_mask,
                    topk_ids_mask_cpu=topk_ids_mask_cpu,
                    send_map_local=send_map_local,
                    send_map_local_gpu=send_map_local_gpu,
                    send_map_local_len=send_map_local_len,
                    recv_map_local_len=recv_map_local_len,
                    recv_experts_cur_rank_tensor=recv_experts_cur_rank_tensor,
                    recv_experts_cur_rank_tensor_cpu=recv_experts_cur_rank_tensor_cpu,
                    recv_num_experts=recv_num_experts,
                    expert_gpu_mapping=expert_gpu_mapping,
                    remote_gpu_expert_load=remote_gpu_expert_load,
                    topk_ids_buf=topk_ids_buf,
                    topk_weights_buf=topk_weights_buf,
                    next_topk_ids_buf=next_topk_ids_buf,
                    topk_ids_hdl=topk_ids_hdl,
                    topk_weights_hdl=topk_weights_hdl,
                    next_topk_ids_hdl=next_topk_ids_hdl,
                    hidden_states_for_prefetch_rebalance=hidden_states_for_prefetch_rebalance,
                )

            return (
                hidden_states, 
                residual, 
                recv_experts_next_cur_rank_tensor, 
                recv_experts_next_cur_rank_tensor_cpu, 
                recv_num_experts_next,
                next_expert_gpu_mapping,
                next_remote_gpu_expert_load,
            )
        else:
            # NOTE: Correctness test: run the original and the Libra path and compare them
            hidden_states_ori = hidden_states.detach().clone().contiguous()
            hidden_states_mid = hidden_states.detach().clone().contiguous()
            residual_ori, residual_mid = None, None
            if residual is not None:
                residual_ori = residual.detach().clone()
                residual_mid = residual.detach().clone()
            hidden_states_ori, residual_ori, hidden_states_ori_all = self.forward_ori(
                positions=positions, 
                hidden_states=hidden_states_ori, 
                forward_batch=forward_batch, 
                residual=residual_ori
            )
            hidden_states_ori_result = hidden_states_ori.detach().clone().contiguous()
            residual_ori_result = residual_ori.detach().clone().contiguous()
            if self.layer_id < q_len:
                (
                    hidden_states_mid, 
                    residual_mid, 
                    recv_experts_next_cur_rank_tensor, 
                    recv_experts_next_cur_rank_tensor_cpu, 
                    recv_num_experts_next,
                    next_expert_gpu_mapping,
                    next_remote_gpu_expert_load,
                ) = self.forward_mid_first(
                    positions=positions, 
                    hidden_states=hidden_states_mid, 
                    forward_batch=forward_batch, 
                    residual=residual_mid,
                    hidden_states_ori_all=hidden_states_ori_all,
                    rebalance_stream=rebalance_stream,
                    prefetch_original_stream=prefetch_original_stream,
                    prefetch_build_stream=prefetch_build_stream,
                    prefetch_exec_w1_stream=prefetch_exec_w1_stream,
                    prefetch_exec_w2_stream=prefetch_exec_w2_stream,
                    dispatch_stream=dispatch_stream,
                    output_stream=output_stream,
                    next_gate=next_gate,
                    next_w13_weight=next_w13_weight,
                    next_w2_weight=next_w2_weight,
                    w13_weight_buf_odd=w13_weight_buf_odd,
                    w2_weight_buf_odd=w2_weight_buf_odd,
                    w13_weight_buf_even=w13_weight_buf_even,
                    w2_weight_buf_even=w2_weight_buf_even,
                    w13_weight_hdl_odd=w13_weight_hdl_odd,
                    w2_weight_hdl_odd=w2_weight_hdl_odd,
                    w13_weight_hdl_even=w13_weight_hdl_even,
                    w2_weight_hdl_even=w2_weight_hdl_even,
                    next_topk_ids_cpu=next_topk_ids_cpu,
                    topk_ids_cpu=topk_ids_cpu,
                    topk_ids_mask=topk_ids_mask,
                    topk_ids_mask_cpu=topk_ids_mask_cpu,
                    send_map_local=send_map_local,
                    send_map_local_gpu=send_map_local_gpu,
                    send_map_local_len=send_map_local_len,
                    recv_map_local_len=recv_map_local_len,
                    recv_experts_cur_rank_tensor=recv_experts_cur_rank_tensor,
                    recv_experts_cur_rank_tensor_cpu=recv_experts_cur_rank_tensor_cpu,
                    recv_num_experts=recv_num_experts,
                    expert_gpu_mapping=expert_gpu_mapping,
                    remote_gpu_expert_load=remote_gpu_expert_load,
                    hidden_states_for_prefetch_rebalance=hidden_states_for_prefetch_rebalance,
                )
            else:
                (
                    hidden_states_mid, 
                    residual_mid, 
                    recv_experts_next_cur_rank_tensor, 
                    recv_experts_next_cur_rank_tensor_cpu, 
                    recv_num_experts_next,
                    next_expert_gpu_mapping,
                    next_remote_gpu_expert_load,
                ) = self.forward_mid(
                    positions=positions, 
                    hidden_states=hidden_states_mid, 
                    forward_batch=forward_batch, 
                    residual=residual_mid,
                    hidden_states_ori_all=hidden_states_ori_all,
                    rebalance_stream=rebalance_stream,
                    prefetch_original_stream=prefetch_original_stream,
                    prefetch_build_stream=prefetch_build_stream,
                    prefetch_exec_w1_stream=prefetch_exec_w1_stream,
                    prefetch_exec_w2_stream=prefetch_exec_w2_stream,
                    dispatch_stream=dispatch_stream,
                    output_stream=output_stream,
                    next_gate=next_gate,
                    next_w13_weight=next_w13_weight,
                    next_w2_weight=next_w2_weight,
                    w13_weight_buf_odd=w13_weight_buf_odd,
                    w2_weight_buf_odd=w2_weight_buf_odd,
                    w13_weight_buf_even=w13_weight_buf_even,
                    w2_weight_buf_even=w2_weight_buf_even,
                    w13_weight_hdl_odd=w13_weight_hdl_odd,
                    w2_weight_hdl_odd=w2_weight_hdl_odd,
                    w13_weight_hdl_even=w13_weight_hdl_even,
                    w2_weight_hdl_even=w2_weight_hdl_even,
                    next_topk_ids_cpu=next_topk_ids_cpu,
                    topk_ids_cpu=topk_ids_cpu,
                    topk_ids_mask=topk_ids_mask,
                    topk_ids_mask_cpu=topk_ids_mask_cpu,
                    send_map_local=send_map_local,
                    send_map_local_gpu=send_map_local_gpu,
                    send_map_local_len=send_map_local_len,
                    recv_map_local_len=recv_map_local_len,
                    recv_experts_cur_rank_tensor=recv_experts_cur_rank_tensor,
                    recv_experts_cur_rank_tensor_cpu=recv_experts_cur_rank_tensor_cpu,
                    recv_num_experts=recv_num_experts,
                    expert_gpu_mapping=expert_gpu_mapping,
                    remote_gpu_expert_load=remote_gpu_expert_load,
                    topk_ids_buf=topk_ids_buf,
                    topk_weights_buf=topk_weights_buf,
                    next_topk_ids_buf=next_topk_ids_buf,
                    topk_ids_hdl=topk_ids_hdl,
                    topk_weights_hdl=topk_weights_hdl,
                    next_topk_ids_hdl=next_topk_ids_hdl,
                    hidden_states_for_prefetch_rebalance=hidden_states_for_prefetch_rebalance,
                )
            hidden_states_ori.copy_(hidden_states_ori_result)
            residual_ori.copy_(residual_ori_result)
            return (
                hidden_states_ori, 
                residual_ori, 
                recv_experts_next_cur_rank_tensor, 
                recv_experts_next_cur_rank_tensor_cpu, 
                recv_num_experts_next,
                next_expert_gpu_mapping,
                next_remote_gpu_expert_load,
            )
    
    def forward_ori(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nvtx.range_push("forward_ori")
        nvtx.range_push("RMSNorm")
        ########################################################################
        # NOTE: Update the expert_map_gpu
        self.mlp.experts.expert_map_gpu_extended.copy_(self.mlp.experts.expert_map_gpu)
        ########################################################################

        ########################################################################
        # NOTE: RMSNorm
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("Attention")
        ########################################################################
        # NOTE: Attention
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        nvtx.range_pop()
        ########################################################################

        ########################################################################
        # NOTE: Dispatch
        # Residual connection
        nvtx.range_push("Dispatch")
        hidden_states += residual
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("MoE")
        ########################################################################
        # NOTE: MoE
        hidden_states = self.mlp(hidden_states, forward_batch, is_ours=True)
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("Combine")
        ########################################################################
        # NOTE: Combine
        if self.local_dp_size > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        ########################################################################
        hidden_states_ori_all = hidden_states.detach().clone()
        nvtx.range_pop()

        nvtx.range_push("Postprocess")
        ########################################################################
        # NOTE: Post-processing => Reduce the first dimension
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        ########################################################################
        nvtx.range_pop()

        nvtx.range_pop()
        return hidden_states, residual, hidden_states_ori_all


    def forward_mid_first(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        hidden_states_ori_all: Optional[torch.Tensor],
        # CUDA Stream
        rebalance_stream: torch.Stream, 
        prefetch_original_stream: torch.Stream, 
        prefetch_build_stream: torch.Stream, 
        prefetch_exec_w1_stream: torch.Stream, 
        prefetch_exec_w2_stream: torch.Stream, 
        dispatch_stream: torch.Stream, 
        output_stream: torch.Stream, 
        next_gate: Optional[torch.nn.Module],
        next_w13_weight: Optional[torch.Tensor], 
        next_w2_weight: Optional[torch.Tensor], 
        w13_weight_buf_odd: Optional[torch.Tensor], 
        w2_weight_buf_odd: Optional[torch.Tensor], 
        w13_weight_buf_even: Optional[torch.Tensor], 
        w2_weight_buf_even: Optional[torch.Tensor], 
        w13_weight_hdl_odd: Any, 
        w2_weight_hdl_odd: Any, 
        w13_weight_hdl_even: Any, 
        w2_weight_hdl_even: Any, 
        next_topk_ids_cpu: Optional[torch.Tensor],
        topk_ids_cpu: Optional[torch.Tensor],
        topk_ids_mask: Optional[torch.Tensor],
        topk_ids_mask_cpu: Optional[torch.Tensor],
        send_map_local: Optional[torch.Tensor],
        send_map_local_gpu: Optional[torch.Tensor],
        send_map_local_len: Optional[torch.Tensor],
        recv_map_local_len: Optional[torch.Tensor],
        # Prefetch info
        recv_experts_cur_rank_tensor: Optional[torch.Tensor],
        recv_experts_cur_rank_tensor_cpu: Optional[torch.Tensor],
        recv_num_experts: int,
        expert_gpu_mapping: Optional[torch.Tensor],
        remote_gpu_expert_load: Optional[torch.Tensor],
        hidden_states_for_prefetch_rebalance: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nvtx.range_push("forward_mid_first")
        ########################################################################
        
        # CUDA Events
        attn_done_event = torch.cuda.Event()
        moe_local_done_event = torch.cuda.Event()
        moe_remote_done_event = torch.cuda.Event()
        prefetch_start_event = torch.cuda.Event()
        prefetch_done_event = torch.cuda.Event()
        dispatch_start_event = torch.cuda.Event()
        dispatch_done_event = torch.cuda.Event()
        output_done_event = torch.cuda.Event()
        ########################################################################
        nvtx.range_push("RMSNorm_first")
        ########################################################################
        # NOTE: RMSNorm & Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        nvtx.range_pop()
        nvtx.range_push("Attention_first")
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("Dispatch_first")
        ########################################################################
        # NOTE: Dispatch
        # Residual connection
        hidden_states += residual
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states_for_prefetch_rebalance.copy_(hidden_states)
        dispatch_done_event.record()
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("MoE_first")
        ########################################################################
        # NOTE: MoE
        hidden_states = self.mlp(hidden_states, forward_batch, is_ours=True)
        moe_remote_done_event.record()
        ########################################################################
        nvtx.range_pop()
        ########################################################################
        # NOTE: Prefetch => Build and Execute Plans
        with torch.cuda.stream(prefetch_build_stream):
            prefetch_build_stream.wait_event(dispatch_done_event)
            nvtx.range_push("Prefetch_first")
            recv_experts_next_cur_rank_tensor = None
            recv_experts_next_cur_rank_tensor_cpu = None
            recv_num_experts_next = None
            next_expert_gpu_mapping = None
            next_remote_gpu_expert_load = None
            if next_gate is not None:
                next_router_logits, _ = next_gate(hidden_states_for_prefetch_rebalance)
                next_topk_output = self.mlp.topk(hidden_states_for_prefetch_rebalance, next_router_logits)
                # WARNING: D2H with Sync
                next_topk_ids_cpu.copy_(next_topk_output.topk_ids, non_blocking=False)

                next_expert_gpu_mapping, next_remote_gpu_expert_load, recv_experts_next_cur_rank_tensor_cpu = prefetch(
                    next_topk_ids=next_topk_ids_cpu,
                    num_experts=self.mlp.experts.num_experts,
                    seq_len=hidden_states_for_prefetch_rebalance.shape[0] // self.local_dp_size,
                    num_gpus=self.local_dp_size,
                    cur_rank=self.local_dp_rank,
                    send_map_local_out=send_map_local,
                    send_map_local_len_out=send_map_local_len,
                    recv_map_local_len_out=recv_map_local_len,
                    max_dup_per_gpu=EXPERT_DUP_FACTOR_N,
                    n_local_dup_per_gpu=EXPERT_DUP_FACTOR_L,
                    max_send_len=EXPERT_DUP_MAX,
                    force_max_dup=False
                )

                recv_num_experts_next = recv_experts_next_cur_rank_tensor_cpu.numel()
                # WARNING: H2D Sync
                recv_experts_next_cur_rank_tensor = recv_experts_next_cur_rank_tensor_cpu.to(hidden_states_for_prefetch_rebalance.device)

                # Build P2P plans for w13 and w2
                cum = compute_cum_recv_map_len(recv_map_local_len)
                prefetch_plan = P2PPlan(
                    w13_weight=next_w13_weight,
                    w2_weight=next_w2_weight,
                    w13_weight_hdl=w13_weight_hdl_odd if self.layer_id % 2 == 0 else w13_weight_hdl_even,
                    w2_weight_hdl=w2_weight_hdl_odd if self.layer_id % 2 == 0 else w2_weight_hdl_even,
                    send_map=send_map_local.clone(),
                    send_map_len=send_map_local_len.clone(),
                    recv_map_len=recv_map_local_len.clone(),
                    cum_recv_map_len=cum,
                    dp_size=self.local_dp_size,
                    dp_rank=self.local_dp_rank,
                    w13_weight_buf_shape=w13_weight_buf_even.shape,
                    w2_weight_buf_shape=w2_weight_buf_even.shape
                )
                num_ori_experts = self.mlp.experts.num_local_experts
                for i in range(prefetch_plan.dp_size):
                    rows = int(prefetch_plan.send_map_len[self.local_dp_rank, i].item())
                    if rows == 0 or i == self.local_dp_rank:
                        continue
                    dst_off = int(prefetch_plan.cum_recv_map_len[i, self.local_dp_rank].item())
                    for k in range(rows):
                        expert_id = int(prefetch_plan.send_map[self.local_dp_rank, i, k].item())
                        # 1-row view [1, D]
                        src_w13_row = prefetch_plan.w13_weight.narrow(0, expert_id, 1)
                        src_w2_row = prefetch_plan.w2_weight.narrow(0, expert_id, 1)
                        peer_w13_buf = prefetch_plan.w13_weight_hdl.get_buffer(
                            i, 
                            prefetch_plan.w13_weight_buf_shape, 
                            dtype=prefetch_plan.w13_weight.dtype
                        )
                        peer_w2_buf = prefetch_plan.w2_weight_hdl.get_buffer(
                            i, 
                            prefetch_plan.w2_weight_buf_shape, 
                            dtype=prefetch_plan.w2_weight.dtype
                        )
                        peer_w13_buf.narrow(0, num_ori_experts + dst_off + k, 1).copy_(src_w13_row)
                        peer_w2_buf.narrow(0, num_ori_experts + dst_off + k, 1).copy_(src_w2_row)
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("Combine_first")
        ########################################################################
        # NOTE: Combine
        if self.local_dp_size > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        ########################################################################
        nvtx.range_pop()
        ########################################################################
        # NOTE: Copy the original weights to large buffers
        with torch.cuda.stream(prefetch_original_stream):
            assert w13_weight_buf_odd.is_contiguous(), "w13_weight_buf_odd is not contiguous"
            assert w2_weight_buf_odd.is_contiguous(), "w2_weight_buf_odd is not contiguous"
            assert w13_weight_buf_even.is_contiguous(), "w13_weight_buf_even is not contiguous"
            assert w2_weight_buf_even.is_contiguous(), "w2_weight_buf_even is not contiguous"
            # Stage the next layer's resident experts in rows [0:num_rows) of the other-parity buffer (row 0 first, then the rest).
            num_rows = next_w13_weight.shape[0]
            prefetch_original_stream.wait_event(output_done_event)
            nvtx.range_push("Copy Buffer_first")
            step = 1
            if self.layer_id % 2 == 0:
                w13_weight_buf_odd[:step].copy_(next_w13_weight[:step])
                w13_weight_buf_odd[step:num_rows].copy_(next_w13_weight[step:num_rows])
                w2_weight_buf_odd[:step].copy_(next_w2_weight[:step])
                w2_weight_buf_odd[step:num_rows].copy_(next_w2_weight[step:num_rows])
            else:
                w13_weight_buf_even[:step].copy_(next_w13_weight[:step])
                w13_weight_buf_even[step:num_rows].copy_(next_w13_weight[step:num_rows])
                w2_weight_buf_even[:step].copy_(next_w2_weight[:step])
                w2_weight_buf_even[step:num_rows].copy_(next_w2_weight[step:num_rows])
        ########################################################################
        nvtx.range_pop()
        ########################################################################
        # NOTE: Correctness
        if hidden_states_ori_all is not None:
            wrong_token_ids = find_rows_with_large_diff(hidden_states_ori_all, hidden_states, threshold=0.1, min_count=1)
            wrong_hidden_states = hidden_states[wrong_token_ids]
            true_hidden_states = hidden_states_ori_all[wrong_token_ids]
            if self.local_dp_rank == 0:
                logger.info(f"forward_mid_first, wrong_token_ids({wrong_token_ids.shape}): {wrong_token_ids}")

        ########################################################################
        nvtx.range_push("Postprocess_first")
        ########################################################################
        # NOTE: Post-processing => Reduce the first dimension
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        ########################################################################

        torch.cuda.current_stream().wait_stream(prefetch_build_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w1_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w2_stream)

        nvtx.range_pop()
        nvtx.range_pop()

        return (
            hidden_states, 
            residual, 
            recv_experts_next_cur_rank_tensor, 
            recv_experts_next_cur_rank_tensor_cpu, 
            recv_num_experts_next,
            next_expert_gpu_mapping,
            next_remote_gpu_expert_load,
        )


    def forward_mid(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        hidden_states_ori_all: Optional[torch.Tensor],
        # CUDA Stream
        rebalance_stream: torch.Stream, 
        prefetch_original_stream: torch.Stream, 
        prefetch_build_stream: torch.Stream, 
        prefetch_exec_w1_stream: torch.Stream, 
        prefetch_exec_w2_stream: torch.Stream, 
        dispatch_stream: torch.Stream, 
        output_stream: torch.Stream, 
        next_gate: Optional[torch.nn.Module],
        next_w13_weight: Optional[torch.Tensor], 
        next_w2_weight: Optional[torch.Tensor], 
        w13_weight_buf_odd: Optional[torch.Tensor], 
        w2_weight_buf_odd: Optional[torch.Tensor], 
        w13_weight_buf_even: Optional[torch.Tensor], 
        w2_weight_buf_even: Optional[torch.Tensor], 
        w13_weight_hdl_odd: Any, 
        w2_weight_hdl_odd: Any, 
        w13_weight_hdl_even: Any, 
        w2_weight_hdl_even: Any, 
        next_topk_ids_cpu: Optional[torch.Tensor],
        topk_ids_cpu: Optional[torch.Tensor],
        topk_ids_mask: Optional[torch.Tensor],
        topk_ids_mask_cpu: Optional[torch.Tensor],
        send_map_local: Optional[torch.Tensor],
        send_map_local_gpu: Optional[torch.Tensor],
        send_map_local_len: Optional[torch.Tensor],
        recv_map_local_len: Optional[torch.Tensor],
        # Prefetch info
        recv_experts_cur_rank_tensor: Optional[torch.Tensor],
        recv_experts_cur_rank_tensor_cpu: Optional[torch.Tensor],
        recv_num_experts: int,
        expert_gpu_mapping: Optional[torch.Tensor],
        remote_gpu_expert_load: Optional[torch.Tensor],
        topk_ids_buf: torch.Tensor,
        topk_weights_buf: torch.Tensor,
        next_topk_ids_buf: torch.Tensor,
        topk_ids_hdl: Any,
        topk_weights_hdl: Any,
        next_topk_ids_hdl: Any,
        hidden_states_for_prefetch_rebalance: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nvtx.range_push("forward_mid")
        nvtx.range_push("RMSNorm")
        ########################################################################
        
        # CUDA Events
        attn_start_1_event = torch.cuda.Event()
        attn_start_2_event = torch.cuda.Event()
        attn_done_event = torch.cuda.Event()
        moe_local_done_event = torch.cuda.Event()
        moe_remote_done_event = torch.cuda.Event()
        prefetch_start_event = torch.cuda.Event()
        prefetch_done_event = torch.cuda.Event()
        dispatch_start_event = torch.cuda.Event()
        dispatch_done_event = torch.cuda.Event()
        output_done_event = torch.cuda.Event()
        local_topk_done_event = torch.cuda.Event()
        broadcast_done_event = torch.cuda.Event()

        hidden_states_device = hidden_states.device
        
        local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
        zero_offset = torch.zeros(
            (), device=hidden_states_device, dtype=torch.int32
        )
        topk_ids_mask_cpu.fill_(0)
        

        ########################################################################

        ########################################################################
        # NOTE: Update the expert_map_gpu
        if recv_experts_cur_rank_tensor_cpu is not None:
            self.mlp.experts.expert_map_gpu_extended.copy_(self.mlp.experts.expert_map_gpu)
            new_local_ids = torch.arange(
                self.mlp.experts.num_local_experts, 
                self.mlp.experts.num_local_experts + recv_experts_cur_rank_tensor_cpu.numel(), 
                device=hidden_states_device, 
                dtype=self.mlp.experts.expert_map_gpu.dtype
            )
            self.mlp.experts.expert_map_gpu_extended[recv_experts_cur_rank_tensor] = new_local_ids
        ########################################################################

        ########################################################################
        # NOTE: RMSNorm + Attention
        attn_start_1_event.record()
        # prepare_attn => input_layernorm
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        nvtx.range_pop()
        nvtx.range_push("Attention")
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                event=attn_start_1_event,
            )
        attn_done_event.record()
        nvtx.range_pop()
        ########################################################################
        
        ########################################################################
        # NOTE: Dispatch
        with torch.cuda.stream(dispatch_stream):
            dispatch_stream.wait_event(attn_done_event)
            nvtx.range_push("Dispatch")
            # Residual connection
            hidden_states += residual
            local_hidden_states = hidden_states.detach().clone()

            # RMSNorm
            local_hidden_states = self.layer_communicator.post_attention_layernorm(local_hidden_states)

            # router_logits: (num_tokens, n_experts)
            local_router_logits, _ = self.mlp.gate(local_hidden_states)
            local_topk_output = self.mlp.topk(local_hidden_states, local_router_logits)
            topk_ids_buf.copy_(local_topk_output.topk_ids)
            topk_weights_buf.copy_(local_topk_output.topk_weights)
            if next_gate is not None:
                next_router_logits, _ = next_gate(local_hidden_states)
                next_topk_output = self.mlp.topk(local_hidden_states, next_router_logits)
                next_topk_ids_buf.copy_(next_topk_output.topk_ids)
            dispatch_start_event.record()

            # Dispatch and post_attention_layernorm
            hidden_states, residual = self.layer_communicator.prepare_mlp( 
                hidden_states, residual, forward_batch
            )
            hidden_states_for_prefetch_rebalance.copy_(hidden_states)
            dispatch_done_event.record()
            nvtx.range_pop()
        ########################################################################

        ########################################################################
        # NOTE: MoE_local
        torch.cuda.current_stream().wait_event(dispatch_start_event)
        nvtx.range_push("MoE Local")
        if self.imbalance_ratio:
            local_topk_ids_processed = self.mlp.experts.expert_map_gpu_extended[local_topk_output.topk_ids]
            num_local_topk_ids_processed = (local_topk_ids_processed > -1).sum()
        local_hidden_states = self.mlp(local_hidden_states, 
                                forward_batch, 
                                is_ours=True, 
                                topk_ids_mask=None, 
                                w1_buf=w13_weight_buf_even if self.layer_id % 2 == 0 else w13_weight_buf_odd,
                                w2_buf=w2_weight_buf_even if self.layer_id % 2 == 0 else w2_weight_buf_odd,
                                topk_output=local_topk_output
                            )
        moe_local_done_event.record()
        ########################################################################
        nvtx.range_pop()
        ########################################################################
        # NOTE: Rebalancer + MoE_remote
        with torch.cuda.stream(rebalance_stream):
            # Rebalance: adaptive token sharding of the remote tokens (CPU)
            rebalance_stream.wait_event(dispatch_start_event)
            # All-gather the routing results (and the next-layer prediction) of all ranks
            if next_gate is None:
                topk_ids, topk_weights = broadcast_topk(
                    topk_ids_buf=topk_ids_buf,
                    topk_weights_buf=topk_weights_buf,
                    topk_ids_hdl=topk_ids_hdl,
                    topk_weights_hdl=topk_weights_hdl,
                    tp_rank=self.local_dp_rank,
                    tp_size=self.local_dp_size,
                    topk=self.mlp.experts.top_k,
                    stream=rebalance_stream,
                    device=hidden_states_device
                )
            else:
                topk_ids, topk_weights, next_topk_ids = broadcast_topk_with_next(
                    topk_ids_buf=topk_ids_buf,
                    topk_weights_buf=topk_weights_buf,
                    next_topk_ids_buf=next_topk_ids_buf,
                    topk_ids_hdl=topk_ids_hdl,
                    topk_weights_hdl=topk_weights_hdl,
                    next_topk_ids_hdl=next_topk_ids_hdl,
                    tp_rank=self.local_dp_rank,
                    tp_size=self.local_dp_size,
                    topk=self.mlp.experts.top_k,
                    stream=rebalance_stream,
                    device=hidden_states_device
                )
                # Asynchronous D2H copy
                next_topk_ids_cpu.copy_(next_topk_ids, non_blocking=True)
            broadcast_done_event.record()
            topk_output = StandardTopKOutput(topk_ids=topk_ids, topk_weights=topk_weights, router_logits=None)
            # WARNING: D2H with sync
            topk_ids_cpu.copy_(topk_ids, non_blocking=False)


            rebalance(
                topk_ids=topk_ids_cpu,
                expert_gpu_mapping=expert_gpu_mapping,
                init_remote_gpu_expert_load=remote_gpu_expert_load,
                num_experts=self.mlp.experts.num_experts,
                seq_len=local_hidden_states.shape[0],
                num_gpus=self.local_dp_size,
                cur_rank=self.local_dp_rank,
                topk_ids_mask_out=topk_ids_mask_cpu,
                min_dup_experts_tokens=MIN_TOKEN_LIMIT
            )
            # Asynchronous H2D copy of the mask
            topk_ids_mask.copy_(topk_ids_mask_cpu, non_blocking=True)

            # MoE_remote
            rebalance_stream.wait_event(moe_local_done_event)
            rebalance_stream.wait_event(dispatch_done_event)
            nvtx.range_push("MoE Remote")

            if self.imbalance_ratio and self.local_dp_rank == 0:
                logger.info(f'======================Layer: {self.layer_id}==========================')
            if self.imbalance_ratio:
                remote_topk_ids_processed = self.mlp.experts.expert_map_gpu_extended[topk_output.topk_ids]
                remote_topk_ids_processed = remote_topk_ids_processed.masked_fill(topk_ids_mask == 0, -1)
                num_remote_topk_ids_processed = (remote_topk_ids_processed > -1).sum()
                num_remote_topk_ids_processed += num_local_topk_ids_processed

                num_total = topk_output.topk_ids.numel()

                #######################################
                # Compare number of local tokens
                original_topk_ids_processed = self.mlp.experts.expert_map_gpu[topk_output.topk_ids]
                num_original_topk_ids_processed = (original_topk_ids_processed > -1).sum()
                original_local_topk_ids_processed = original_topk_ids_processed[self.local_dp_rank * local_hidden_states.shape[0]: (self.local_dp_rank+1)
                                                                                * local_hidden_states.shape[0]]
                num_original_local_topk_ids_processed = (original_local_topk_ids_processed > -1).sum()

                wrong_local_ids = find_rows_with_large_diff(local_topk_ids_processed, original_local_topk_ids_processed, threshold=0.1, min_count=1)
                ori_ids = original_local_topk_ids_processed[wrong_local_ids]
                wrong_ids = local_topk_ids_processed[wrong_local_ids]
                if self.local_dp_rank == 0:
                    logger.info(f"wrong_local_ids({wrong_local_ids.shape}): {wrong_local_ids}")
                    logger.info(f"ori_ids({ori_ids.shape}): {ori_ids}")
                    logger.info(f"wrong_ids({wrong_ids.shape}): {wrong_ids}")

                #######################################

                assert num_remote_topk_ids_processed == num_original_topk_ids_processed, "Processed number of tokens must be equal"
            remote_hidden_states_ours = self.mlp(hidden_states_for_prefetch_rebalance, 
                                                    forward_batch, 
                                                    is_ours=True, 
                                                    topk_ids_mask=topk_ids_mask, 
                                                    w1_buf=w13_weight_buf_even if self.layer_id % 2 == 0 else w13_weight_buf_odd,
                                                    w2_buf=w2_weight_buf_even if self.layer_id % 2 == 0 else w2_weight_buf_odd,
                                                    topk_output=topk_output
                                                )
            moe_remote_done_event.record()
            nvtx.range_pop()

        ########################################################################

        ########################################################################
        # NOTE: Prefetch => Build Plans for w2 and w13
        with torch.cuda.stream(prefetch_build_stream):
            prefetch_build_stream.wait_event(broadcast_done_event)
            prefetch_build_stream.wait_event(dispatch_done_event)
            nvtx.range_push("Prefetch")
            recv_experts_next_cur_rank_tensor = None
            recv_experts_next_cur_rank_tensor_cpu = None
            recv_num_experts_next = None
            next_expert_gpu_mapping = None
            next_remote_gpu_expert_load = None
            if next_gate is not None:
                next_expert_gpu_mapping, next_remote_gpu_expert_load, recv_experts_next_cur_rank_tensor_cpu = prefetch(
                    next_topk_ids=next_topk_ids_cpu,
                    num_experts=self.mlp.experts.num_experts,
                    seq_len=hidden_states_for_prefetch_rebalance.shape[0] // self.local_dp_size,
                    num_gpus=self.local_dp_size,
                    cur_rank=self.local_dp_rank,
                    send_map_local_out=send_map_local,
                    send_map_local_len_out=send_map_local_len,
                    recv_map_local_len_out=recv_map_local_len,
                    max_dup_per_gpu=EXPERT_DUP_FACTOR_N,
                    n_local_dup_per_gpu=EXPERT_DUP_FACTOR_L,
                    max_send_len=EXPERT_DUP_MAX,
                    force_max_dup=False
                )

                recv_num_experts_next = recv_experts_next_cur_rank_tensor_cpu.numel()
                # WARNING: H2D Sync
                recv_experts_next_cur_rank_tensor = recv_experts_next_cur_rank_tensor_cpu.to(hidden_states_for_prefetch_rebalance.device)

                # Build P2P plans for w13 and w2
                cum = compute_cum_recv_map_len(recv_map_local_len)
                prefetch_plan = P2PPlan(
                    w13_weight=next_w13_weight,
                    w2_weight=next_w2_weight,
                    w13_weight_hdl=w13_weight_hdl_odd if self.layer_id % 2 == 0 else w13_weight_hdl_even,
                    w2_weight_hdl=w2_weight_hdl_odd if self.layer_id % 2 == 0 else w2_weight_hdl_even,
                    send_map=send_map_local.clone(),
                    send_map_len=send_map_local_len.clone(),
                    recv_map_len=recv_map_local_len.clone(),
                    cum_recv_map_len=cum,
                    dp_size=self.local_dp_size,
                    dp_rank=self.local_dp_rank,
                    w13_weight_buf_shape=w13_weight_buf_even.shape,
                    w2_weight_buf_shape=w2_weight_buf_even.shape
                )
                num_ori_experts = self.mlp.experts.num_local_experts
                for i in range(prefetch_plan.dp_size):
                    rows = int(prefetch_plan.send_map_len[self.local_dp_rank, i].item())
                    if rows == 0 or i == self.local_dp_rank:
                        continue
                    dst_off = int(prefetch_plan.cum_recv_map_len[i, self.local_dp_rank].item())
                    for k in range(rows):
                        expert_id = int(prefetch_plan.send_map[self.local_dp_rank, i, k].item())
                        # 1-row view [1, D]
                        src_w13_row = prefetch_plan.w13_weight.narrow(0, expert_id, 1)
                        src_w2_row = prefetch_plan.w2_weight.narrow(0, expert_id, 1)
                        peer_w13_buf = prefetch_plan.w13_weight_hdl.get_buffer(
                            i, 
                            prefetch_plan.w13_weight_buf_shape, 
                            dtype=prefetch_plan.w13_weight.dtype
                        )
                        peer_w2_buf = prefetch_plan.w2_weight_hdl.get_buffer(
                            i, 
                            prefetch_plan.w2_weight_buf_shape, 
                            dtype=prefetch_plan.w2_weight.dtype
                        )
                        peer_w13_buf.narrow(0, num_ori_experts + dst_off + k, 1).copy_(src_w13_row)
                        peer_w2_buf.narrow(0, num_ori_experts + dst_off + k, 1).copy_(src_w2_row)
        ########################################################################

        ########################################################################
        # NOTE: Add outputs/Combine
        torch.cuda.current_stream().wait_stream(rebalance_stream)
        torch.cuda.current_stream().wait_stream(dispatch_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w1_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w2_stream)
        nvtx.range_pop()
        nvtx.range_push("Combine")

        # Combine outputs of local tokens and remote tokens
        hidden_states = fused_add_without_zero_init_v2(
            hidden_states=hidden_states, 
            local_hidden_states=local_hidden_states, 
            remote_hidden_states_ours=remote_hidden_states_ours, 
            zero_offset=zero_offset,
            local_start_pos=local_start_pos,
            local_num_tokens=local_num_tokens
        )
        output_done_event.record()

        if self.local_dp_size > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("Copy Buffer")
        ########################################################################
        # NOTE: Copy the original weights to large buffers
        with torch.cuda.stream(prefetch_original_stream):
            if next_gate is not None:
                assert w13_weight_buf_odd.is_contiguous(), "w13_weight_buf_odd is not contiguous"
                assert w2_weight_buf_odd.is_contiguous(), "w2_weight_buf_odd is not contiguous"
                assert w13_weight_buf_even.is_contiguous(), "w13_weight_buf_even is not contiguous"
                assert w2_weight_buf_even.is_contiguous(), "w2_weight_buf_even is not contiguous"
                # Stage the next layer's resident experts in rows [0:num_rows) of the other-parity buffer (row 0 first, then the rest).
                num_rows = next_w13_weight.shape[0]
                prefetch_original_stream.wait_event(output_done_event)
                step = 1
                if self.layer_id % 2 == 0:
                    w13_weight_buf_odd[:step].copy_(next_w13_weight[:step])
                    w13_weight_buf_odd[step:num_rows].copy_(next_w13_weight[step:num_rows])
                    w2_weight_buf_odd[:step].copy_(next_w2_weight[:step])
                    w2_weight_buf_odd[step:num_rows].copy_(next_w2_weight[step:num_rows])
                else:
                    w13_weight_buf_even[:step].copy_(next_w13_weight[:step])
                    w13_weight_buf_even[step:num_rows].copy_(next_w13_weight[step:num_rows])
                    w2_weight_buf_even[:step].copy_(next_w2_weight[:step])
                    w2_weight_buf_even[step:num_rows].copy_(next_w2_weight[step:num_rows])
        ########################################################################
        nvtx.range_pop()
        nvtx.range_push("Postprocess")
        ########################################################################
        # NOTE: Correctness
        if hidden_states_ori_all is not None:
            wrong_token_ids = find_rows_with_large_diff(hidden_states_ori_all, hidden_states, threshold=0.1, min_count=1)
            wrong_hidden_states = hidden_states[wrong_token_ids]
            true_hidden_states = hidden_states_ori_all[wrong_token_ids]
            if self.local_dp_rank == 0:
                logger.info(f"wrong_hidden_states({wrong_hidden_states.shape}): {wrong_hidden_states}")
                logger.info(f"true_hidden_states({true_hidden_states.shape}): {true_hidden_states}")
        ########################################################################

        ########################################################################
        # NOTE: Post-processing
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        ########################################################################

        torch.cuda.current_stream().wait_stream(prefetch_build_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w1_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w2_stream)

        nvtx.range_pop()
        nvtx.range_pop()

        return (
            hidden_states, 
            residual, 
            recv_experts_next_cur_rank_tensor, 
            recv_experts_next_cur_rank_tensor_cpu, 
            recv_num_experts_next,
            next_expert_gpu_mapping,
            next_remote_gpu_expert_load,
        )

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        state.hidden_states_mlp_output = self.mlp(hidden_states, state.forward_batch)

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output


class Qwen3MoeModel(Qwen2MoeModel):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3MoeDecoderLayer,
            alt_stream=alt_stream,
        )


class Qwen3MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = get_moe_impl_class().make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # TODO mimic deepseek
        self.routed_experts_weights_of_layer = {
            layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
            for layer_id in range(self.start_layer, self.end_layer)
            if isinstance(self.model.layers[layer_id].mlp, Qwen3MoeSparseMoeBlock)
        }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = Qwen3MoeForCausalLM

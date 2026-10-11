# Copyright 2025-2026 SGLang Team
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

"""Inference-only GLM-4.5 model compatible with HuggingFace weights"""

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import (
    DeepEPMoE,
    get_moe_impl_class,
    use_flashinfer_trtllm_moe,
)
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from sglang.srt.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.utils import (
    BumpAllocator,
    DeepEPMode,
    LazyValue,
    add_prefix,
    bind_or_assign,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    get_int_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    use_intel_amx_backend,
)


import os
from torch.cuda import nvtx
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import (
    get_dp_local_info,
    get_local_attention_dp_rank
)
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
    create_large_buffer_symm,
    get_libra_scheme,
)
from sglang.load_balance import prefetch, rebalance
from sglang.srt.layers.moe.ep_moe.kernels import fused_add_without_zero_init_v2
from sglang.srt.layers.moe.topk import StandardTopKOutput
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from collections import deque
from dataclasses import dataclass

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()

if _is_cuda:
    from sgl_kernel import dsv3_router_gemm
elif _is_cpu and _is_cpu_amx_available:
    pass

logger = logging.getLogger(__name__)


class Glm4MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_batch=None, can_fuse_mlp_allreduce=False):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, can_fuse_mlp_allreduce=can_fuse_mlp_allreduce)
        return x


class Glm4MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        partial_rotary_factor: float = 0.5,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-05,
        attention_bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        use_qk_norm: bool = False,
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
        self.use_qk_norm = use_qk_norm
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
            bias=False,
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
            partial_rotary_factor=partial_rotary_factor,
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

        if self.use_qk_norm:
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
        if self.use_qk_norm:
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
            event=None,
        )
        return self.forward_core(s)


class Glm4MoeGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts))
        )
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])

    def forward(self, hidden_states):
        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states,
                self.weight,
                None,  # bias
                True,  # is_vnni
            )

        # NOTE: For some unknown reason, router_gemm seems degrade accept length.
        if (
            _is_cuda
            and not self.is_nextn
            and hidden_states.shape[0] < 4
            and hidden_states.shape[1] == 7168
            and self.weight.shape[0] == 256
            and _device_sm >= 90
        ):
            logits = dsv3_router_gemm(hidden_states, self.weight).to(
                hidden_states.dtype
            )
        else:
            logits = F.linear(hidden_states, self.weight, None)

        return logits


class Glm4MoeSparseMoeBlock(DeepseekV2MoE):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if global_server_args_dict["disable_shared_experts_fusion"]
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = Glm4MoeGate(
            config=config, prefix=add_prefix("gate", prefix), is_nextn=is_nextn
        )
        
        self.topk = (
            TopK(
                top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
                renormalize=config.norm_topk_prob,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                num_fused_shared_experts=self.num_fused_shared_experts,
                topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
            )
            if not use_flashinfer_trtllm_moe
            else None
        )

        self.experts = get_moe_impl_class()(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
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
            **(
                dict(
                    renormalize=config.norm_topk_prob,
                    use_grouped_topk=True,
                    num_expert_group=config.n_group,
                    num_fused_shared_experts=self.num_fused_shared_experts,
                    topk_group=config.topk_group,
                    correction_bias=self.gate.e_score_correction_bias,
                )
                if use_flashinfer_trtllm_moe
                else {}
            ),
        )

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        # self.shared_experts_weight_block_size = None
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if global_server_args_dict["enable_deepep_moe"]
                    else {}
                ),
            )
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            )
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )

        self.top_k = config.num_experts_per_tok

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"]
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

            self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.n_routed_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,
                return_recv_hook=True,
            )

        self._enable_deepep_moe = global_server_args_dict["enable_deepep_moe"]


class Glm4MoeDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.layer_id = layer_id
        self.self_attn = Glm4MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            use_qk_norm=config.use_qk_norm,
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)

        num_layers = 1 if is_nextn else config.num_hidden_layers
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=num_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Glm4MoeSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
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

        
        self.local_dp_size = get_local_attention_dp_size()
        self.local_dp_rank = get_local_attention_dp_rank()
        self.is_correctness = False
        self.is_ori = get_libra_scheme() in ("sglang", "eplb")
        self.imbalance_ratio = False

    # Dispatches to forward_ori (baseline), forward_mid_first (first MoE layer) or forward_mid (Libra).
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
            # NOTE: Baseline forward (SGLang / EPLB)
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
            if self.layer_id < self.config.first_k_dense_replace:
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
                    None,
                )
            elif self.layer_id < self.config.first_k_dense_replace + q_len:
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
            # NOTE: Correctness test (runs the baseline and the Libra path and compares them)
            hidden_states_ori = hidden_states.detach().clone().contiguous()
            hidden_states_mid = hidden_states.detach().clone().contiguous()
            residual_ori, residual_mid = None, None
            if residual is not None:
                residual_ori = residual.detach().clone().contiguous()
                residual_mid = residual.detach().clone().contiguous()
            hidden_states_ori, residual_ori, hidden_states_ori_all = self.forward_ori(
                positions=positions, 
                hidden_states=hidden_states_ori, 
                forward_batch=forward_batch, 
                residual=residual_ori
            )
            hidden_states_ori_result = hidden_states_ori.detach().clone().contiguous()
            residual_ori_result = residual_ori.detach().clone().contiguous()

            if self.layer_id < self.config.first_k_dense_replace:

                hidden_states_mid, residual_mid, _ = self.forward_ori(
                    positions=positions, 
                    hidden_states=hidden_states_mid, 
                    forward_batch=forward_batch, 
                    residual=residual_mid
                )
                return (
                    hidden_states_mid, 
                    residual_mid, 
                    None, 
                    None, 
                    None,
                    None,
                    None,
                )
            elif self.layer_id < self.config.first_k_dense_replace + q_len:
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
    ) -> torch.Tensor:
        nvtx.range_push("forward_ori")
        nvtx.range_push("RMSNorm")
        ########################################################################
        # NOTE: Update the expert_map_gpu
        if isinstance(self.mlp, Glm4MoeSparseMoeBlock):
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
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        ########################################################################
        nvtx.range_pop()

        ########################################################################
        # NOTE: Dispatch
        # Residual connection
        nvtx.range_push("Dispatch")
        hidden_states += residual
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        nvtx.range_pop()
        ########################################################################

        nvtx.range_push("MoE")
        ########################################################################
        # NOTE: MoE
        hidden_states = self.mlp(hidden_states, forward_batch)
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
        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype
        ########################################################################
        nvtx.range_push("RMSNorm_first")
        ########################################################################
        # NOTE: RMSNorm & Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        nvtx.range_pop()
        nvtx.range_push("Attention_first")
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
        hidden_states = self.mlp(hidden_states, forward_batch, is_ours=True, is_remote=True)
        output_done_event.record()
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
                next_router_logits = next_gate(hidden_states_for_prefetch_rebalance)
                next_topk_output = self.mlp.topk(hidden_states_for_prefetch_rebalance, next_router_logits)
                # WARNING: D2H with sync
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
                # WARNING: H2D with sync
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
                logger.info(f"forward_mid_first, wrong_hidden_states({wrong_hidden_states.shape}): {wrong_hidden_states}")
                logger.info(f"forward_mid_first, true_hidden_states({true_hidden_states.shape}): {true_hidden_states}")

        ########################################################################

        nvtx.range_push("Postprocess_first")
        ########################################################################
        # NOTE: Post-processing => Reduce the first dimension
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        torch.cuda.current_stream().wait_stream(prefetch_build_stream)

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
        topk_1_done_event = torch.cuda.Event()

        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype

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
        # NOTE: RMSNorm
        attn_start_1_event.record()
        # prepare_attn => input_layernorm
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
                event=None,
            )
        # Residual connection
        hidden_states += residual
        local_hidden_states = hidden_states.detach().clone()
        attn_done_event.record()
        nvtx.range_pop()
        ########################################################################

        ########################################################################
        # NOTE: Top-k of the current layer and lookahead prediction for the next layer
        with torch.cuda.stream(output_stream):
            output_stream.wait_event(attn_done_event)
            nvtx.range_push("Dispatch")
            # RMSNorm
            local_hidden_states = self.layer_communicator.post_attention_layernorm(local_hidden_states)

            # router_logits: (num_tokens, n_experts)
            local_router_logits = self.mlp.gate(local_hidden_states)
            local_topk_output = self.mlp.topk(local_hidden_states, local_router_logits)
            topk_ids_buf.copy_(local_topk_output.topk_ids)
            topk_weights_buf.copy_(local_topk_output.topk_weights)
            topk_1_done_event.record()
            if next_gate is not None:
                next_router_logits = next_gate(local_hidden_states)
                next_topk_output = self.mlp.topk(local_hidden_states, next_router_logits)
                next_topk_ids_buf.copy_(next_topk_output.topk_ids)
            dispatch_start_event.record()
            nvtx.range_pop()

        ########################################################################

        ########################################################################
        # NOTE: MoE_local
        torch.cuda.current_stream().wait_event(dispatch_start_event)
        torch.cuda.current_stream().wait_stream(prefetch_original_stream)
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
                                topk_output=local_topk_output,
                                is_remote=False
                            )
        moe_local_done_event.record()
        ########################################################################
        nvtx.range_pop()
        ########################################################################
        # NOTE: Dispatch
        with torch.cuda.stream(dispatch_stream):
            dispatch_stream.wait_event(topk_1_done_event)

            # Dispatch and post_attention_layernorm
            hidden_states, residual = self.layer_communicator.prepare_mlp( 
                hidden_states, residual, forward_batch
            )
            hidden_states_for_prefetch_rebalance.copy_(hidden_states)
            dispatch_done_event.record()
        ########################################################################

        ########################################################################
        # NOTE: Rebalancer + MoE_remote
        with torch.cuda.stream(rebalance_stream):
            rebalance_stream.wait_event(dispatch_start_event)
            # All-gather the routing results (and the predicted next-layer top-k ids) of all ranks
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
            topk_ids_cpu.copy_(topk_output.topk_ids, non_blocking=False)

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
                num_rebalanced_topk_ids_processed = num_remote_topk_ids_processed + num_local_topk_ids_processed

                num_total = topk_output.topk_ids.numel()

                #######################################
                # Compare the gathered top-k ids of this rank's tokens with the locally computed ones
                original_local_topk_ids = topk_output.topk_ids[self.local_dp_rank * local_hidden_states.shape[0]: (self.local_dp_rank+1)
                                                                                * local_hidden_states.shape[0]].detach().clone()
                original_local_topk_ids_sorted, _ = torch.sort(original_local_topk_ids, dim=-1)
                
                local_topk_ids_sorted, _ = torch.sort(local_topk_output.topk_ids, dim=-1)

                wrong_local_ids = find_rows_with_large_diff(original_local_topk_ids_sorted, local_topk_ids_sorted, threshold=0.1, min_count=1)
                ori_ids = local_topk_ids_sorted[wrong_local_ids]
                wrong_ids = original_local_topk_ids_sorted[wrong_local_ids]
                logger.info(f"wrong_local_ids({wrong_local_ids.shape}): {wrong_local_ids}")
                #######################################

            remote_hidden_states_ours = self.mlp(hidden_states_for_prefetch_rebalance, 
                                                    forward_batch, 
                                                    is_ours=True, 
                                                    topk_ids_mask=topk_ids_mask,
                                                    w1_buf=w13_weight_buf_even if self.layer_id % 2 == 0 else w13_weight_buf_odd,
                                                    w2_buf=w2_weight_buf_even if self.layer_id % 2 == 0 else w2_weight_buf_odd,
                                                    topk_output=topk_output,
                                                    is_remote=True
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
                # WARNING: H2D with sync
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
                logger.info(f"wrong_token_ids({wrong_token_ids.shape}): {wrong_token_ids}")
                logger.info(f"wrong_hidden_states({wrong_hidden_states.shape}): {wrong_hidden_states}")
                logger.info(f"true_hidden_states({true_hidden_states.shape}): {true_hidden_states}")
        ########################################################################

        ########################################################################
        # NOTE: Post-processing
        hidden_states_result, residual_result = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        ########################################################################


        torch.cuda.current_stream().wait_stream(prefetch_build_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w1_stream)
        torch.cuda.current_stream().wait_stream(prefetch_exec_w2_stream)

        nvtx.range_pop()
        nvtx.range_pop()


        return (
            hidden_states_result, 
            residual_result, 
            recv_experts_next_cur_rank_tensor, 
            recv_experts_next_cur_rank_tensor_cpu, 
            recv_num_experts_next,
            next_expert_gpu_mapping,
            next_remote_gpu_expert_load,
        )


# Replication plan computed in layer i for layer i + 1, handed to the next layer through a deque (q_len = 1).
@dataclass
class NextTwoPrefetchInfo:
    recv_experts_next_cur_rank_tensor: torch.Tensor | None
    recv_experts_next_cur_rank_tensor_cpu: torch.Tensor | None
    recv_num_experts_next: int | None
    next_expert_gpu_mapping: torch.Tensor | None
    next_remote_gpu_expert_load: torch.Tensor | None

class Glm4MoeModel(DeepseekV2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers = nn.ModuleList(
            [
                Glm4MoeDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dp_size = get_local_attention_dp_size()

        ###################################################
        # Common
        self.tp_size = get_local_attention_dp_size()
        self.tp_rank = get_local_attention_dp_rank()
        # CUDA Streams
        self.rebalance_stream = torch.cuda.Stream(priority=-3)
        self.prefetch_original_stream = torch.cuda.Stream(priority=-1)
        self.prefetch_build_stream = torch.cuda.Stream(priority=-6)
        self.prefetch_exec_w1_stream = torch.cuda.Stream(priority=-6)
        self.prefetch_exec_w2_stream = torch.cuda.Stream(priority=-1)
        self.dispatch_stream = torch.cuda.Stream(priority=-2)
        self.output_stream = torch.cuda.Stream(priority=-4)
        # Double-buffered SymmetricMemory expert weights for FusedMoE (resident experts + N replica slots).
        # Odd layers => odd buffers are used and even buffers are prefetched
        # Even layers => even buffers are used and odd buffers are prefetched
        # Shape template for the buffers: any MoE layer (layers 0-2 of GLM-4.5 are dense).
        temp_idx = 3

        self.w13_weight_buf_odd, self.w13_weight_hdl_odd = create_large_buffer_symm(
            self.layers[temp_idx].mlp.experts.w13_weight.data, 
            EXPERT_DUP_FACTOR_N
        )
        self.w2_weight_buf_odd, self.w2_weight_hdl_odd = create_large_buffer_symm(
            self.layers[temp_idx].mlp.experts.w2_weight.data, 
            EXPERT_DUP_FACTOR_N
        )
        self.w13_weight_buf_even, self.w13_weight_hdl_even = create_large_buffer_symm(
            self.layers[temp_idx].mlp.experts.w13_weight.data, 
            EXPERT_DUP_FACTOR_N
        )
        self.w2_weight_buf_even, self.w2_weight_hdl_even = create_large_buffer_symm(
            self.layers[temp_idx].mlp.experts.w2_weight.data, 
            EXPERT_DUP_FACTOR_N
        )
        # SymmetricMemory buffers for exchanging the per-rank routing results
        topk = 8
        # Per-rank token count (batch size * input length), set via $SEQ_LENS_SUM; must equal the actual per-rank token count.
        seq_lens_sum = int(os.getenv("SEQ_LENS_SUM", 4096))

        self.topk_ids_buf = symm_mem.empty((seq_lens_sum,  topk), dtype=torch.int32, device="cuda")
        self.topk_weights_buf = symm_mem.empty((seq_lens_sum, topk), dtype=torch.float32, device="cuda")
        self.next_topk_ids_buf = symm_mem.empty((seq_lens_sum,  topk), dtype=torch.int32, device="cuda")
        self.topk_ids_hdl = symm_mem.rendezvous(self.topk_ids_buf, dist.group.WORLD)
        self.topk_weights_hdl = symm_mem.rendezvous(self.topk_weights_buf, dist.group.WORLD)
        self.next_topk_ids_hdl = symm_mem.rendezvous(self.next_topk_ids_buf, dist.group.WORLD)

        # Buffer for prefetch and rebalance
        self.hidden_states_for_prefetch_rebalance = None
        ###################################################

    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        total_num_layers = len(self.layers)
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None

        normal_num_layers = (
            self.first_k_dense_replace
            if forward_batch.can_run_tbo
            else total_num_layers
        )

        ###################################################
        # For local/remote split
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device
        global_num_tokens = torch.sum(forward_batch.global_num_tokens_gpu)
        _, local_num_tokens = get_dp_local_info(forward_batch)
        remote_hidden_states = torch.empty(
            (global_num_tokens - local_num_tokens, hidden_states.size(1)),
            dtype=hidden_states_dtype,
            device=hidden_states_device,
        )
        # For the prefetcher
        topk = 8
        self.next_topk_ids_cpu = torch.empty(
            (forward_batch.seq_lens_sum * self.tp_size, topk),
            device=torch.device("cpu"),
            pin_memory=True,
            dtype=torch.int64,
        )
        self.send_map_local = torch.empty(
            (self.tp_size, self.tp_size, EXPERT_DUP_FACTOR_N),
            dtype=torch.int64,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        self.send_map_local_gpu = torch.empty(
            (self.tp_size, self.tp_size, EXPERT_DUP_FACTOR_N),
            dtype=torch.int64,
            device=hidden_states_device,
        )
        self.send_map_local_len = torch.empty(
            (self.tp_size, self.tp_size),
            dtype=torch.int64,
            device=torch.device("cpu"),
        )
        self.recv_map_local_len = torch.empty(
            (self.tp_size, self.tp_size),
            dtype=torch.int64,
            device=torch.device("cpu"),
        )
        self.hidden_states_for_prefetch_rebalance = torch.empty(
            (forward_batch.seq_lens_sum * self.tp_size, hidden_states.size(1)),
            dtype=hidden_states_dtype,
            device=hidden_states_device,
        )
        # For Rebalancer
        self.topk_ids_cpu = torch.empty(
            (forward_batch.seq_lens_sum * self.tp_size, topk),
            device=torch.device("cpu"),
            pin_memory=True,
            dtype=torch.int64,
        )
        self.topk_ids_mask = torch.empty(
            (forward_batch.seq_lens_sum * self.tp_size, topk),
            device=hidden_states_device,
            dtype=torch.int64,
        )
        self.topk_ids_mask_cpu = torch.empty(
            (forward_batch.seq_lens_sum * self.tp_size, topk),
            device=torch.device("cpu"),
            pin_memory=True,
            dtype=torch.int64,
        )
        # Next layer
        q_len = 1
        prefetch_q: deque[NextTwoPrefetchInfo] = deque(maxlen=q_len)
        recv_experts_next_cur_rank_tensor = None
        recv_experts_next_cur_rank_tensor_cpu = None 
        recv_num_experts_next = None
        next_expert_gpu_mapping = None
        next_remote_gpu_expert_load = None
        prefetch_plan = None
        ###################################################


        for i in range(normal_num_layers):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]

                ###################################################
                
                if len(prefetch_q) >= q_len:       # i > 0: plan returned by the previous layer
                    prefetch_info = prefetch_q.popleft()
                else:                          # i == 0: no plan yet
                    prefetch_info = NextTwoPrefetchInfo(*(None,)*5)

                
                next_gate = None
                next_w13_weight = None
                next_w2_weight = None
                target_layer_id = i + q_len
                if i >= self.first_k_dense_replace and target_layer_id < normal_num_layers:
                    next_gate = self.layers[target_layer_id].mlp.gate
                    next_w13_weight = self.layers[target_layer_id].mlp.experts.w13_weight.data
                    next_w2_weight = self.layers[target_layer_id].mlp.experts.w2_weight.data
                (
                    hidden_states, 
                    residual, 
                    recv_experts_next_cur_rank_tensor, 
                    recv_experts_next_cur_rank_tensor_cpu, 
                    recv_num_experts_next,
                    next_expert_gpu_mapping,
                    next_remote_gpu_expert_load,
                ) = layer(
                    positions=positions, 
                    hidden_states=hidden_states, 
                    forward_batch=forward_batch, 
                    residual=residual,
                    rebalance_stream=self.rebalance_stream, 
                    prefetch_original_stream=self.prefetch_original_stream, 
                    prefetch_build_stream=self.prefetch_build_stream, 
                    prefetch_exec_w1_stream=self.prefetch_exec_w1_stream, 
                    prefetch_exec_w2_stream=self.prefetch_exec_w2_stream, 
                    dispatch_stream=self.dispatch_stream, 
                    output_stream=self.output_stream, 
                    # Prefetcher
                    next_gate=next_gate,
                    next_w13_weight=next_w13_weight,
                    next_w2_weight=next_w2_weight,
                    # MoE computation buffers
                    w13_weight_buf_odd=self.w13_weight_buf_odd,
                    w2_weight_buf_odd=self.w2_weight_buf_odd,
                    w13_weight_buf_even=self.w13_weight_buf_even,
                    w2_weight_buf_even=self.w2_weight_buf_even,
                    # MoE computation handlers
                    w13_weight_hdl_odd=self.w13_weight_hdl_odd,
                    w2_weight_hdl_odd=self.w2_weight_hdl_odd,
                    w13_weight_hdl_even=self.w13_weight_hdl_even,
                    w2_weight_hdl_even=self.w2_weight_hdl_even,
                    next_topk_ids_cpu=self.next_topk_ids_cpu,
                    topk_ids_cpu=self.topk_ids_cpu,
                    topk_ids_mask=self.topk_ids_mask,
                    topk_ids_mask_cpu=self.topk_ids_mask_cpu,
                    send_map_local=self.send_map_local,
                    send_map_local_gpu=self.send_map_local_gpu,
                    send_map_local_len=self.send_map_local_len,
                    recv_map_local_len=self.recv_map_local_len,
                    recv_experts_cur_rank_tensor=prefetch_info.recv_experts_next_cur_rank_tensor, 
                    recv_experts_cur_rank_tensor_cpu=prefetch_info.recv_experts_next_cur_rank_tensor_cpu, 
                    recv_num_experts=prefetch_info.recv_num_experts_next,
                    expert_gpu_mapping=prefetch_info.next_expert_gpu_mapping,
                    remote_gpu_expert_load=prefetch_info.next_remote_gpu_expert_load,
                    topk_ids_buf=self.topk_ids_buf,
                    topk_weights_buf=self.topk_weights_buf,
                    next_topk_ids_buf=self.next_topk_ids_buf,
                    topk_ids_hdl=self.topk_ids_hdl,
                    topk_weights_hdl=self.topk_weights_hdl,
                    next_topk_ids_hdl=self.next_topk_ids_hdl,
                    hidden_states_for_prefetch_rebalance=self.hidden_states_for_prefetch_rebalance,
                    q_len=q_len,
                )
                prefetch_q.append(
                    NextTwoPrefetchInfo(
                        recv_experts_next_cur_rank_tensor,
                        recv_experts_next_cur_rank_tensor_cpu,
                        recv_num_experts_next,
                        next_expert_gpu_mapping,
                        next_remote_gpu_expert_load,
                    )
                )
        prefetch_q.clear()
                ###################################################
                
        if normal_num_layers != total_num_layers:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_num_layers:],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_num_layers - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Glm4MoeForCausalLM(DeepseekV2ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config.moe_layer_freq = 1
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts("Glm4MoeForCausalLM")
        self.model = Glm4MoeModel(
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
        self.dp_size = get_local_attention_dp_size()

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )

    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        if global_server_args_dict["disable_shared_experts_fusion"]:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if (
            not _is_cuda
            or torch.cuda.get_device_capability("cuda") < (8, 0)
            or self.config.architectures[0] != architecture
            or self.config.n_routed_experts != 128
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Only GLM-4.5 on NV-platform with capability >= 80 can use shared experts fusion optimization."
        elif (
            global_server_args_dict["enable_deepep_moe"]
            or global_server_args_dict["enable_ep_moe"]
        ):
            disable_reason = "Deepseek and GLM-4.5 can not use shared experts fusion optimization when in deepep_moe or ep_moe mode."

        if disable_reason is not None:
            global_server_args_dict["disable_shared_experts_fusion"] = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            weights_list = list(weights)
            weights_dict = dict(weights_list)
            if self.quant_config is not None:
                if self.quant_config.get_name() == "w8a8_int8":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                    ]
                elif (
                    self.quant_config.get_name() == "fp8"
                    or self.quant_config.get_name() == "blockwise_int8"
                    or self.quant_config.get_name() == "compressed_tensors"
                ):
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                    ]
                elif self.quant_config.get_name() == "awq":
                    suffix_list = [
                        "down_proj.qweight",
                        "down_proj.qzeros",
                        "down_proj.scales",
                        "gate_proj.qweight",
                        "gate_proj.qzeros",
                        "gate_proj.scales",
                        "up_proj.qweight",
                        "up_proj.qzeros",
                        "up_proj.scales",
                    ]
                elif self.quant_config.get_name() == "modelopt_fp4":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "down_proj.weight_scale_2",
                        "down_proj.input_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "gate_proj.weight_scale_2",
                        "gate_proj.input_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                        "up_proj.weight_scale_2",
                        "up_proj.input_scale",
                    ]
                else:
                    raise ValueError(
                        f"Unsupported shared expert fusion for quantization: {self.quant_config.get_name()}."
                    )
            else:
                suffix_list = [
                    "down_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                ]
            names_to_remove = []

            moe_layers = (
                range(
                    self.config.first_k_dense_replace,
                    self.config.num_hidden_layers,
                    self.config.moe_layer_freq,
                )
                if not is_nextn
                else [nextn_layer_id]
            )

            for moe_layer in moe_layers:
                for suffix in suffix_list:
                    shared_expert_weight_name = (
                        f"model.layers.{moe_layer}.mlp.shared_experts.{suffix}"
                    )
                    # online fp8 quantization does not load weight_scale
                    if shared_expert_weight_name not in weights_dict:
                        continue
                    weights_list.append(
                        (
                            f"model.layers.{moe_layer}."
                            f"mlp.experts."
                            f"{self.config.n_routed_experts + 0}"
                            f".{suffix}",
                            weights_dict[shared_expert_weight_name],
                        )
                    )
                    names_to_remove += [shared_expert_weight_name]
            weights = [w for w in weights_list if w[0] not in names_to_remove]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = get_moe_impl_class().make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        params_dict = dict(self.named_parameters())
        weight_names = []
        for name, loaded_weight in weights:
            weight_names.append(name)

            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

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
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
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
                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            fused_weight = torch.cat(
                                [q_a_proj_weight, kv_a_proj_weight], dim=0
                            )
                            param_name = (
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                                )
                            )
                            param = params_dict[param_name]

                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        if (
                            "k_scale" in name or "v_scale" in name
                        ) and name not in params_dict:
                            # modelopt attn kv scale is named differently
                            if any(scale in name for scale in ["k_scale", "v_scale"]):
                                name = name.replace("_proj", "attn_mqa")
                            else:
                                logger.warning(
                                    f"Unknown scale found in checkpoint: {name}"
                                )
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)


EntryClass = [Glm4MoeForCausalLM]

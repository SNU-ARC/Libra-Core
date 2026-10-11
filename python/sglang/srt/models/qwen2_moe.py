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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

import os
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.ep_moe.layer import EPMoE, get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import add_prefix, make_layers


from sglang.srt.layers.dp_attention import (
    get_dp_local_info,
    get_local_attention_dp_rank
)
from sglang.srt.utils import (
    EXPERT_DUP_FACTOR_N,
    create_large_buffer_symm
)
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from collections import deque


from sglang.lina.lina_prefetch_route_v2 import Lina


import os

logger = logging.getLogger(__name__)


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
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
        )

        self.experts = get_moe_impl_class()(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
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
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
            )
        else:
            self.shared_expert = None
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
                )

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
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
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
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
        qkv_bias = getattr(config, "qkv_bias", True)
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            qkv_bias=qkv_bias,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.local_dp_size = get_local_attention_dp_size()

        # Qwen2MoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen2MoeMLP(
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual

# Replication plan computed in layer i for layer i + 1 (handed over by Qwen2MoeModel.forward)
@dataclass
class NextTwoPrefetchInfo:
    recv_experts_next_cur_rank_tensor: torch.Tensor | None
    recv_experts_next_cur_rank_tensor_cpu: torch.Tensor | None
    recv_num_experts_next: int | None
    next_expert_gpu_mapping: torch.Tensor | None
    next_remote_gpu_expert_load: torch.Tensor | None


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not global_server_args_dict["enable_dp_attention"],
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2MoeDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []
        
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
        # Large expert-weight buffers for FusedMoE: resident experts + EXPERT_DUP_FACTOR_N replica slots
        # Odd layers => odd buffers are used and even buffers are prefetched
        # Even layers => even buffers are used and odd buffers are prefetched
        # Layer whose expert weights serve as the shape template for the buffers
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
        # Symmetric buffers for the per-rank routing results (Libra plumbing; passed through but unused by the Lina path)
        topk = 8
        # NOTE: must equal the number of tokens per rank (SEQ_LENS_SUM, exported by the launcher)
        seq_lens_sum = int(os.getenv("SEQ_LENS_SUM", 4096))

        self.topk_ids_buf = symm_mem.empty((seq_lens_sum,  topk), dtype=torch.int32, device="cuda")
        self.topk_weights_buf = symm_mem.empty((seq_lens_sum, topk), dtype=torch.float32, device="cuda")
        self.next_topk_ids_buf = symm_mem.empty((seq_lens_sum,  topk), dtype=torch.int32, device="cuda")
        self.topk_ids_hdl = symm_mem.rendezvous(self.topk_ids_buf, dist.group.WORLD)
        self.topk_weights_hdl = symm_mem.rendezvous(self.topk_weights_buf, dist.group.WORLD)
        self.next_topk_ids_hdl = symm_mem.rendezvous(self.next_topk_ids_buf, dist.group.WORLD)

        # Buffer for prefetch and rebalance
        self.hidden_states_for_prefetch_rebalance = None

        
        self.lina_L = 3
        self.lina = Lina(
            num_layers=config.num_hidden_layers,
            L=self.lina_L,
            num_gpus=self.tp_size,
            topk=topk,
            max_dup_per_gpu=EXPERT_DUP_FACTOR_N,
            device=self.w13_weight_buf_odd.device,
            verbose=True,
        )
        ept_dir = os.getenv('LINA_EPT_DIR')
        assert ept_dir is not None, 'Env var LINA_EPT_DIR must be set'
        self.lina.load_expert_path_tables(
            expert_path_table_dir=ept_dir,
            cache_dir=os.getenv('LINA_CACHE_DIR'),
        )
        self.num_correctness_tests = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        self.lina.new_batch(hidden_states.shape[0] * self.tp_size)

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            
            # For local/remote split
            hidden_states_dtype = hidden_states.dtype
            hidden_states_device = hidden_states.device
            global_num_tokens = torch.sum(forward_batch.global_num_tokens_gpu)
            _, local_num_tokens = get_dp_local_info(forward_batch)
            # For Prefetcher
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

            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    layer = self.layers[i]

                    
                    if len(prefetch_q) >= q_len:  # plan produced by the previous layer
                        prefetch_info = prefetch_q.popleft()
                    else:  # first layer: no plan yet
                        prefetch_info = NextTwoPrefetchInfo(*(None,)*5)

                    
                    next_gate = None
                    next_w13_weight = None
                    next_w2_weight = None
                    target_layer_id = i + q_len
                    if target_layer_id < self.end_layer:
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
                        lina=self.lina,
                        is_correctness=(self.num_correctness_tests > 0),
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

        self.num_correctness_tests -= 1

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Qwen2MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2MoeModel(
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
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE

        expert_params_mapping = MoEImpl.make_expert_params_mapping(
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

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = Qwen2MoeForCausalLM

# -*- encoding: utf-8 -*-
'''
@File    :   modeling_llama_moe_vllm.py
@Time    :   2024/06/17 
@Author  :   gaoqiang 
@Version :   1.0
@Contact :   gaoqiang_mx@163.com
@Desc    :   integrate custom modeling file to vllm 
'''

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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


"""Inference-only Mixtral model."""
from typing import Iterable, List, Optional, Tuple,Dict,Any

import torch
from torch import nn
import pdb
import sys

sys.path.append("/apdcephfs_qy3/share_301372554/share_info/qianggao/modeling_file/llama3_moe/")
from configuration_llama_moe import LlamaConfig

from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
# from vllm.model_executor.layers.quantization.fp8 import (
                                                        # Fp8Config,
                                                        #  per_tensor_dequantize,
                                                        #  per_tensor_quantize)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader,kv_cache_scales_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput
from vllm.utils import print_warning_once


from vllm.utils import is_hip, print_warning_once

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
    
class LlamaSparseMoeBlock_(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}.")

        self.experts = nn.ModuleList([
            LlamaMLP(hidden_size=config.hidden_size,
                        intermediate_size=config.intermediate_size,
                        hidden_act=config.hidden_act,
                        linear_method=linear_method,
                        )
            for idx in range(self.n_routed_experts)
        ])
        self.pack_params()

        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.n_routed_experts,
                                     bias=False,
                                     linear_method=None)

    def pack_params(self):
        w1 = []
        w2 = []
        for expert in self.experts:
            w1.append(expert.gate_up_proj.weight)
            w2.append(expert.down_proj.weight)
        self.w1 = torch._utils._flatten_dense_tensors(w1)
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)
        for data, param in zip(w1s, w1):
            param.data = data
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)

        self.w2 = torch._utils._flatten_dense_tensors(w2)
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)
        for data, param in zip(w2s, w2):
            param.data = data

        self.w2 = self.w2.view(len(w2), *w2s[0].shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
      
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = fused_moe(hidden_states,
                                        self.w1,
                                        self.w2,
                                        router_logits,
                                        self.top_k,
                                        renormalize=False,
                                        inplace=True)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

class LlamaSparseMoeBlock(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        tp_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.quant_config = quant_config

        # FIXME(pcmoritz): Make this more general to support different
        # quantization schemes
        # self.use_fp8 = isinstance(quant_config, Fp8Config)
        self.use_fp8 = False

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(self.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     params_dtype=self.params_dtype,
                                    )

        if self.use_fp8 and self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        self.w13_weight = nn.Parameter(torch.empty(self.num_total_experts,
                                                   2 * self.intermediate_size,
                                                   self.hidden_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        self.w2_weight = nn.Parameter(torch.empty(self.num_total_experts,
                                                  self.hidden_size,
                                                  self.intermediate_size,
                                                  dtype=params_dtype),
                                      requires_grad=False)

        set_weight_attrs(self.w13_weight, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w2_weight, {
            "weight_loader": self.weight_loader,
        })

        # Used for fp8.
        self.w13_scale = None
        self.w2_scale = None
        self.a13_scale = None
        self.a2_scale = None

        if self.use_fp8:
            # WEIGHT_SCALE (for fp8)
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            self.w13_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                     2,
                                                     dtype=torch.float32),
                                          requires_grad=False)
            self.w2_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                    dtype=torch.float32),
                                         requires_grad=False)

            # If loading fp8 checkpoint, pass the weight loaders.
            # If loading an fp16 checkpoint, do not (we will quantize in
            #   process_weights_after_loading()
            if quant_config.is_checkpoint_fp8_serialized:
                set_weight_attrs(self.w13_scale, {
                    "weight_loader": self.weight_loader,
                })
                set_weight_attrs(self.w2_scale, {
                    "weight_loader": self.weight_loader,
                })

            # INPUT_SCALE (for fp8)
            if quant_config.activation_scheme == "static":
                if not quant_config.is_checkpoint_fp8_serialized:
                    raise ValueError(
                        "Found static activation scheme for checkpoint that "
                        "was not serialized fp8.")
                self.a13_scale = nn.Parameter(torch.ones(
                    self.num_total_experts, dtype=torch.float32),
                                              requires_grad=False)
                self.a2_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                        dtype=torch.float32),
                                             requires_grad=False)

                set_weight_attrs(self.a13_scale, {
                    "weight_loader": self.weight_loader,
                })
                set_weight_attrs(self.a2_scale, {
                    "weight_loader": self.weight_loader,
                })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("gate_proj.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("up_proj.weight"):
            param_data[expert_id,
                       shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("down_proj.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

        # Loading scales
        if "input_scale" in weight_name or "down_proj.weight_scale" in weight_name:
            if param_data[expert_id] != 1 and (param_data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}")
            param_data[expert_id] = loaded_weight
        elif "weight_scale" in weight_name:
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            assert "gate_proj" in weight_name or "up_proj" in weight_name
            shard_id = 0 if "gate_proj" in weight_name else 1
            param_data[expert_id][shard_id] = loaded_weight

    def process_weights_after_loading(self):
        # Fp8 is the only case where we need to process after loading.
        if not self.use_fp8:
            return

        # If checkpoint is fp16, quantize here.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            w13_weight = torch.empty_like(self.w13_weight.data,
                                          dtype=torch.float8_e4m3fn)
            w2_weight = torch.empty_like(self.w2_weight.data,
                                         dtype=torch.float8_e4m3fn)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            self.w13_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                     dtype=torch.float32),
                                          requires_grad=False)
            for expert in range(self.num_total_experts):
                w13_weight[expert, :, :], self.w13_scale[
                    expert] = ops.scaled_fp8_quant(
                        self.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], self.w2_scale[
                    expert] = ops.scaled_fp8_quant(
                        self.w2_weight.data[expert, :, :])
            self.w13_weight = nn.Parameter(w13_weight, requires_grad=False)
            self.w2_weight = nn.Parameter(w2_weight, requires_grad=False)

        else:
            # If checkpoint is fp8 + static, cleanup input_scales.
            #   Since state_dict has an input_scale per expert but our kernels
            #   are passed one input_scale shared across all experts.
            if self.quant_config.activation_scheme == "static":
                if self.a13_scale is None or self.a2_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None.")

                if (not all_close_1d(self.a13_scale)
                        or not all_close_1d(self.a2_scale)):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. ")

                self.a13_scale = nn.Parameter(self.a13_scale.max(),
                                              requires_grad=False)
                self.a2_scale = nn.Parameter(self.a2_scale.max(),
                                             requires_grad=False)

            assert self.w13_scale is not None
            shard_size = self.intermediate_size
            max_w13_scales = self.w13_scale.max(dim=1).values
            for expert_id in range(self.num_total_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        self.w13_weight[expert_id][start:start +
                                                   shard_size, :],
                        self.w13_scale[expert_id][shard_id])
                    self.w13_weight[expert_id][
                        start:start + shard_size, :] = per_tensor_quantize(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            self.w13_scale = nn.Parameter(max_w13_scales, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = fused_moe(hidden_states,
                                        self.w13_weight,
                                        self.w2_weight,
                                        router_logits,
                                        self.top_k,
                                        renormalize=True,
                                        inplace=True,
                                        )
                                        
                                       

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))

class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            linear_method=linear_method
            
           
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            linear_method=linear_method
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            linear_method=linear_method
        )
        self.block_sparse_moe = LlamaSparseMoeBlock_(
            config,
            linear_method=linear_method
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config=config,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              linear_method=linear_method)

            for idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    


class  LlamaMoEForCausalLM(nn.Module):
    
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.linear_method = linear_method

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                linear_method=linear_method
                                )
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

        

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # pdb.set_trace()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # #copy from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py and modify
        # expert_params_mapping = [
        #     # These are the weight scales for the experts
        #     # (param_name, weight_name, expert_id)
        #     ("w13_scale" if weight_name in ["gate_proj", "up_proj"] else "w2_scale",
        #      f"experts.{expert_id}.{weight_name}.weight_scale", expert_id)
        #     for expert_id in range(self.config.num_local_experts)
        #     for weight_name in ["gate_proj", "down_proj", "up_proj"]
        # ] + [
        #     # These are the weights for the experts
        #     # (param_name, weight_name, expert_id)
        #     ("w13_weight" if weight_name in ["gate_proj", "up_proj"] else "w2_weight",
        #      f"experts.{expert_id}.{weight_name}.weight", expert_id)
        #     for expert_id in range(self.config.num_local_experts)
        #     for weight_name in ["gate_proj", "down_proj", "up_proj"]
        # ] + [
        #     # These are the activation scales for the experts
        #     # (param_name, weight_name, expert_id)
        #     ("a13_scale" if weight_name in ["gate_proj", "up_proj"] else "a2_scale",
        #      f"experts.{expert_id}.{weight_name}.input_scale", expert_id)
        #     for expert_id in range(self.config.num_local_experts)
        #     for weight_name in ["gate_proj", "down_proj", "up_proj"]
        # ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            # else:
            #     # add for MoE
            #     for param_name, weight_name, expert_id in expert_params_mapping:
            #         if weight_name not in name:
            #             continue
            #         name = name.replace(weight_name, param_name)
            #         param = params_dict[name]
            #         weight_loader = param.weight_loader
            #         weight_loader(param,
            #                       loaded_weight,
            #                       weight_name,
            #                       expert_id=expert_id)
            #         break
            else:

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
               
               # Skip experts that are not assigned to this worker.
                if (("mlp.experts." in name or "mlp.shared_expert." in name)
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")
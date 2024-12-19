import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate
from einops import rearrange
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input

if os.environ.get("TRAIN_MODE", "False") == "True":
    from megatron.core import mpu
    from deepspeed.sequence import SeqAllToAll4D

from .cudnn_attention import cudnn_attn_check_capability, cudnn_attn_func
from .downsampling import TokenMerge
from .normalization import RMSNorm
from .upsampling import TokenSplit


class AttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        use_flash_attn: bool = False,
        use_cudnn_attn: bool = False,
        qk_norm: bool = False,
        embed_dim: int = 72,
        eps: float = 1e-6,
        token_merge_size: Optional[int] = None,
        inner_dim: int = 1152,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()

        if token_merge_size is not None:
            self.token_merge = TokenMerge(in_features=inner_dim, out_features=inner_dim, token_merge_size=token_merge_size)
            self.token_split = TokenSplit(in_features=inner_dim, out_features=inner_dim, token_merge_size=token_merge_size)
        else:
            self.token_merge = None
            self.token_split = None

        if qk_norm:
            self.q_norm = RMSNorm(embed_dim, eps=eps)
            self.k_norm = RMSNorm(embed_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.use_flash_attn = use_flash_attn
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        if torch.cuda.is_available() and torch.version.hip:
            self.flash_attn_max_head_dim = 128
        elif torch.cuda.is_available() and torch.version.cuda:
            self.flash_attn_max_head_dim = 256
        else:
            self.flash_attn_max_head_dim = None
        self.use_cudnn_attn = use_cudnn_attn

    def _attn_varlen(self, query, key, value, crossattn_mask_kwargs=None, selfattn_mask_kwargs=None):
        assert crossattn_mask_kwargs != None or selfattn_mask_kwargs != None, "crossattn_mask_kwargs 和 selfattn_mask_kwargs不可同时为None"

        batch, seqlen = query.shape[:2]

        # for q
        if selfattn_mask_kwargs is not None:
            max_seqlen_in_batch_q = selfattn_mask_kwargs["max_seqlen_in_batch"]
            cu_seqlens_q = selfattn_mask_kwargs["cu_seqlens"]
            indices_q = selfattn_mask_kwargs["indices"]
            query = index_first_axis(rearrange(query, "b s ... -> (b s) ..."), indices_q)
        else:
            max_seqlen_in_batch_q = query.shape[1]
            cu_seqlens_q = torch.arange(0, query.shape[0] * query.shape[1] + 1, query.shape[1], dtype=torch.int32, device=query.device)
            indices_q = torch.arange(0, query.shape[0] * query.shape[1], device=query.device)
            query = rearrange(query, "b s ... -> (b s) ...")

        # for k & v
        if crossattn_mask_kwargs is not None:
            cu_seqlens_kv = crossattn_mask_kwargs["cu_seqlens"]
            max_seqlen_in_batch_kv = crossattn_mask_kwargs["max_seqlen_in_batch"]
            indices_kv = crossattn_mask_kwargs["indices"]
        else:
            cu_seqlens_kv = selfattn_mask_kwargs["cu_seqlens"]
            max_seqlen_in_batch_kv = selfattn_mask_kwargs["max_seqlen_in_batch"]
            indices_kv = selfattn_mask_kwargs["indices"]

        # TODO: index_first_axis is not efficient.
        key = index_first_axis(rearrange(key, "b s ... -> (b s) ..."), indices_kv)
        value = index_first_axis(rearrange(value, "b s ... -> (b s) ..."), indices_kv)
        hidden_states = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_kv,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )

        hidden_states = pad_input(hidden_states, indices_q, batch, seqlen)
        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        patch_resolution: Optional[Tuple[int, int, int]] = None,
        selfattn_mask_kwargs: Optional[dict] = None,
        crossattn_mask_kwargs: Optional[dict] = None,
        rope: Optional[nn.Module] = None,
        is_video_batch: bool = False,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if self.token_merge is not None:
            hidden_states, patch_resolution = self.token_merge(hidden_states, patch_resolution)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if rope is not None:
            query = rope.forward(query, patch_resolution)
            key = rope.forward(key, patch_resolution)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.use_flash_attn and query.dtype is not torch.float32 and query.shape[-1] <= self.flash_attn_max_head_dim:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if selfattn_mask_kwargs is None and crossattn_mask_kwargs is None:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()
                if self.use_cudnn_attn and cudnn_attn_check_capability(query, key, value, causal=False):
                    hidden_states = cudnn_attn_func(query, key, value, causal=False)
                else:
                    hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
            else:
                hidden_states = self._attn_varlen(query, key, value, crossattn_mask_kwargs=crossattn_mask_kwargs, selfattn_mask_kwargs=selfattn_mask_kwargs)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        del query, key, value

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if self.token_split is not None:
            hidden_states, patch_resolution = self.token_split(hidden_states, patch_resolution)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


if __name__ == "__main__":
    import argparse
    import os.path as osp

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=77)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--width", type=int, default=26)
    parser.add_argument("--in_channel", type=int, default=2880)
    parser.add_argument("--out_channel", type=int, default=2880)
    parser.add_argument("--num_attention_heads", type=int, default=40)
    parser.add_argument("--attention_head_dim", type=int, default=72)
    parser.add_argument("--cross_attention_tokens", type=int, default=256)
    parser.add_argument("--cross_attention_dim", type=int, default=4096)
    parser.add_argument("--output_root", type=str, default=None)
    
    args = parser.parse_args()
    # 定义输入参数
    batch_size = 2 if args.batch_size is None else int(args.batch_size)
    num_frames = 20 if args.num_frames is None else int(args.num_frames)
    sequence_length = args.height * args.width
    in_channel = 2880 if args.in_channel is None else int(args.in_channel)
    out_channel = 2880 if args.out_channel is None else int(args.out_channel)
    num_attention_heads = 40 if args.num_attention_heads is None else int(args.num_attention_heads)
    attention_head_dim = 72 if args.attention_head_dim is None else int(args.attention_head_dim)
    dropout = 0.1
    cross_attention_tokens = args.cross_attention_tokens
    cross_attention_dim = args.cross_attention_dim

    model = AttnProcessor2_0(
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        dropout=dropout,
        cross_attention_dim=cross_attention_dim,
        cross_attention_tokens=cross_attention_tokens,
    )
    model.to(device="cuda", dtype=torch.bfloat16)

    x = torch.randn(batch_size, num_frames*sequence_length, in_channel).to(device="cuda", dtype=torch.bfloat16)
    y = model(x)
    print(y.shape)
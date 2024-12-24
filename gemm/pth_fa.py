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
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.use_flash_attn = use_flash_attn
        self.use_cudnn_attn = use_cudnn_attn
        if torch.cuda.is_available() and torch.version.hip:
            self.flash_attn_max_head_dim = 128
        elif torch.cuda.is_available() and torch.version.cuda:
            self.flash_attn_max_head_dim = 256
        else:
            self.flash_attn_max_head_dim = None

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
        hidden_states: torch.FloatTensor,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        selfattn_mask_kwargs: Optional[dict] = None,
        crossattn_mask_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        batch_size, num_heads, sequence_length, head_dim = query.shape
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
                if self.use_cudnn_attn:
                    pass
                else:
                    hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
            else:
                hidden_states = self._attn_varlen(query, key, value, crossattn_mask_kwargs=crossattn_mask_kwargs, selfattn_mask_kwargs=selfattn_mask_kwargs)
        else:
            if attention_mask is not None:
                attention_mask = Attention.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, num_heads, -1, attention_mask.shape[-1])

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        return hidden_states
 

if __name__ == "__main__":
    import argparse
    import os.path as osp

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seqlen_q", type=int, default=4680)
    parser.add_argument("--seqlen_kv", type=int, default=4680)
    parser.add_argument("--num_head", type=int, default=40)
    parser.add_argument("--head_dim", type=int, default=72)
    parser.add_argument("--cross_attention_tokens", type=int, default=256)
    parser.add_argument("--use_flash_attn", action="store_true", help="Enable flash attention")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--run_iter", type=int, default=10)
    parser.add_argument("--warmup_iter", type=int, default=10)
    parser.add_argument("--hw_tflops", type=int, default=165)
    parser.add_argument("--note", type=str, default=None)
    args = parser.parse_args()
    if args.use_flash_attn:
        print("***** use flash attn")
    else:
        print("***** use pytorch scale dot product attention")
        print(f"Flash Attention Enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Math SDP Enabled: {torch.backends.cuda.math_sdp_enabled()}")
        print(f"Mem Efficient SDP Enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    # 定义输入参数
    batch_size = 2 if args.batch_size is None else int(args.batch_size)
    seqlen_q = 4680 if args.seqlen_q is None else int(args.seqlen_q)
    seqlen_kv = 4680 if args.seqlen_kv is None else int(args.seqlen_kv)
    num_head = 40 if args.num_head is None else int(args.num_head)
    head_dim = 72 if args.head_dim is None else int(args.head_dim)
    cross_attention_tokens = 256 if args.cross_attention_tokens is None else int(args.cross_attention_tokens)
    output_root = args.output_root

    model = AttnProcessor2_0(
        use_flash_attn=args.use_flash_attn,
    )
    model.to(device="cuda", dtype=torch.bfloat16)

    # 随机生成输入张量
    attention_mask = torch.randint(0, 2, (batch_size, num_head, seqlen_q, seqlen_kv), dtype=torch.bfloat16, device="cuda", requires_grad=False)
    q = torch.randn(batch_size, num_head, seqlen_q, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    k = torch.randn(batch_size, num_head, seqlen_kv, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    v = torch.randn(batch_size, num_head, seqlen_kv, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    hidden_states = torch.randn(batch_size, num_head, seqlen_q, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    output = torch.randn(batch_size, num_head, seqlen_q, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=False)
    

    for i in range(args.run_iter):
        output = model(hidden_states,q, k, v)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Forward pass
    start_event.record()
    for i in range(args.run_iter):
        output = model(hidden_states,q, k, v)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / args.run_iter
    GFLOPs_theory = (2*2*batch_size*num_head*head_dim*seqlen_q*seqlen_kv)/1000/1000/1000
    GFLOPS_real = GFLOPs_theory/avg_time_ms
    mfu = GFLOPS_real/args.hw_tflops*100
    # mbu = batch_size*num_frames*sequence_length*in_channel + batch_size*num_frames*sequence_length*in_channel*out_channel
    print(f"***{args.note}******* \n"
        f"{q.shape}*{k.shape}\t time={avg_time_ms}ms, \t GFLOPs theory={GFLOPs_theory:.4f} \t GFLOPS real={GFLOPS_real:.4f} \t mfu={mfu}%")

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, token_merge_size=2):
        super().__init__()
        self.token_merge_size = token_merge_size
        self.proj = nn.Linear(in_features * token_merge_size * token_merge_size, out_features, bias=False)

    def _forward(self, x):
        x = rearrange(x, "... f (h nh) (w nw) e -> ... f h w (nh nw e)", nh=self.token_merge_size, nw=self.token_merge_size)
        x = self.proj(x)
        return x

    def forward(self, hidden_states, patch_resolution):
        if isinstance(patch_resolution, list):
            if not all(isinstance(res, tuple) for res in patch_resolution):
                raise ValueError(f"patch_resolution should be a list of tuples, got {patch_resolution}")

            batch_size, sequence_length, dim = hidden_states.shape
            token_merge_sequence_length = sequence_length // (self.token_merge_size * self.token_merge_size)
            output = torch.zeros(batch_size, token_merge_sequence_length, dim, device=hidden_states.device, dtype=hidden_states.dtype)
            for i, resolution in enumerate(patch_resolution):
                valid_sequence_length = math.prod(resolution)
                valid_hidden_states = torch.narrow(hidden_states[i], dim=0, start=0, length=valid_sequence_length)
                valid_hidden_states = rearrange(valid_hidden_states, "(f h w) d -> f h w d", f=resolution[0], h=resolution[1], w=resolution[2])
                valid_hidden_states = self._forward(valid_hidden_states)
                valid_hidden_states = rearrange(valid_hidden_states, "f h w d -> (f h w) d")
                token_merge_valid_sequence_length = valid_sequence_length // (self.token_merge_size * self.token_merge_size)
                output[i,:token_merge_valid_sequence_length] = valid_hidden_states
                patch_resolution[i] = (resolution[0], resolution[1] // self.token_merge_size, resolution[2] // self.token_merge_size)

            return output, patch_resolution
        else:
            if not isinstance(patch_resolution, tuple):
                raise ValueError(f"patch_resolution should be a tuple, got {patch_resolution}")

            hidden_states = rearrange(hidden_states, "b (f h w) d -> b f h w d", f=patch_resolution[0], h=patch_resolution[1], w=patch_resolution[2])
            hidden_states = self._forward(hidden_states)
            pf, ph, pw = hidden_states.shape[1:-1]
            patch_resolution = (pf, ph, pw)
            # assert patch_resolution == (patch_resolution[0], patch_resolution[1] // self.token_merge, patch_resolution[2] // self.token_merge)
            hidden_states = rearrange(hidden_states, "b f h w d -> b (f h w) d")

            return hidden_states, patch_resolution


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, mult=4.0, dropout=0.0, bias=False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        self.linear1 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear2 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear3 = nn.Linear(inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear3.weight, mode='fan_out', nonlinearity='relu')
        
    @torch.compile
    def silu_multiply(self, a, b):
        return F.silu(a) * b

    def forward(self, hidden_states):
        hidden_states_1 = self.linear1(hidden_states)
        hidden_states_2 = self.linear2(hidden_states)
        hidden_states = self.silu_multiply(hidden_states_1, hidden_states_2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear3(hidden_states)
        return hidden_states
    
class GemmForward(nn.Module):
    def __init__(self, dim, inner_dim, mult=4.0, dropout=0.0, bias=False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        self.linear1 = nn.Linear(dim, inner_dim, bias=bias)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, hidden_states):
        hidden_states_1 = self.linear1(hidden_states)
        return hidden_states_1


if __name__ == "__main__":
    import argparse
    import os.path as osp

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=2880)
    parser.add_argument("--in_channel", type=int, default=2880)
    parser.add_argument("--out_channel", type=int, default=2880)
    parser.add_argument("--num_attention_heads", type=int, default=40)
    parser.add_argument("--attention_head_dim", type=int, default=72)
    parser.add_argument("--cross_attention_tokens", type=int, default=256)
    parser.add_argument("--cross_attention_dim", type=int, default=4096)
    parser.add_argument("--run_iter", type=int, default=10)
    parser.add_argument("--warmup_iter", type=int, default=10)
    parser.add_argument("--hw_tflops", type=int, default=232)
    parser.add_argument("--ops", type=str, default="gemm", help=" can choose gemm、 ffn")
    parser.add_argument("--note", type=str, default=None)
    args = parser.parse_args()
    # print(args)
    # 定义输入参数
    batch_size = 2 if args.batch_size is None else int(args.batch_size)
    sequence_length = 936 if args.sequence_length is None else int(args.sequence_length)
    in_channel = 2880 if args.in_channel is None else int(args.in_channel)
    out_channel = 2880 if args.out_channel is None else int(args.out_channel)
    num_attention_heads = 40 if args.num_attention_heads is None else int(args.num_attention_heads)
    attention_head_dim = 72 if args.attention_head_dim is None else int(args.attention_head_dim)
    dropout = 0.1
    cross_attention_tokens = args.cross_attention_tokens
    cross_attention_dim = args.cross_attention_dim


    if args.ops == "gemm":
        model = GemmForward(in_channel, out_channel,bias=True)
    else:
        model = SwiGLUFeedForward(in_channel, out_channel,bias=True)

    
    model.to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(batch_size, sequence_length, in_channel).to(device="cuda", dtype=torch.bfloat16)
    for i in range(args.warmup_iter):
        y = model(x)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(args.run_iter):
        y = model(x)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / args.run_iter
    gflops = 2*batch_size*sequence_length*in_channel*out_channel/1000/1000/1000
    mfu = gflops/avg_time_ms/args.hw_tflops*100
    # mbu = batch_size*num_frames*sequence_length*in_channel + batch_size*num_frames*sequence_length*in_channel*out_channel
    print(f"***{args.note}****{args.ops}****** \n"
        f"{x.shape}*{model.linear1.weight.shape}\t time={avg_time_ms}ms, \t gflops={gflops} \t mfu={mfu}%")


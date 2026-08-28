import math

import torch

from torchfeather.model_args import DeepseekV3ModelArgs

def precompute_freq_cis(args: DeepseekV3ModelArgs) -> torch.Tensor:
    dim=args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_pair_index_that_rotate_N_times(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
        return (dim * math.log(max_seq_len / 2 * math.pi * num_rotations)) / (2 * math.log(base))

    # base RoPE frequencies, we attribute a frequency to each pair of the dimension
    # DIMENSION: [d/2]
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )

    # position indices
    # DIMENSION: [seqlen]
    t = torch.arange(seqlen)

    # Outer product which is going to give back an object of dimension [positions] x [frequencies]
    # It is basically computing all combinations of positions x frequencies
    # DIMENSION : [seqlen, d/2]
    freqs = torch.outer(t, freqs)

    # it is going to compute a tensor of complex numbers e^(ixposxfreq)
    # DIMENSION: [seqlen, d/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x:torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
    # x.shape = [B, S, H, D]
    dtype = x.dtype

    x = x.float() #converts the values to float32
    x = x.view(*x.shape[:-1], -1, 2) # [B, S, H, D/2, 2]
    x = torch.view_as_complex(x) # [B, S, H, D/2]

    freq_cis = freq_cis.view(1, x.size(1), 1, x.size(-1)) # [B, S, H, D/2]

    y = torch.view_as_real(x*freq_cis).flatten(3) # [B, S, H, D] because view are real gives back [B, S, H, D/2, 2] and then we flatten last two dimensions together
    return y.to(dtype)

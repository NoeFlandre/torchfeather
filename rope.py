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
        """
        This function is returning the pair index whose RoPE frequency does rotate N times.
        This formula just come from isolating i in the formula: N = Lwi / 2pi
        """
        return (dim * math.log(max_seq_len / 2 * math.pi * num_rotations)) / (2 * math.log(base))


    def find_correction_range(low_rot:float, high_rot:float, dim:int, base:float, max_seq_len:int) -> tuple[int, int]:
        """
        Given a number of low rotations and high rotations, returns the tuple of pair index between this low and high rotations number.
        """

        low = math.floor(find_pair_index_that_rotate_N_times(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_pair_index_that_rotate_N_times(high_rot, dim, base, max_seq_len))

        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min:float, max:float, dim:int) -> torch.Tensor:
        """
        Returns the linear ramp factor for the dimension pairs between min and max
        """

        if min == max:
            max += 0.001

        linear_ramp_factor = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        linear_ramp_factor = torch.clamp(linear_ramp_factor, 0, 1)

        return linear_ramp_factor

    # base RoPE frequencies, we attribute a frequency to each pair of the dimension
    # DIMENSION: [d/2]
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )

    if seq_len > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)

        smooth = 1 - linear_ramp_factor(low, high, dim //2)
        freqs = (freqs / factor)*(1 - smooth) + freqs*smooth

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

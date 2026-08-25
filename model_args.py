from dataclasses import dataclass # we will be using dataclass to create a structured container for configuration values
from dataclasses import field # this is useful if we want to give special rules to certain attributes (e.g a mutable object like a list which you wouldn't want multiple instances to share)

from loguru import logger # for us to handle our logs. Logs are useful because they do not simply print text but comes with levels (info, warning, error...) and timestamps

from torch import nn # the neural network module from Pytorch

from torchfeather.model.moe import MoEArgs

@dataclass # the dataclass decorator
class DeepseekV3ModelArgs: # we define our class
    max_seq_len: int = 4096 * 4 # the maximum sequence length
    vocab_size: int = 102400 # the number of different unique tokens we can process
    dim: int = 2048 # the main hidden dimension our tokens have in the model
    inter_dim: int = 10944 # the hidden dimension inside a dense feed forward neural network
    moe_inter_dim: int = 1408 # the hidden dimension inside an expert feed forward neural network
    n_layers: int = 27 # number of layers
    n_dense_layers: int = 1 # number of dense layers
    n_heads: int = 16 # number of attention heads
    norm_eps: float=1e-5 # this is the epsilon we are going to use in the RMSNorm

    moe_args: MoEArgs = field(default_factory=MoEArgs) # whenever a new DeepseekV3ModelArgs is created, create a brand new MoEArgs for it

    # Multi-Head Latent Attention (MLA)
    q_lora_rank: int = 0 # TBD, I do not understand this yet
    kv_lora_rank: int = 512 # TBD, I do not understand this yet
    qk_nope_head_dim: int = 128 # TBD, I do not understand this yet
    qk_rope_head_dim: int= 64 # TBD, I do not understand this yet
    v_head_dim = 128 # TBD, I do not understand this yet

    #YaRN
    original_seq_len: int=4096 # The original sequence length RoPE was calibrated for
    rope_theta: float=10,000.0 # this is RoPE base value and it controls the spacing between frequencies
    rope_factor: int=40 # TBD, I do not understand this yet
    beta_fast: int=32 # TBD, I do not understand this yet
    beta_slow:int=1 # TBD, I do not understand this yet
    mscale:float=1.0 # TBD, I do not understand this yet

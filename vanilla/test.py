import torch
from model import Sona
from dataclasses import dataclass

device="cpu"

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 100278
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384
    sample_rate: int = 48000
    out_channels: int = 384
    kernel_size: int = 251
    stride: int = 1
    padding: int = 0


model = Sona(Config.sample_rate,Config(vocab_size=100278))
from torchinfo import summary
audio = torch.randn(1, 1, 160797).to(device=device)                    # dummy audio batch
text  = torch.randint(0, 100278, (1, 39)).to(device=device)
print(text.shape)

summary(model, input_data=[audio, text])
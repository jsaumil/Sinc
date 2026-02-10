import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import inspect
import torchaudio
import tiktoken

from sinc import SincConv_fast

enc = tiktoken.get_encoding("cl100k_base")

def encode(text):
   return enc.encode(text, allowed_special="all")

im_start_id = enc.encode("<|endoftext|>", allowed_special="all")
im_end_id = enc.encode("<|endoftext|>", allowed_special="all")

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.size()

        # generate PE dynamically
        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, C, 2, device=x.device) * (-math.log(10000.0) / C))

        pe = torch.zeros(T, C, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe  # [B, T, C]

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class EncoderCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # qkv
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        # calculate qkv for all heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1,2).contiguous().view(B,T,C)
        # output projection
        y = self.c_proj(y)
        return y

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = EncoderCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

class DecoderCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #k,q,v projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x. size()
        # calculate q, k, v for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k , v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)
        #output projection
        y = self.c_proj(y)
        return y
    
class CausalCrossAttentionHead(nn.Module):
    """One head of cross-attention"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd %  config.n_head == 0
        # qkv
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self, embedding_q, embedding_kv):
        B,T,C = embedding_q.shape
        k = self.key(embedding_kv)
        v = self.value(embedding_kv)
        q = self.query(embedding_q)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        y = self.c_proj(y)

        return y
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = DecoderCausalSelfAttention(config)
        self.cross_attn = CausalCrossAttentionHead(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x))
        x = x + self.mlp(self.ln_3(x))
        return x
    

class Sona(nn.Module):
    def __init__(self, sample_rate, config):
        super().__init__()

        self.config = config
        self.sample_rate = sample_rate

        self.act1 = nn.GELU()
        self.sinc = SincConv_fast(out_channels=4*config.out_channels, kernel_size=config.kernel_size, sample_rate=sample_rate, stride=config.stride, padding=config.padding)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv1 = nn.Conv1d(in_channels=4*config.out_channels, out_channels=2*config.out_channels, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=2*config.out_channels, out_channels=config.out_channels, kernel_size=10, stride=3)
        self.encoder = nn.ModuleDict(dict(
            sin_pos = SinusoidalPositionEncoding(embed_size=config.out_channels),
            e = nn.ModuleList([Encoder(config) for _ in range(config.n_layer)]),
            e_ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.decoder = nn.ModuleDict(dict(
            d_wte = nn.Embedding(config.vocab_size, config.n_embd), # what padding_idx does here
            d_pos = nn.Embedding(config.block_size, config.n_embd),
            d = nn.ModuleList([Decoder(config) for _ in range(config.n_layer)]),
            d_ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, audio, idx, targets=None):
        sinc = self.sinc(audio)
        sinc = self.pool(sinc)
        cnn = self.conv1(cnn)
        cnn = self.pool(cnn)
        cnn = self.conv2(cnn)
        cnn = self.act1(cnn)
        cnn = self.pool(cnn)
        cnn = cnn.transpose(1,2)
        audio_pos_emb = self.encoder.sin_pos(cnn)
        enc_out = audio_pos_emb
        for block in self.encoder.e:
            enc_out = block(enc_out)

        enc_out = self.encoder.e_ln_f(enc_out)

        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward, model block size is exhausted: {T} > {self.block_size}"

        text_pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.decoder.pos(text_pos)
        tok_embd = self.decoder.d_wte(idx)
        dec_out = tok_embd + pos_embd

        for block in self.decoder.d:
            dec_out = block(dec_out, enc_out)

        x = self.decoder.d_ln_f(dec_out)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}

        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_group = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=(0.9,0.95), eps=1e-8)
        return optimizer
    
    def generate(self,audio_path):
        self.eval()
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            
        waveform, sr = torchaudio.load(audio_path)  # waveform shape: (channels, samples)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.unsqueeze(0).to(device)
        y = torch.tensor([im_start_id], device=device)
        generated = []
        with torch.no_grad():
          while True:
            logits, loss = self(waveform, y)
            next_token = logits[:,-1].argmax(-1).item()

            if next_token == im_end_id:
              break
            generated.append(next_token)

            y = torch.cat([y, torch.tensor([[next_token]], device=device)], dim=1)

        return logits
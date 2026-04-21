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


class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, config):
        super().__init__()

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embedding):
        B,T,C = embedding.shape
        k = self.key(embedding)
        # print(k.shape)
        # print(k)
        q = self.query(embedding)
        # print(q.shape)
        # print(q)
        scale = q.size(-1)**0.5
        wei = (q @ k.transpose(-2,-1))/scale
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(embedding)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self,x):
        # print(x.shape)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Encoder(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = MLP(config)
        # self.ffwd = MoE(n_embd=n_embd,num_experts=10,k=1)
        # self.ffwd = SwitchFeedForward(capacity_factor=1, drop_tokens=True, is_scale_prob=True, n_experts=4, n_embd=n_embd, d_model=n_embd)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        # x = x + self.ffwd(self.ln2(x))
        x_ln = self.ln2(x)
        # B, T, C = x_ln.shape

        # x_flat = x_ln.view(B*T, C)
        x, loss = self.ffwd(x_ln)

        return x, loss
    
class DecoderHead(nn.Module):
    """one head of self-attention"""
    def __init__(self, config):
        super().__init__()

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(0.2)

    def forward(self, embedding):
        B,T,C = embedding.shape
        k = self.key(embedding)
        # print(k.shape)
        # print(k)
        q = self.query(embedding)
        # print(q.shape)
        # print(q)
        scale = q.size(-1)**0.5
        wei = (q @ k.transpose(-2,-1))/scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(embedding)
        out = wei @ v

        return out

class MultiHeadAttentionDecoder(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([DecoderHead(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class CrossAttentionHead(nn.Module):
    """One head of cross-attention"""
    def __init__(self, config):
        super().__init__()

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embedding_q, embedding_kv):
        B,T,C = embedding_q.shape
        k = self.key(embedding_kv)
        # print(k.shape)
        # print(k)
        q = self.query(embedding_q)
        # print(q.shape)
        # print(q)
        scale = q.size(-1)**0.5
        wei = (q @ k.transpose(-2,-1))/scale
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(embedding_kv)
        out = wei @ v

        return out

class MultiHeadCrossAttention(nn.Module):
    """multiple head of cross attention in parallel"""
    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, y):
        out = torch.cat([h(x,y) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Decoder(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, config):
        super().__init__()
        self.self_sa = MultiHeadAttentionDecoder(config)
        self.cross_sa = MultiHeadCrossAttention(config)
        self.ffwd = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(0.2)

        # self.tril = torch.tril(torch.ones(1000, 1000))

    def forward(self,x, enco_out):
        x = x + self.self_sa(self.ln1(x))
        x = x + self.cross_sa(self.ln2(x), enco_out)
        # x = x + self.ffwd(self.ln3(x))
        x_ln = self.ln2(x)
        # B, T, C = x_ln.shape

        # x_flat = x_ln.view(B*T, C)
        x, aux_loss = self.ffwd(x_ln)
        # y = y_flat.view(B, T, C)

        # x = x + self.dropout(y)
        return x, aux_loss
    
class Sona(nn.Module):
    def __init__(self, sample_rate, config):
        super().__init__()

        self.config = config
        self.sample_rate = sample_rate

        self.act1 = nn.GELU()
        self.sinc = SincConv_fast(out_channels=4 * config.n_embd, kernel_size=config.kernel_size, sample_rate=sample_rate, stride=config.stride, padding=config.padding)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv1 = nn.Conv1d(in_channels=4*config.out_channels, out_channels=2*config.out_channels, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=2*config.out_channels, out_channels=config.out_channels, kernel_size=5, stride=2)
        self.encoder = nn.ModuleDict(dict(
            sin_pos = SinusoidalPositionEncoding(embed_size=config.out_channels),
            e = nn.ModuleList([Encoder(config) for _ in range(config.n_layer)]),
            e_ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.decoder = nn.ModuleDict(dict(
            d_wte = nn.Embedding(config.vocab_size, config.n_embd),
            d_pos = nn.Embedding(config.block_size, config.n_embd),
            d = nn.ModuleList([Decoder(config) for _ in range(config.n_layer)]),
            d_ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #init params
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
        cnn = self.conv1(sinc)
        cnn = self.pool(cnn)
        cnn = self.conv2(cnn)
        cnn = self.act1(cnn)
        cnn = self.pool(cnn)
        cnn = cnn.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        audio_pos_emb = self.encoder.sin_pos(cnn)
        enc_out = audio_pos_emb
        for block in self.encoder.e:
            enc_out, _ = block(enc_out)

        enc_out = self.encoder.e_ln_f(enc_out)

        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward, model block size is exhausted: {T} > {self.config.block_size}"

        text_pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.decoder.d_pos(text_pos)
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
            {'params': decay_params, 'weight_decay': weight_decay},
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
    
    def generate(self, audio_path):
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
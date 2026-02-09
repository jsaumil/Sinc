import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

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

class SwitchFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.capacity_factor = config.capacity_factor
        self.is_scale_prob = config.is_scale_prob
        self.n_experts = config.n_experts
        self.drop_tokens = config.drop_tokens
        self.n_embd = config.n_embd
        self.experts = nn.ModuleList([MLP(self.n_embd) for i in range(config.n_experts)])

        self.switch = nn.Linear(config.d_model, config.n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(-1, C)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        final_output = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        dropped = []

        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) <= capacity:
                    continue

                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                dropped.append(indexes_list[i][capacity:])

                indexes_list[i] = indexes_list[i][:capacity]

            expert_output = [self.experts[i](x[indexes_list[i],:]) for i in range(self.n_experts)]

            for i in range(self.n_experts):
                final_output[indexes_list[i], :] = expert_output[i].to(final_output.dtype)

            if dropped:
                dropped = torch.cat(dropped)
                final_output[dropped, :] = x[dropped, :]

            if self.is_scale_prob:
                final_output = final_output * route_prob_max.view(-1,1)
            else:
                final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1,1)

            final_output = final_output.view(B,T,C)

            return final_output

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
        self.ln_2 = nn.Linear(config.n_embd)
        # self.mlp = EncoderMLP(config)
        self.moe = SwitchFeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        x = x + self.moe(self.ln_2(x))
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
        self.register_buffer("bias",torch.tril(torch.one(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

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
        # self.mlp = DecoderMLP(config)
        self.moe = SwitchFeedForward(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x))
        # x = x + self.mlp(self.ln_3(x))
        x = x + self.moe(self.ln_3(x))
        return x
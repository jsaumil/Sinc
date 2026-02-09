import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4*n_embd)
        self.fc2 = nn.Linear(4*n_embd, n_embd)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SwitchFeedForward(nn.Module):
    def __init__(self, capacity_factor, drop_tokens, is_scale_prob, n_experts, n_embd, d_model):
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens
        self.n_embd = n_embd
        self.experts = nn.ModuleList([MLP(self.n_embd) for i in range(n_experts)])
        
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(-1, C)
        
        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]
        
        final_output = x.new_zero(x.shape)

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
            final_output[indexes_list[i], :] = expert_output[i]

        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            final_output = final_output * route_prob_max.view(-1,1)
        else:
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1,1)
            
        final_output = final_output.view(B,T,C)

        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max
    
    

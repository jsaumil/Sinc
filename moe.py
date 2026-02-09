import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a sparsedispatcher"""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1,dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:,1],0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        Create one input Tensor for each expert
        Args:
            inp: a 'Tensor' of shape "[batch_size, <extra_input_dims>]"
        Returns:
            a list of 'num_experts' Tensor's with shapes
            '[expert_batch_size_i, <extra_input_dims>]'
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_size
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)
    
    def combine(self, expert_out, multiply_by_gates=True):
        """
        Sum together the expert output, weighted by the gates.
        Args:
            expert_out: a list 'num_experts' Tensor's each with shape
            '[expert_batch_size_i, <extra_output_dims>]'
            multiply_by_gates: a boolean
        Returns: 
            a Tensor's with shape '[batch_size, <extra_output_dims>]'
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine sample that have been processed by same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined
    
    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert Tensor's
        Returns:
            a list of 'num_experts' one-dimensional Tensor's with type 'tf.float32'
            and shapes '[expert_batch_size_i]'
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out
    
class MoE(nn.Module):
    """
    Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
        input_size: integer - size of the input
        output_size: integer - size of the input
        num_experts: an integer - number of experts
        hidden_size: an integer - hidden size of the experts
        noisy_gating: a boolean
        k: an integer - how many experts to use for each batch element
    """
    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """
        The squared coefficient of variation of sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability
        Returns 0 for an empty Tensor.
        Args:
        x: a 'Tensor'
        Returns:
        a 'Scalar'
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.type)
        return x.float().var()/(x.float().mean()**2 + eps)
    
    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        Args:
            clean_values: a Tensor of shape [batch, n]
            noisy_values: a Tensor of shape [batch, n]. Equal to clean values plus
                normally distributed noise with standard deviation noise_stddev
            noise_stddev: a Tensor of shape [batch, n] or None
            noisy_top_values: a Tensor of shape [batch, m]
                "values" Output of tf.top_k(noisy_top_values, m). m >= k+1
            Returns:
                a Tensor of shape [batch, n]
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device, dtype=torch.long) * m + (self.k - 1)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in) # tensor of true and false
        threshold_positions_if_out = threshold_positions_if_in - 1
        
        # FORCE dtype (defensive)
        # threshold_positions_if_in = threshold_positions_if_in.long()
        threshold_positions_if_out = threshold_positions_if_out.long()
        
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        Docstring for noisy_top_k_gating
        
        :param x: input Tensor with shape [batch_size, input_size]
        :param train: a boolean - we only add noise at training time
        :param noise_epsilon: a float

        Returns:
         gates: a Tensor with shape [batch_size, num_experts]
         load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:,:self.k]
        top_k_indices = top_indices[:,:self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_k_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
    
    def forward(self, x, loss_coef=1e-2):
        """
        Docstring for forward
        
        :param x: tensor shape [batch_size, input_size]
        :param loss_coef: a scalar - multiplier on load-balancing losses
        train: a boolean scalar

        Returns:
            y: a tensor with shape [batch_size, output_size]
            extra_training_loss: a scalar. this should be added into overall
            training loss of model. the backpropagation of this loss
            encourages all experts to be approximately equally used across a batch
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        #calculate importance loss
        importance = gates.sum(0)
        
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss
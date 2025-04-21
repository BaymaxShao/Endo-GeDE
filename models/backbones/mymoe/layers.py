import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class DVExpert(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_U = nn.Parameter(self.weight.new_zeros(r, 1))
            self.lora_V = nn.Parameter(self.weight.new_zeros(out_features, 1))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_U, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_V, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            if self.r > 0:
                result_delta = (self.lora_dropout(x) @ (self.lora_A * self.lora_U).T @ (
                            self.lora_B * self.lora_V).T) * self.scaling
            return result_delta
        else:
            return "Not using LoRA"


class LoRAExpert(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MoELayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        num_experts: int,
        top_k:int,
        dropout: float,
        merge_weights: bool
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts
        self.top_k = top_k
        # Optional dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        self.merge_weights = merge_weights


class MLPExpert(nn.Linear):
    def __init__(self, in_features, out_features, dropout=0.1, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(in_features , 4 * in_features),
            nn.ReLU(),
            nn.Linear(4 * in_features, out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)




class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices


class MoELinear(nn.Linear, MoELayer):
    def __init__(self, n_in, n_out, num_experts, top_k, r, lora_alpha, merge_weights=False, dropout=0.1, fan_in_fan_out=False, **kwargs):
        nn.Linear.__init__(self, n_in, n_out, **kwargs)
        MoELayer.__init__(self, r=r, lora_alpha=lora_alpha, num_experts=num_experts, top_k=top_k, dropout=dropout, merge_weights=merge_weights)
        self.router = NoisyTopkRouter(n_in, num_experts, top_k)
        self.experts = nn.ModuleList([DVExpert(n_in, n_out, r=r, lora_alpha=lora_alpha, lora_dropout=dropout, fan_in_fan_out=fan_in_fan_out,
                           merge_weights=merge_weights) for _ in range(num_experts)])
        self.out = n_out
        self.top_k = top_k
        self.fan_in_fan_out = fan_in_fan_out

    def forward(self, x):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        gating_output, indices = self.router(x)
        final_output = torch.zeros((x.shape[0], x.shape[1], self.out)).cuda()

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output*gating_scores

                final_output[expert_mask] += weighted_output.squeeze(1)
        result += final_output

        return result



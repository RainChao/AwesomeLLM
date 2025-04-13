import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class SparseMoEConfig:
    num_experts: int
    hidden_dim: int
    top_k: int


class BasicExpert(nn.Module):
    def __init__(self, config: SparseMoEConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x):
        return self.linear(x)

class SparseMoERouter(torch.nn.Module):
    def __init__(self, config: SparseMoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.num_experts)

    def forward(self, x):
        router_logits = self.gate(x) # [batch_size*context_lenth, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        router_weights, selected_experts = torch.topk(router_probs, k=self.config.top_k, dim=-1)
        # [batch_size*context_lenth, top_k]
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        # [batch_size*context_lenth, top_k, num_experts]
        expert_mask = F.one_hot(selected_experts, num_classes=self.config.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0) # [num_experts, top_k, batch_size*context_lenth]
        return router_weights, expert_mask


class SparseMoEModel(torch.nn.Module):
    def __init__(self, config: SparseMoEConfig):
        super().__init__()
        self.config = config
        self.router = SparseMoERouter(config)
        self.experts = nn.ModuleList([BasicExpert(config) for _ in range(config.num_experts)])
        self.topk = config.top_k

    def forward(self, x):
        batch_size, context_length, hidden_dim = x.size()
        x = x.view(-1, hidden_dim) # [batch_size * context_length, hidden_dim]
        router_weights, expert_mask = self.router(x) # [batch_size * context_length, top_k]
        final_output = torch.zeros_like(x, device=x.device) # [batch_size * context_length, hidden_dim]
        for expert_index in range(self.config.num_experts):
            expert_layer = self.experts[expert_index]
            topi, gidx = torch.where(expert_mask[expert_index])
            selected_seq = x[gidx, :] # [num_selected_seq, hidden_dim]
            expert_output = expert_layer(selected_seq) * router_weights[gidx, topi].unsqueeze(-1)
            final_output.index_add_(0, gidx, expert_output)
        final_output = final_output.reshape(batch_size, context_length, hidden_dim)
        return final_output


if __name__ == "__main__":
    batch_size = 2
    context_length = 3
    hidden_dim = 64
    top_k = 2
    num_experts = 4
    config = SparseMoEConfig(num_experts=num_experts, hidden_dim=hidden_dim, top_k=top_k)
    model = SparseMoEModel(config)
    x = torch.randn(batch_size, context_length, hidden_dim)
    output = model(x)
    print(output.shape)

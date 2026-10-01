import torch 
import torch.nn as nn 
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Layers 
        self.layer_norm = nn.LayerNorm(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.quant = self.absmax_quantization
        self.dequant = self.dequantization
        self.bitlinear = self.bitnet

        # Constants
        self.b = 8 # Can change
        self.Qb = 2 ** (self.b - 1)


    def forward(self, x):
        x = self.layer_norm(x)
        x = self.quant(x)
        x = self.bitlinear(x)
        x = self.dequant(x)
        return x

    def absmax_quantization(self, x, dim=-1, eps=1e-8):
        absmax = x.detach().abs().max(dim=dim, keepdim=True).values
        self.absmax = torch.maximum(absmax, torch.tensor(eps, device=x.device))
        x = x * (self.Qb / self.absmax)
        x = x.clamp(-self.Qb + eps, self.Qb - eps)
        x_q = x + (x.round() - x).detach()
        return x_q

    def dequantization(self, x):
        return x * (self.absmax / self.Qb)

    def bitnet(self, x, dim=-1):
        absavg = self.weight.detach().abs().mean(dim=dim, keepdim=True)
        wq = torch.clamp(torch.round(self.weight / absavg), -1, 1)
        q_w = wq * absavg
        weight_size = self.weight + (q_w - self.weight).detach()
        return F.linear(x, weight_size)


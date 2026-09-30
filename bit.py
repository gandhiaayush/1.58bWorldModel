import torch 

import torch.nn as nn 
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Layers 
        self.layer_norm = nn.LayerNorm(in_features)
        self.tlinear = nn.Parameter(torch.randn(out_features, in_features))


    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        return x


    def absmax_quantization(self, x, dim=1, eps=1e-8):
        # Absmax 
        abs_x = x.abs() 
        absmax = torch.max(abs_x, dim=dim, keepdim=True)
        absmax = torch.clamp(absmax, min=eps)
        return x / absmax


    def bit_weights(self, x, dim=1):
        # Bit Linear 
        quantized_tlinear = self.tlinear.detach().clone()
        abs_weights = quantized_tlinear.abs()
        absavg = torch.mean(abs_weights, dim=dim, keepdim=True)
        quantized_tlinear = torch.clamp(torch.round(quantized_tlinear / absavg), -1, 1)
        return F.linear(x, quantized_tlinear)
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymInfoNCE(nn.Module):
    def __init__(self, temperature=0.07, margin=1.0, use_margin=True, lambda_margin=0.3):
        super().__init__()
        self.t = temperature
        self.m = margin
        self.use_margin = use_margin
        self.lam = lambda_margin

    def forward(self, v, a, labels):
        B = v.size(0)
        logits = v @ a.t() / self.t   
        target = torch.arange(B, device=v.device)

        real_mask = (labels == 0)
        if real_mask.sum() > 1:
            v_r, a_r = v[real_mask], a[real_mask]
            logits_r = v_r @ a_r.t() / self.t
            tgt_r = torch.arange(v_r.size(0), device=v.device)
            l_info = 0.5 * (F.cross_entropy(logits_r, tgt_r)
                          + F.cross_entropy(logits_r.t(), tgt_r))
        else:
            l_info = torch.tensor(0.0, device=v.device)

        if self.use_margin:
            d = (v - a).pow(2).sum(-1)  
            y = 1.0 - labels.float()  
            l_mar = (y * d + (1 - y) * F.relu(self.m - d).pow(2)).mean()
        else:
            l_mar = torch.tensor(0.0, device=v.device)

        return l_info + self.lam * l_mar
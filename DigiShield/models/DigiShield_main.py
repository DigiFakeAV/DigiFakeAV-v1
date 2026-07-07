import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv3d(c // r, c, 1), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)


def temporal_shift(x, fold_div=8):
    B, C, T, H, W = x.shape
    fold = C // fold_div
    out = torch.zeros_like(x)
    out[:, :fold, :-1] = x[:, :fold, 1:]
    out[:, fold:2*fold, 1:] = x[:, fold:2*fold, :-1]
    out[:, 2*fold:] = x[:, 2*fold:]
    return out


class R2Plus1DBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None, use_tsm=True):
        super().__init__()
        mid = (3 * 3 * in_c * out_c) // (3 * 3 * in_c + 3 * out_c)
        self.spatial = nn.Sequential(
            nn.Conv3d(in_c, mid, (1,3,3), (1,stride,stride), (0,1,1), bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=True))
        self.temporal = nn.Sequential(
            nn.Conv3d(mid, out_c, (3,1,1), (1,1,1), (1,0,0), bias=False),
            nn.BatchNorm3d(out_c))
        self.se = SEBlock(out_c)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.use_tsm = use_tsm

    def forward(self, x):
        identity = x
        if self.use_tsm: x = temporal_shift(x)
        out = self.spatial(x)
        out = self.temporal(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class Visual3DBackbone(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, (3,7,7), (1,2,2), (1,3,3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((1,3,3), (1,2,2), (0,1,1)))
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, out_dim, 2, stride=2)

    def _make_layer(self, in_c, out_c, n, stride):
        down = None
        if stride != 1 or in_c != out_c:
            down = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, (1,stride,stride), bias=False),
                nn.BatchNorm3d(out_c))
        layers = [R2Plus1DBlock(in_c, out_c, stride, down)]
        for _ in range(n-1):
            layers.append(R2Plus1DBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x


class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c, c//r), nn.ReLU(inplace=True), nn.Linear(c//r, c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca
        avg_s = x.mean(1, keepdim=True); max_s, _ = x.max(1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


class AudioBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_c, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.cbam = CBAM(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out)
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)


class Audio2DBackbone(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.in_c = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make(64, 3, 1)
        self.layer2 = self._make(128, 4, 2)
        self.layer3 = self._make(256, 6, 2)
        self.layer4 = self._make(out_dim // 4, 3, 2)

    def _make(self, planes, n, stride):
        down = None
        if stride != 1 or self.in_c != planes*4:
            down = nn.Sequential(
                nn.Conv2d(self.in_c, planes*4, 1, stride, bias=False),
                nn.BatchNorm2d(planes*4))
        layers = [AudioBottleneck(self.in_c, planes, stride, down)]
        self.in_c = planes*4
        for _ in range(n-1):
            layers.append(AudioBottleneck(self.in_c, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x


class MHA(nn.Module):
    def __init__(self, d, h=8, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop)
    def forward(self, q, k, v):
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)
        out, _ = self.attn(q, k, v)
        return out.transpose(0, 1)


class FFN(nn.Module):
    def __init__(self, d, mlp=4, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d*mlp), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d*mlp, d), nn.Dropout(drop))
    def forward(self, x): return self.net(x)


class FusionBlock(nn.Module):
    def __init__(self, d=1024, h=8, drop=0.1):
        super().__init__()
        self.ln_q1 = nn.LayerNorm(d); self.ln_kv1 = nn.LayerNorm(d)
        self.cross = MHA(d, h, drop)
        self.ln_s1 = nn.LayerNorm(d); self.self1 = MHA(d, h, drop)
        self.ln_s2 = nn.LayerNorm(d); self.self2 = MHA(d, h, drop)
        self.ln_ff = nn.LayerNorm(d); self.ffn = FFN(d, 4, drop)

    def forward(self, x, ctx):
        x = x + self.cross(self.ln_q1(x), self.ln_kv1(ctx), self.ln_kv1(ctx))
        x_ = self.ln_s1(x); x = x + self.self1(x_, x_, x_)
        x_ = self.ln_s2(x); x = x + self.self2(x_, x_, x_)
        x = x + self.ffn(self.ln_ff(x))
        return x


class MultimodalSpatiotemporalFusion(nn.Module):
    def __init__(self, d=1024, n=2, h=8, drop=0.1):
        super().__init__()
        self.v_blocks = nn.ModuleList([FusionBlock(d, h, drop) for _ in range(n)])
        self.a_blocks = nn.ModuleList([FusionBlock(d, h, drop) for _ in range(n)])
        self.gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())

    def forward(self, fv, fa):
        for vb, ab in zip(self.v_blocks, self.a_blocks):
            fv_new = vb(fv, fa)
            fa_new = ab(fa, fv)
            fv, fa = fv_new, fa_new
        v_vec = fv.mean(1); a_vec = fa.mean(1)
        g = self.gate(torch.cat([v_vec, a_vec], dim=-1))
        z = g * v_vec + (1 - g) * a_vec
        return z, v_vec, a_vec


class DigiShield(nn.Module):
    def __init__(self, d=1024, n_fusion=2, num_classes=2, dropout=0.5,
                 max_v_tokens=2048, max_a_tokens=512):
        super().__init__()
        self.visual = Visual3DBackbone(out_dim=d)
        self.audio = Audio2DBackbone(out_dim=d)

        self.v_pos = nn.Parameter(torch.zeros(1, max_v_tokens, d))
        self.a_pos = nn.Parameter(torch.zeros(1, max_a_tokens, d))
        nn.init.trunc_normal_(self.v_pos, std=0.02)
        nn.init.trunc_normal_(self.a_pos, std=0.02)

        self.fusion = MultimodalSpatiotemporalFusion(d=d, n=n_fusion)

        self.v_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 256))
        self.a_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 256))

        self.down = nn.Sequential(
            nn.Linear(d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout))
        self.fc = nn.Linear(512, num_classes)

    def _flatten(self, feat, pos):
        x = feat.flatten(2).transpose(1, 2)   
        N = x.shape[1]
        N_pos = pos.shape[1]
        if N > N_pos:
            pos_ = pos.transpose(1, 2)                              
            pos_ = F.interpolate(pos_, size=N, mode='linear', align_corners=False)
            pos_ = pos_.transpose(1, 2)                           
            return x + pos_
        else:
            return x + pos[:, :N]

    def forward(self, video, audio):
        fv_map = self.visual(video)
        fa_map = self.audio(audio)
        fv = self._flatten(fv_map, self.v_pos)
        fa = self._flatten(fa_map, self.a_pos)

        z, v_vec, a_vec = self.fusion(fv, fa)
        logits = self.fc(self.down(z))

        v_p = F.normalize(self.v_proj(v_vec), dim=-1)
        a_p = F.normalize(self.a_proj(a_vec), dim=-1)
        return v_p, a_p, logits
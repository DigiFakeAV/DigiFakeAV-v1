import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model=1024, nhead=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class BiCroAtt(nn.Module):
    def __init__(self, d_model=1024, nhead=8):
        super().__init__()
        self.a2v_att = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.v2a_att = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, audio_feat, visual_feat):
        # Audio-to-Visual
        a2v_output, _ = self.a2v_att(visual_feat, audio_feat, audio_feat)
        # Visual-to-Audio
        v2a_output, _ = self.v2a_att(audio_feat, visual_feat, visual_feat)
        
        visual_feat = visual_feat + self.dropout(a2v_output)
        audio_feat = audio_feat + self.dropout(v2a_output)
        
        return self.norm(visual_feat), self.norm(audio_feat)

class FeedForward(nn.Module):
    def __init__(self, d_model=1024, d_ff=2048):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        return self.norm(x)

class AVTFLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_self_att = SelfAttention()
        self.visual_self_att = SelfAttention()
        
        self.cross_att = BiCroAtt()
        
        self.audio_ff = FeedForward()
        self.visual_ff = FeedForward()

    def forward(self, audio_feat, visual_feat):
        audio_feat = self.audio_self_att(audio_feat)
        visual_feat = self.visual_self_att(visual_feat)
        
        visual_feat, audio_feat = self.cross_att(audio_feat, visual_feat)
        
        audio_feat = self.audio_ff(audio_feat)
        visual_feat = self.visual_ff(visual_feat)
        
        return audio_feat, visual_feat

class AVTF(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([AVTFLayer() for _ in range(num_layers)])
        self.final_fusion = nn.Linear(2048, 1024) 

    def forward(self, audio_feat, visual_feat):
        audio_feat = audio_feat.unsqueeze(1)
        visual_feat = visual_feat.unsqueeze(1)  
        
        for layer in self.layers:
            audio_feat, visual_feat = layer(audio_feat, visual_feat)
        
        fused_feat = torch.cat([audio_feat, visual_feat], dim=-1)
        fused_feat = fused_feat.squeeze(1)  
        
        return self.final_fusion(fused_feat) 

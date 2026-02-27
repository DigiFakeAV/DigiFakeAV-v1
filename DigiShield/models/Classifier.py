
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, 
                 fusion_dim=1024, 
                 num_classes=2,  
                 dropout_prob=0.5):
        super().__init__()
        
        self.refine = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),           
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes)
        )

    def forward(self, fused_feat):
        x = self.refine(fused_feat)
        return self.classifier(x)

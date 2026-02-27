import torch
import torch.nn as nn
from models.VA_Net import VA_Net
from models.Xception import Xception
from models.AVTF import AVTF
from models.Classifier import Classifier

class DigiShield(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_backbone = VA_Net(img_dim=224, network='resnet50')
        
        self.audio_backbone = Xception()
        
        self.avt_fusion = AVTF(num_layers=2)
        
        self.classifier = Classifier()

    def forward(self, visual_input, audio_input):

        visual_feat = self.visual_backbone(visual_input)  
        audio_feat = self.audio_backbone(audio_input)    
        
        fused_feat = self.avt_fusion(visual_feat, audio_feat)
        
        logits = self.classifier(fused_feat)  

        return visual_feat, audio_feat, logits 


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet3D import select_resnet
import math

class VA_Net(nn.Module):
    def __init__(self, img_dim, network='resnet50', num_layers_in_fc_layers=1024, dropout=0.5):
        super(VA_Net, self).__init__()

        self.netcnnlip, self.param = select_resnet(network, track_running_stats=False)
        
        self.last_size = int(math.ceil(img_dim / 32))  

        self.space_pool = nn.AdaptiveAvgPool2d((self.last_size, self.last_size))  
        self.netfclip = nn.Sequential(
            nn.Linear(self.param['feature_size'] * self.last_size * self.last_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_layers_in_fc_layers),
        )
        
        self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_lip.weight.data.fill_(1)
        self.final_bn_lip.bias.data.zero_()

        self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        self._initialize_weights(self.final_fc_lip)

        self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_aud.weight.data.fill_(1)
        self.final_bn_aud.bias.data.zero_()

    def forward(self, x):
        
        x = x.permute(0, 2, 1, 3, 4)
        batch_size, C, num_frames, H, W = x.shape

        feature = self.netcnnlip(x) 

        t_dim = feature.size(2)  
        feature = F.avg_pool3d(
            feature, 
            kernel_size=(t_dim, 1, 1),  
            stride=(1, 1, 1)
        ).squeeze(dim=2)  

        feature = self.space_pool(feature)

        feature = feature.view(batch_size, -1)

        out = self.netfclip(feature)
        out = self.final_bn_lip(out)
        return out

    def _initialize_weights(self, module):
        for m in module:
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, (nn.ReLU, nn.MaxPool2d, nn.Dropout)):
                continue
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
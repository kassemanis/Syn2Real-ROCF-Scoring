import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from mixstyle import MixStyle

class LaplacianFilter(nn.Module):
    """
    Laplacian filter for edge detection. This layer is not trainable.
    """
    def __init__(self, in_channels):
        super(LaplacianFilter, self).__init__()
        

        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32)
        

        self.kernel = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        

        self.groups = in_channels
        self.padding = 1

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.groups)

class AttentionModule(nn.Module):
    """
    Attention module that uses edge detection (Laplacian filter) to generate an attention map.
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.laplacian_filter = LaplacianFilter(in_channels)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        edges = self.laplacian_filter(x)
        attention_map = self.attention_conv(edges)
        return x * attention_map, attention_map

class ROCFNet(nn.Module):

    def __init__(self, num_classes=1):
        super(ROCFNet, self).__init__()
        pretrained_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        
        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4
        
        self.mixstyle1 = MixStyle()
        self.mixstyle2 = MixStyle()
        
        self.attention = AttentionModule(1024)  


        self.avgpool = pretrained_resnet.avgpool
        self.fc = nn.Sequential(
            nn.Linear(pretrained_resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Block 1 with MixStyle
        x = self.layer1(x)
        if self.training:
            x = self.mixstyle1(x)
        
        # Block 2 with MixStyle
        x = self.layer2(x)
        if self.training:
            x = self.mixstyle2(x)
        
        # Block 3 with attention
        x = self.layer3(x)
        x, _ = self.attention(x)
        
        # Block 4 
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
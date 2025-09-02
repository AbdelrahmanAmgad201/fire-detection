import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AutoModel.from_pretrained("facebook/dinov3-convnext-base-pretrain-lvd1689m")

        def forward(self, x):
             outputs = self.model(**x)
             return outputs.last_hidden_state
        
class Neck(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)

    def froward (self,x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    


class YoloDetectionHead(nn.Module):
    def __init__(self, in_channels=1024, num_classes=2, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.act1 = nn.ReLU()  

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.act2 = nn.SiLU()

        
        self.pred = nn.Conv2d(
            256, num_anchors * (num_classes + 5), kernel_size=1
        )

    def forward(self, x):
        
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        pred = self.pred(x)

        
        B, _, H, W = pred.shape
        pred = pred.view(B, self.num_anchors, self.num_classes + 5, H, W)
        return pred




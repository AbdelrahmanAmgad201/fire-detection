import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.engine.results import Results
# from ultralytics.utils.ops import non_max_suppression, scale_boxes

from types import SimpleNamespace

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        x = self.norm(features[-1].mean([-2, -1]))  # global average pooling
        x = self.head(x)
        return x

def convnext_tiny(pretrained=False, weights_path=None, **kwargs):
    """ConvNext Tiny model"""
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    
    if pretrained and weights_path:
        print(f"🔄 Loading pretrained weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("✅ Pretrained weights loaded successfully!")
    elif pretrained:
        print("⚠️ Pretrained=True but no weights_path provided")
        
    return model

class ConvNextBackbone(nn.Module):
    """ConvNext backbone optimized for YOLO integration"""
    
    def __init__(self, weights_path=None, pretrained=True):
        super().__init__()
        
        # Create ConvNext Tiny with official architecture
        self.convnext = convnext_tiny(pretrained=pretrained, weights_path=weights_path)
        
        # Remove classification head (we only need features)
        self.convnext.norm = nn.Identity()
        self.convnext.head = nn.Identity()
        
        # ConvNext Tiny feature dimensions: [96, 192, 384, 768]
        self.feature_dims = [192, 384, 768]  # Last 3 stages for multi-scale detection
        target_dims = [256, 512, 1024]       # YOLO expected channels
        
        # Feature adaptation layers
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ) for in_dim, out_dim in zip(self.feature_dims, target_dims)
        ])
        
        print("✅ ConvNext Tiny backbone ready for YOLO integration")
        print(f"📊 Feature channels: {self.feature_dims} → {target_dims}")

    def forward(self, x):
        # Extract multi-scale features from ConvNext
        features = self.convnext.forward_features(x)
        
        # Take last 3 feature maps: [Stage1: 192ch, Stage2: 384ch, Stage3: 768ch]
        selected_features = features[-3:]
        
        # Adapt to YOLO expected channels
        adapted_features = []
        for feat, adapter in zip(selected_features, self.adapters):
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
        
        return adapted_features



ultra_model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results=ultra_model(frame, conf=0.6)
    annotated = results[0].plot()
    cv2.imshow("Detections", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
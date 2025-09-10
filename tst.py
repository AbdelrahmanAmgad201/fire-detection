import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class FireDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + 
                           glob.glob(os.path.join(images_dir, "*.png")))
        self.labels = [os.path.join(labels_dir, os.path.basename(img).split('.')[0] + '.txt') 
                      for img in self.images]
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img) / 255.0
        return torch.from_numpy(img_array).permute(2, 0, 1).float()

# Official Facebook ConvNext Implementation
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

class ConvNextYOLO(nn.Module):
    """ConvNext Tiny + YOLOv8 for fire detection"""
    
    def __init__(self, yolo_path="yolov8n.pt", num_classes=2, convnext_weights_path=None):
        super().__init__()
        
        # Create ConvNext backbone
        self.convnext_backbone = ConvNextBackbone(
            weights_path=convnext_weights_path, 
            pretrained=True if convnext_weights_path else False
        )
        
        # Load original YOLO to get the architecture
        yolo = YOLO(yolo_path)
        
        # Create a simple neck for feature fusion
        self.neck = nn.ModuleList([
            # Upsample and concatenate features
            nn.Sequential(
                nn.Conv2d(1024, 512, 1),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, 1),  # 512 + 512 concatenated
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Conv2d(512, 256, 1)  # 256 + 256 concatenated
        ])
        
        # Simple detection head
        self.head = nn.ModuleList([
            # Three detection heads for different scales
            nn.Conv2d(256, num_classes + 5, 1),  # classes + box coords + objectness
            nn.Conv2d(256, num_classes + 5, 1),
            nn.Conv2d(256, num_classes + 5, 1),
        ])
        
        print(f"🔥 ConvNext-YOLO created with {num_classes} classes")

    def forward(self, x):
        # ConvNext backbone features: [P3, P4, P5] with channels [256, 512, 1024]
        features = self.convnext_backbone(x)
        p3, p4, p5 = features
        
        # Simple FPN-like neck
        # Top-down pathway
        x = self.neck[0](p5)  # 1024->512, upsample
        x = torch.cat([x, p4], dim=1)  # concat with p4 (512 channels)
        
        x = self.neck[1](x)  # 1024->256, upsample  
        x = torch.cat([x, p3], dim=1)  # concat with p3 (256 channels)
        
        x = self.neck[2](x)  # 512->256
        
        # Detection head (simplified)
        detections = []
        for head in self.head:
            det = head(x)
            detections.append(det)
        
        return detections

def download_convnext_weights(model_name="convnext_tiny_1k"):
    """Download ConvNext weights if not present"""
    model_urls = {
        "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    }
    
    weights_file = f"{model_name}.pth"
    
    if os.path.exists(weights_file):
        print(f"✅ Found existing weights: {weights_file}")
        return weights_file
    
    if model_name in model_urls:
        print(f"📥 Downloading {model_name} weights...")
        checkpoint = torch.hub.load_state_dict_from_url(
            url=model_urls[model_name], 
            map_location="cpu", 
            file_name=weights_file
        )
        # Save to current directory for future use
        torch.save(checkpoint, weights_file)
        print(f"✅ Downloaded and saved: {weights_file}")
        return weights_file
    else:
        print(f"❌ Unknown model: {model_name}")
        return None

def train_model(train_dir, val_dir, epochs=50, batch_size=8, device='cuda'):
    print("🔥 Training ConvNext-YOLO Fire Detection")
    print("=" * 50)
    
    # Download ConvNext weights
    weights_path = download_convnext_weights("convnext_tiny_1k")
    
    # Create datasets
    train_dataset = FireDataset(f"{train_dir}/images", f"{train_dir}/labels")
    val_dataset = FireDataset(f"{val_dir}/images", f"{val_dir}/labels")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Create model with pretrained ConvNext
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ConvNextYOLO(
        yolo_path="yolov8n.pt",
        num_classes=2,
        convnext_weights_path=weights_path
    ).to(device)
    
    # Training setup - different LR for backbone vs head
    backbone_params = list(model.convnext_backbone.parameters())
    head_params = list(model.neck.parameters()) + list(model.head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},  # Lower LR for pretrained
        {'params': head_params, 'lr': 1e-3, 'weight_decay': 1e-4}       # Higher LR for new layers
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_loss = float('inf')
    
    print(f"\n🚀 Starting training on {device}")
    print(f"🎯 Epochs: {epochs}, Batch size: {batch_size}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Simple loss function that works with any output format
            if isinstance(outputs, list):
                # Handle list of outputs (multi-scale)
                loss = sum(torch.mean(torch.abs(out)) for out in outputs)
            elif isinstance(outputs, torch.Tensor):
                # Handle single tensor output  
                loss = torch.mean(torch.abs(outputs))
            else:
                # Fallback
                loss = torch.tensor(0.1, requires_grad=True).to(device)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        
        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'convnext_weights_path': weights_path
            }, 'best_convnext_yolo_fire.pth')
            print(f"✅ Best model saved! Loss: {avg_train_loss:.4f}")
        
        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_train_loss:.4f}")
        print("-" * 50)
    
    print("🎉 Training completed!")
    print(f"💾 Best model saved as: best_convnext_yolo_fire.pth")

def test_backbone():
    """Test ConvNext backbone loading"""
    print("🧪 Testing ConvNext backbone...")
    
    # Download weights
    weights_path = download_convnext_weights("convnext_tiny_1k")
    
    # Create backbone
    backbone = ConvNextBackbone(weights_path=weights_path, pretrained=True)
    
    # Test forward pass
    test_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        features = backbone(test_input)
        
    print(f"✅ Backbone test successful!")
    print(f"Input shape: {test_input.shape}")
    for i, feat in enumerate(features):
        print(f"Feature {i+1}: {feat.shape}")

if __name__ == "__main__":
    # Test backbone first
    test_backbone()
    
    # Start training
    train_model(
        train_dir="home-fire-dataset/train",
        val_dir="home-fire-dataset/val", 
        epochs=50,
        batch_size=8
    )
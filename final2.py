import torch
import torch.nn as nn
from transformers import ConvNextModel, ConvNextConfig
from ultralytics import YOLO
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import yaml
import cv2

class FireDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + 
                           glob.glob(os.path.join(images_dir, "*.png")))
        self.labels_dir = labels_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.images)
    
    def _load_labels(self, label_path):
        """Load YOLO format labels and convert coordinates"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        labels.append([class_id, x_center, y_center, width, height])
        return np.array(labels) if labels else np.zeros((0, 5))

    def _scale_labels(self, labels, original_size, target_size):
        """Scale bounding boxes when resizing image"""
        if len(labels) == 0:
            return labels
        
        labels = labels.copy()
        # YOLO format is normalized, so coordinates remain the same when image is resized
        return labels

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Create dummy image if file doesn't exist
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        original_size = img.shape[:2]  # (height, width)
        
        # Load labels
        label_path = os.path.join(self.labels_dir, 
                                 os.path.basename(img_path).split('.')[0] + '.txt')
        labels = self._load_labels(label_path)
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Scale labels
        labels = self._scale_labels(labels, original_size, (self.img_size, self.img_size))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        
        return img_tensor, torch.from_numpy(labels).float()

class ConvNextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        try:
            # Use a smaller ConvNext model for better compatibility
            model_name = "facebook/convnext-tiny-224"
            self.backbone = ConvNextModel.from_pretrained(model_name)
            print(f"✅ Loaded ConvNext: {model_name}")
            self.use_convnext = True
        except Exception as e:
            print(f"⚠️ ConvNext failed ({e}), using ResNet50 fallback")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.use_convnext = False
        
        self._setup_feature_adaptation()
        
    def _setup_feature_adaptation(self):
        """Setup feature adaptation layers to match YOLOv8 expectations"""
        if self.use_convnext:
            # ConvNext-tiny has 768 feature dimensions at 7x7 for 224x224 input
            feature_dim = 768
            
            # Create multi-scale features matching YOLO expectations
            self.scale_adapters = nn.ModuleList([
                # P3: 80x80 (stride 8)
                nn.Sequential(
                    nn.ConvTranspose2d(feature_dim, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                ),
                # P4: 40x40 (stride 16)  
                nn.Sequential(
                    nn.ConvTranspose2d(feature_dim, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                # P5: 20x20 (stride 32)
                nn.Sequential(
                    nn.Conv2d(feature_dim, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            ])
        else:
            # ResNet features
            self.scale_adapters = nn.ModuleList([
                nn.Conv2d(2048, 128, 1),
                nn.Conv2d(2048, 256, 1), 
                nn.Conv2d(2048, 512, 1)
            ])
    
    def _extract_raw_features(self, x):
        """Extract raw features from backbone"""
        if self.use_convnext:
            # Resize input to 224x224 for ConvNext
            x_resized = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = self.backbone(x_resized)
            return outputs.last_hidden_state  # Shape: [B, H*W, C]
        else:
            return self.backbone(x)
    
    def forward(self, x):
        """Forward pass returning multi-scale features for YOLO"""
        features = self._extract_raw_features(x)
        
        if self.use_convnext:
            # Reshape ConvNext output: [B, H*W, C] -> [B, C, H, W]
            B, HW, C = features.shape
            H = W = int(HW ** 0.5)  # Assuming square feature map (7x7 for ConvNext-tiny)
            features = features.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Generate multi-scale features
        multi_scale_features = []
        for i, adapter in enumerate(self.scale_adapters):
            adapted_features = adapter(features)
            # Resize to expected YOLO feature map sizes
            if i == 0:  # P3: 80x80
                adapted_features = torch.nn.functional.interpolate(
                    adapted_features, size=(80, 80), mode='bilinear', align_corners=False)
            elif i == 1:  # P4: 40x40
                adapted_features = torch.nn.functional.interpolate(
                    adapted_features, size=(40, 40), mode='bilinear', align_corners=False)
            else:  # P5: 20x20
                adapted_features = torch.nn.functional.interpolate(
                    adapted_features, size=(20, 20), mode='bilinear', align_corners=False)
            
            multi_scale_features.append(adapted_features)
            
        return multi_scale_features

class ConvNextYOLO(nn.Module):
    def __init__(self, yolo_path="yolov8n.pt", num_classes=2):
        super().__init__()
        
        # Load YOLO model
        yolo = YOLO(yolo_path)
        self.yolo_model = yolo.model
        
        # Replace backbone with ConvNext
        self.convnext_backbone = ConvNextBackbone()
        
        # Extract YOLO detection head only (skip neck for simplicity)
        self.head = None
        for module in self.yolo_model.modules():
            if hasattr(module, 'cv2') and hasattr(module, 'cv3'):  # Detection head
                self.head = module
                break
        
        if self.head is None:
            # Create a simple detection head if not found
            self.head = self._create_simple_head(num_classes)
        else:
            self._update_head_for_classes(num_classes)
        
    def _create_simple_head(self, num_classes):
        """Create a simple detection head"""
        class SimpleDetectionHead(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes
                self.heads = nn.ModuleList([
                    # For each scale
                    nn.Conv2d(128, (num_classes + 5) * 3, 1),  # 3 anchors per scale
                    nn.Conv2d(256, (num_classes + 5) * 3, 1),
                    nn.Conv2d(512, (num_classes + 5) * 3, 1),
                ])
                
            def forward(self, x_list):
                outputs = []
                for i, x in enumerate(x_list):
                    out = self.heads[i](x)
                    outputs.append(out)
                return outputs
        
        return SimpleDetectionHead(num_classes)
        
    def _update_head_for_classes(self, num_classes):
        """Update detection head for custom number of classes"""
        if hasattr(self.head, 'nc'):
            self.head.nc = num_classes
        
        # Safely update classification heads
        if hasattr(self.head, 'cv3'):  # Classification head
            for i, cv3_module in enumerate(self.head.cv3):
                if hasattr(cv3_module, '__len__') and len(cv3_module) > 0:
                    # Get the last layer (usually the output conv)
                    last_idx = len(cv3_module) - 1
                    old_conv = cv3_module[last_idx]
                    
                    if isinstance(old_conv, nn.Conv2d):
                        # Calculate output channels (typically num_classes * num_anchors)
                        num_anchors = 3  # Default for YOLO
                        out_channels = num_classes * num_anchors
                        
                        new_conv = nn.Conv2d(
                            old_conv.in_channels,
                            out_channels,
                            old_conv.kernel_size,
                            old_conv.stride,
                            old_conv.padding,
                            bias=old_conv.bias is not None
                        )
                        
                        # Initialize weights
                        nn.init.normal_(new_conv.weight, 0.0, 0.01)
                        if new_conv.bias is not None:
                            nn.init.constant_(new_conv.bias, 0)
                        
                        cv3_module[last_idx] = new_conv
    
    def forward(self, x):
        """Forward pass through ConvNext backbone + YOLO head"""
        # Get multi-scale features from ConvNext backbone
        backbone_features = self.convnext_backbone(x)
        
        # Pass through detection head
        detections = self.head(backbone_features)
        
        return detections

def collate_fn(batch):
    """Custom collate function for batching variable-sized labels"""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pad labels to same size
    max_labels = max(len(l) for l in labels) if labels else 0
    if max_labels == 0:
        return images, torch.zeros((len(batch), 0, 5))
    
    padded_labels = []
    for label in labels:
        if len(label) == 0:
            padded_label = torch.zeros((max_labels, 5))
        else:
            padded_label = torch.zeros((max_labels, 5))
            padded_label[:len(label)] = label
        padded_labels.append(padded_label)
    
    return images, torch.stack(padded_labels, 0)

def compute_yolo_loss(predictions, targets, device):
    """Simplified YOLO loss computation"""
    if isinstance(predictions, (list, tuple)):
        # Multi-scale predictions
        total_loss = 0
        for pred in predictions:
            # Simple regression loss for now
            total_loss += torch.mean(pred ** 2) * 0.01
        return total_loss
    else:
        return torch.mean(predictions ** 2) * 0.01

def create_dummy_data():
    """Create some dummy data for testing"""
    # Create dummy images
    for split in ['train', 'val']:
        img_dir = f"home-fire-dataset/{split}/images"
        label_dir = f"home-fire-dataset/{split}/labels"
        
        # Create a few dummy images
        for i in range(5):
            # Create dummy image
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(f"{img_dir}/dummy_{i}.jpg", dummy_img)
            
            # Create dummy label
            with open(f"{label_dir}/dummy_{i}.txt", 'w') as f:
                # Fire class (0) at center with some size
                f.write("0 0.5 0.5 0.2 0.2\n")

def train_model(train_dir, val_dir, epochs=50, batch_size=8, device='cuda'):
    print("🔥 Training ConvNext-YOLO Fire Detection")
    
    # Create dummy data if directories are empty
    train_images = glob.glob(f"{train_dir}/images/*.jpg") + glob.glob(f"{train_dir}/images/*.png")
    if len(train_images) == 0:
        print("📝 Creating dummy data for testing...")
        create_dummy_data()
    
    # Create datasets
    train_dataset = FireDataset(f"{train_dir}/images", f"{train_dir}/labels")
    val_dataset = FireDataset(f"{val_dir}/images", f"{val_dir}/labels")
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    model = ConvNextYOLO(num_classes=2).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(images)
                loss = compute_yolo_loss(outputs, targets, device)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches > 0:
            avg_train_loss = train_loss / num_batches
            scheduler.step(avg_train_loss)
            
            print(f"📈 Epoch {epoch+1}/{epochs} - Avg Loss: {avg_train_loss:.4f}")
            
            # Save best model
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }, 'best_convnext_yolo.pth')
                print(f"✅ Best model saved! Loss: {avg_train_loss:.4f}")
    
    print("🎉 Training completed!")

def test_model(model_path="best_convnext_yolo.pth"):
    """Test the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNextYOLO(num_classes=2).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        return
    
    model.eval()
    
    # Test with dummy input
    with torch.no_grad():
        test_input = torch.randn(1, 3, 640, 640).to(device)
        output = model(test_input)
        print(f"🧪 Test output shape: {[o.shape for o in output] if isinstance(output, list) else output.shape}")

if __name__ == "__main__":
    # Create directory structure
    os.makedirs("home-fire-dataset/train/images", exist_ok=True)
    os.makedirs("home-fire-dataset/train/labels", exist_ok=True)
    os.makedirs("home-fire-dataset/val/images", exist_ok=True)
    os.makedirs("home-fire-dataset/val/labels", exist_ok=True)
    
    train_model(
        train_dir="home-fire-dataset/train",
        val_dir="home-fire-dataset/val", 
        epochs=5,  # Reduced for testing
        batch_size=2   # Reduced for memory efficiency
    )
    
    # Test the model after training
    test_model()
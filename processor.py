from transformers import AutoImageProcessor
from PIL import Image
import torch

class Processor:
    def __init__(self):
       
        self.preprocessor = AutoImageProcessor.from_pretrained("facebook/dinov3-convnext-base-pretrain-lvd1689m")
        self.extra_transforms = []

    def add_transform(self, transform_fn):
        self.extra_transforms.append(transform_fn)

    def __call__(self, image):
        for t in self.extra_transforms:
            image = t(image)

        return self.preprocessor(images=image, return_tensors="pt")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from scipy.spatial.distance import cosine

import torch
import torch.nn as nn
import torchvision.models as models

class VideoFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(VideoFeatureExtractor, self).__init__()
        
        # Load ResNet18. Pretrained weights give it a massive head start on understanding basic edges/colors
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Strip the final classification layer (we want embeddings, not labels)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        vec = self.features(x)
        
        # Flatten from [batch_size, 512, 1, 1] to [batch_size, 512]
        return vec.view(vec.size(0), -1)

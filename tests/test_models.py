"""Tests for model definitions."""

import torch
import torch.nn as nn
from torchvision import models


def test_resnet18_head_shape():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)

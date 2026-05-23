# File: python/arya_stark/models/resnet34.py
# ResNet-34 Implementation for arya-STARK FL System
# Ready to use - just copy into your models/ directory

"""
ResNet-34: 34-layer CNN with skip connections for Medical Imaging FL.

This module provides a ResNet-34 implementation that exposes the same
interface as MLPModel, allowing seamless integration into existing FL
infrastructure without changing client/server code.

Features:
    • 21.8M parameters (vs. 50K for MLP)
    • Realistic CNN with ReLU, BatchNorm, skip connections
    • Binary classification (healthy/cancer)
    • Image input: (B, 256, 256, 3) or (B, 3, 256, 256)
    • PyTorch backend for efficient gradient computation
    • FL-compatible interface (flat parameters, flat gradients)

Usage:
    >>> model = ResNet34Model(input_shape=(3, 256, 256), num_classes=2)
    >>> logits = model.forward(X_images)
    >>> loss = model.loss(X_images, y_labels)
    >>> grad = model.gradient_flat(X_images, y_labels)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock: Conv3x3 → BN → ReLU → Conv3x3 → BN → Add (skip).
    
    Formula: F(x) + x where F(x) is the residual function.
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        
        # First conv
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv (stride always 1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut (skip) connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save identity for skip connection
        identity = x
        
        # Residual function F(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add: F(x) + x
        out = out + identity
        out = F.relu(out)
        
        return out


class ResNet34(nn.Module):
    """
    ResNet-34: 34-layer residual network.
    
    Architecture:
        Layer 0: Conv 7×7, stride 2 → BN → ReLU → MaxPool
        Layer 1: 3 BasicBlocks @ 64 channels
        Layer 2: 4 BasicBlocks @ 128 channels (stride=2)
        Layer 3: 6 BasicBlocks @ 256 channels (stride=2)
        Layer 4: 3 BasicBlocks @ 512 channels (stride=2)
        GlobalAvgPool → FC 512 → num_classes
    
    Total: 1 + (3 + 4 + 6 + 3) * 2 = 34 conv layers
    
    Parameters:
        in_channels : int
            Number of input channels (3 for RGB, 1 for grayscale).
        num_classes : int
            Number of output classes (2 for binary classification).
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_classes
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks (layer1-4)
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Weight initialization
        self._init_weights()
    
    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int = 1):
        """Construct a residual layer with multiple blocks."""
        layers = []
        
        # First block may have stride > 1
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet-34.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, 3, height, width).
        
        Returns
        -------
        torch.Tensor
            Logits, shape (batch_size, num_classes).
        """
        # Initial conv + BN + ReLU + MaxPool
        x = F.relu(self.bn1(self.conv1(x)))  # → (B, 64, 128, 128)
        x = self.maxpool(x)                    # → (B, 64, 64, 64)
        
        # Residual blocks
        x = self.layer1(x)                     # → (B, 64, 64, 64)
        x = self.layer2(x)                     # → (B, 128, 32, 32)
        x = self.layer3(x)                     # → (B, 256, 16, 16)
        x = self.layer4(x)                     # → (B, 512, 8, 8)
        
        # Global average pooling + FC
        x = self.avgpool(x)                    # → (B, 512, 1, 1)
        x = torch.flatten(x, 1)                # → (B, 512)
        x = self.fc(x)                         # → (B, num_classes)
        
        return x


class ResNet34Model:
    """
    Wrapper exposing MLPModel-compatible interface for FL.
    
    This class wraps the PyTorch ResNet34 module and exposes the same
    methods as MLPModel (forward, loss, accuracy, gradient, etc.) so that
    existing FL client/server code requires no modifications.
    
    Parameters
    ----------
    input_shape : tuple
        (channels, height, width). Default: (3, 256, 256) for RGB images.
    num_classes : int
        Number of output classes. Default: 2 for binary classification.
    seed : int
        Random seed for reproducibility.
    device : str
        Device for computation: "cpu" or "cuda".
    """
    
    def __init__(
        self,
        input_shape: tuple = (3, 256, 256),
        num_classes: int = 2,
        seed: int = 0,
        device: str = "cpu",
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        
        # Create PyTorch model
        self.model = ResNet34(
            in_channels=input_shape[0],
            num_classes=num_classes
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # Count total parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    @classmethod
    def from_config(cls, config, seed: int = 0) -> ResNet34Model:
        """
        Create ResNet34Model from configuration object.
        
        Parameters
        ----------
        config : object
            Configuration with attributes:
                - name: "resnet34"
                - num_classes: int
                - input_shape: tuple (optional)
                - device: str (optional)
        seed : int
            Random seed.
        
        Returns
        -------
        ResNet34Model
        """
        if config.name != "resnet34":
            raise ValueError(f"Expected name='resnet34', got '{config.name}'")
        
        input_shape = getattr(config, "input_shape", (3, 256, 256))
        device = getattr(config, "device", "cpu")
        
        return cls(
            input_shape=input_shape,
            num_classes=config.num_classes,
            seed=seed,
            device=device,
        )
    
    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        """
        Convert NumPy array to PyTorch tensor with correct shape.
        
        Handles both (B, H, W, C) and (B, C, H, W) formats.
        """
        # Detect and fix channel ordering
        if X.shape[-1] in [1, 3] and len(X.shape) == 4:
            # Probably (B, H, W, C), convert to (B, C, H, W)
            X = np.transpose(X, (0, 3, 1, 2))
        
        # Convert to PyTorch tensor
        X_torch = torch.from_numpy(X).float().to(self.device)
        return X_torch
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet-34.
        
        Parameters
        ----------
        X : np.ndarray
            Input images. Shape can be:
            - (batch_size, height, width, channels) → auto-transpose to (B, C, H, W)
            - (batch_size, channels, height, width) → used as-is
        
        Returns
        -------
        np.ndarray
            Logits, shape (batch_size, num_classes).
        """
        X_torch = self._prepare_input(X)
        
        with torch.no_grad():
            logits = self.model(X_torch)
        
        return logits.cpu().numpy()
    
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Parameters
        ----------
        X : np.ndarray
            Images.
        y : np.ndarray
            Labels, shape (batch_size,), dtype int.
        
        Returns
        -------
        float
            Scalar loss value.
        """
        X_torch = self._prepare_input(X)
        y_torch = torch.from_numpy(y).long().to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_torch)
            loss = self.criterion(logits, y_torch)
        
        return float(loss.item())
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Parameters
        ----------
        X : np.ndarray
            Images.
        y : np.ndarray
            Labels.
        
        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        X_torch = self._prepare_input(X)
        y_torch = torch.from_numpy(y).long().to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_torch)
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == y_torch).float().mean()
        
        return float(accuracy.item())
    
    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradients via backpropagation.
        
        Parameters
        ----------
        X : np.ndarray
            Images.
        y : np.ndarray
            Labels.
        
        Returns
        -------
        np.ndarray
            Flattened gradient vector, shape (total_params,).
        """
        X_torch = self._prepare_input(X)
        y_torch = torch.from_numpy(y).long().to(self.device)
        X_torch.requires_grad = True
        
        # Forward pass
        logits = self.model(X_torch)
        loss = self.criterion(logits, y_torch)
        
        # Backward pass
        loss.backward()
        
        # Extract gradients as flat vector
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.cpu().numpy().ravel())
        
        return np.concatenate(grads).astype(np.float32)
    
    def gradient_flat(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient and return as flat vector (for FL compatibility).
        
        This is equivalent to gradient() but emphasized for FL use.
        """
        return self.gradient(X, y)
    
    def get_flat_params(self) -> np.ndarray:
        """
        Flatten all parameters into a single vector.
        
        Returns
        -------
        np.ndarray
            Flattened parameters, shape (total_params,), dtype float32.
        """
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().numpy().ravel())
        
        return np.concatenate(params).astype(np.float32)
    
    def set_flat_params(self, flat: np.ndarray) -> None:
        """
        Set parameters from a flattened vector.
        
        Parameters
        ----------
        flat : np.ndarray
            Flattened parameter vector, shape (total_params,).
        """
        flat_torch = torch.from_numpy(flat).float()
        
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            new_value = flat_torch[idx:idx+param_size].reshape(param.shape)
            param.data = new_value.to(self.device)
            idx += param_size
    
    def sgd_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        """
        Perform one SGD step: params -= lr * gradient.
        
        Parameters
        ----------
        X : np.ndarray
            Images.
        y : np.ndarray
            Labels.
        lr : float
            Learning rate.
        """
        grad = self.gradient(X, y)
        current = self.get_flat_params()
        self.set_flat_params(current - lr * grad)
    
    def apply_update(self, delta: np.ndarray, lr: float) -> None:
        """
        Apply a gradient delta: params -= lr * delta.
        
        This is used in FL when the server sends an aggregated gradient.
        
        Parameters
        ----------
        delta : np.ndarray
            Flattened gradient (or gradient delta).
        lr : float
            Learning rate.
        """
        current = self.get_flat_params()
        self.set_flat_params(current - lr * delta)


__all__ = [
    "ResNet34Model",
    "ResNet34",
    "BasicBlock",
]

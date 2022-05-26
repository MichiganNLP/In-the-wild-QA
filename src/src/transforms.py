"""From https://github.com/pytorch/vision/blob/993325d/references/video_classification/transforms.py"""
import random
from typing import Any

import torch
import torch.nn as nn
from overrides import overrides
from torchvision.transforms import InterpolationMode, RandomResizedCrop, functional as F

from src.utils.tensor_utils import pad


class ConvertBHWCtoBCHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)."""
    @overrides(check_signature=False)
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v.permute(0, 3, 1, 2)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)."""
    @overrides(check_signature=False)
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v.permute(1, 0, 2, 3)


# Added by us:


class ConvertBHWCtoCBHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (C, B, H, W)."""
    @overrides(check_signature=False)
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v.permute(3, 0, 1, 2)


class PadToMinFrames:
    def __init__(self, min_frames: int, frame_dim: int = 0, padding_value: Any = 0) -> None:
        self.min_frames = min_frames
        self.frame_dim = frame_dim
        self.padding_value = padding_value

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return pad(video, min_size=self.min_frames, dim=self.frame_dim, value=self.padding_value)


class MaxFrames:
    def __init__(self, max_frames: int, frame_dim: int = 0) -> None:
        self.max_frames = max_frames
        self.frame_dim = frame_dim

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video[(slice(None),) * self.frame_dim + (slice(None, self.max_frames),)]


class RandomResizedCropWithRandomInterpolation(RandomResizedCrop):
    @overrides
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        i, j, h, w = self.get_params(img, self.scale, self.ratio)  # noqa
        interpolation = random.choice([InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

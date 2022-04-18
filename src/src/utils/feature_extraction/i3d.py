# Adapted from https://github.com/piergiaj/pytorch-i3d/
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_3_t
from torch.nn.modules.utils import _triple


class SamePad3d(nn.Module):
    def __init__(self, kernel_size: _size_3_t = 1, stride: _size_3_t = 1) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)

    def compute_pad(self, dim: int, size: int) -> int:
        if size % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (size % self.stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.shape

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        return F.pad(x, [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b])


class MaxPool3dWithSamePadding(nn.MaxPool3d):
    def __init__(self, kernel_size: _size_3_t, stride: _size_3_t | None = None,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(kernel_size=kernel_size, stride=stride, return_indices=return_indices, ceil_mode=ceil_mode)
        self.same_padding = SamePad3d(kernel_size=kernel_size, stride=stride or kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.same_padding(x))


class Unit3D(nn.Module):
    def __init__(self, in_channels: int, output_channels: int, kernel_size: _size_3_t = 1, stride: _size_3_t = 1,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] | None = F.relu,
                 use_batch_norm: bool = True, use_bias: bool = False, name: str = "unit_3d",
                 eps: float = 1e-3) -> None:
        super().__init__()

        self._activation_fn = activation_fn
        self.name = name

        self.same_padding = SamePad3d(kernel_size=kernel_size, stride=stride)

        # We always want padding to be 0 here. We will dynamically pad based on input size in forward function.
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=output_channels, kernel_size=kernel_size,  # noqa
                                stride=stride, bias=use_bias)  # noqa

        self.bn = nn.BatchNorm3d(self._output_channels, eps=eps, momentum=0.01) if use_batch_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.same_padding(x)
        x = self.conv3d(x)
        if self.bn is not None:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: tuple[int, int, int, int, int, int], name: str) -> None:
        super().__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0],
                         name=f"{name}/Branch_0/Conv3d_0a_1x1")
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1],
                          name=f"{name}/Branch_1/Conv3d_0a_1x1")
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_size=3,
                          name=f"{name}/Branch_1/Conv3d_0b_3x3")
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3],
                          name=f"{name}/Branch_2/Conv3d_0a_1x1")
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_size=3,
                          name=f"{name}/Branch_2/Conv3d_0b_3x3")
        self.b3a = MaxPool3dWithSamePadding(kernel_size=3, stride=1)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5],
                          name=f"{name}/Branch_3/Conv3d_0b_1x1")
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat((b0, b1, b2, b3), dim=1)


class I3D(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        https://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(self, num_classes: int = 400, spatial_squeeze: bool = True, final_endpoint: str = "Logits",
                 name: str = "inception_i3d", in_channels: int = 3, dropout_keep_prob: float = 0.5) -> None:
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default "Logits").
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        super().__init__()
        self._spatial_squeeze = spatial_squeeze

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")

        self.avg_pool = None
        self.dropout = None
        self.logits = None

        self.end_points = {}
        self._add_end_points(num_classes=num_classes, final_endpoint=final_endpoint, name=name, in_channels=in_channels,
                             dropout_keep_prob=dropout_keep_prob)

        for k in self.end_points:
            self.add_module(k, self.end_points[k])

    def _add_end_points(self, num_classes: int = 400, final_endpoint: str = "Logits", name: str = "inception_i3d",
                        in_channels: int = 3, dropout_keep_prob: float = 0.5) -> None:
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_size=7, stride=2,
                                            name=f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dWithSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        if final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, name=f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_size=3,
                                            name=f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dWithSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        if final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(192, (64, 96, 128, 16, 32, 32), f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(256, (128, 128, 192, 32, 96, 64), f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dWithSamePadding(kernel_size=3, stride=2)
        if final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, (192, 96, 208, 16, 48, 64),
                                                     f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, (160, 112, 224, 24, 64, 64),
                                                     f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, (128, 128, 256, 24, 64, 64),
                                                     f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, (112, 144, 288, 32, 64, 64),
                                                     f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, (256, 160, 320, 32, 128, 128),
                                                     f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dWithSamePadding(kernel_size=2)
        if final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, (256, 160, 320, 32, 128, 128),
                                                     f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, (384, 192, 384, 48, 128, 128),
                                                     f"{name}{end_point}")
        if final_endpoint == end_point:
            return

        # end_point = "Logits"
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=num_classes, activation_fn=None,
                             use_batch_norm=False, use_bias=True, name="logits")

    def forward(self, x: torch.Tensor, extract_features: bool = False) -> torch.Tensor:
        for k in self.VALID_ENDPOINTS:
            if k in self.end_points:
                x = self._modules[k](x)  # use _modules to work with data-parallel

        if self.avg_pool is not None:
            x = self.avg_pool(x)

        if extract_features:
            return x

        logits = self.logits(self.dropout(x))

        if self._spatial_squeeze:
            logits = logits.squeeze(3).squeeze(3)

        return logits

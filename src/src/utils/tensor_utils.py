from typing import Any

import torch
import torch.nn.functional as F


def pad(t: torch.Tensor, min_size: int, dim: int = 1, value: Any = 0) -> torch.Tensor:
    """Pads the dim `dim` in `t` with the value `value` so the size is at least `min_size`."""
    if dim < 0:
        dim += len(t.shape)

    if (count := t.shape[dim]) < min_size:
        # `pad` keyword arg goes from the last dim to the first one in pairs, where the first value of the pair is
        # for left padding and the other one for right padding.
        return F.pad(t, pad=(0, 0) * (len(t.shape) - 1 - dim) + (0, min_size - count), value=value)
    else:
        return t

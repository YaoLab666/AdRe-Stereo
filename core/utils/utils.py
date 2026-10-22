"""Tensor utilities used by AdRe-Stereo inference and geometry lookup."""

import torch
import torch.nn.functional as F


class InputPadder:
    """Pad and unpad images to a requested spatial divisor."""

    def __init__(self, dims, divis_by=8):
        self.height, self.width = dims[-2:]
        pad_height = (-self.height) % divis_by
        pad_width = (-self.width) % divis_by
        self._pad = [
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ]

    def pad(self, *inputs):
        if not all(tensor.ndim == 4 for tensor in inputs):
            raise ValueError("InputPadder expects four-dimensional image tensors")
        return [F.pad(tensor, self._pad, mode="replicate") for tensor in inputs]

    def unpad(self, tensor):
        if tensor.ndim != 4:
            raise ValueError("InputPadder expects a four-dimensional tensor")
        height, width = tensor.shape[-2:]
        top, bottom = self._pad[2], height - self._pad[3]
        left, right = self._pad[0], width - self._pad[1]
        return tensor[..., top:bottom, left:right]


def bilinear_sampler(image, coordinates):
    """Sample a one-dimensional stereo volume at pixel coordinates."""
    height, width = image.shape[-2:]
    x_grid, y_grid = coordinates.split([1, 1], dim=-1)
    x_grid = 2 * x_grid / max(width - 1, 1) - 1
    if torch.unique(y_grid).numel() != 1 or height != 1:
        raise ValueError("bilinear_sampler expects a one-dimensional stereo volume")
    grid = torch.cat([x_grid, y_grid], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", align_corners=True)

"""The three self-supervised terms in AdRe-Stereo's TTT objective."""

import torch
import torch.nn.functional as F


def warp_right_to_left(image_r, disp_l, return_valid_mask=False):
    """
    Warp right image to left view using left disparity.
    Uses bilinear grid sampling.
    
    Args:
        image_r: [B, 3, H, W] right image
        disp_l: [B, 1, H, W] left disparity
        
    Returns:
        warped: right tensor warped to left coordinates. If
            ``return_valid_mask`` is true, also returns a binary sampling mask.
    """
    B, C, H, W = image_r.shape
    
    # Create coordinate grids
    ys, xs = torch.meshgrid(
        torch.arange(H, device=image_r.device, dtype=image_r.dtype),
        torch.arange(W, device=image_r.device, dtype=image_r.dtype),
        indexing='ij'
    )
    xs = xs[None, None].expand(B, -1, -1, -1)  # [B, 1, H, W]
    ys = ys[None, None].expand(B, -1, -1, -1)  # [B, 1, H, W]
    
    # Apply disparity to x-coordinates (rightward shift)
    x_warped = xs - disp_l
    
    # Normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * x_warped / max(W - 1, 1) - 1.0
    y_norm = 2.0 * ys / max(H - 1, 1) - 1.0
    
    # Stack normalized coordinates. grid_sample expects [B, H, W, 2].
    grid = torch.stack([x_norm.squeeze(1), y_norm.squeeze(1)], dim=-1)
    
    # Grid sample with boundary handling
    warped = F.grid_sample(
        image_r, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    if not return_valid_mask:
        return warped

    valid = ((x_warped >= 0.0) & (x_warped <= float(W - 1))).to(warped.dtype)
    return warped, valid


def left_right_consistency_map(disp_l, disp_r):
    """Map right-view disparity to left coordinates and return error + validity."""
    if disp_l.shape != disp_r.shape:
        raise ValueError(
            "left and right disparities must have identical shapes, got "
            f"{tuple(disp_l.shape)} and {tuple(disp_r.shape)}"
        )
    disp_r_to_l, valid = warp_right_to_left(
        disp_r, disp_l, return_valid_mask=True
    )
    error = (disp_l - disp_r_to_l).abs() * valid
    return error, disp_r_to_l, valid


def soft_census_transform(x, window_size=3, eps=1e-3):
    """Differentiable Census-like descriptor based on center comparisons."""
    gray = x.mean(dim=1, keepdim=True)
    patches = F.unfold(gray, kernel_size=window_size, padding=window_size // 2)
    patches = patches.view(x.shape[0], window_size ** 2, x.shape[2], x.shape[3])
    center = gray
    differences = patches - center
    return differences / torch.sqrt(differences.pow(2) + eps ** 2)


def photometric_loss(image_l, image_r_warped, mask=None):
    """Robust RGB/Census-like reprojection loss with paper weights 0.85/0.15."""
    image_l = image_l.float() / 255.0
    image_r_warped = image_r_warped.float() / 255.0
    difference = image_l - image_r_warped
    rgb_loss = torch.sqrt(difference.pow(2) + 1e-6).mean(dim=1, keepdim=True)
    census_l = soft_census_transform(image_l, window_size=3)
    census_r = soft_census_transform(image_r_warped, window_size=3)
    census_loss = torch.sqrt(
        (census_l - census_r).pow(2) + 1e-6
    ).mean(dim=1, keepdim=True)
    loss_map = 0.85 * rgb_loss + 0.15 * census_loss
    
    # Aggregate with optional masking
    if mask is not None:
        return (loss_map * mask).sum() / (mask.sum() + 1e-8)
    return loss_map.mean()


def lr_consistency_loss(disp_l, disp_r_to_l=None, mask=None):
    """
    Left-right consistency loss.
    Encourages symmetric disparities between left and right estimates.
    
    Args:
        disp_l: [B, 1, H, W] left disparity
        disp_r_to_l: [B, 1, H, W] right disparity mapped to left coordinates
        mask: [B, 1, H, W] optional mask
        
    Returns:
        loss: scalar loss
    """
    if disp_r_to_l is None:
        return disp_l.new_tensor(0.0)
    
    loss_map = (disp_l - disp_r_to_l).abs()
    
    if mask is not None:
        return (loss_map * mask).sum() / (mask.sum() + 1e-8)
    return loss_map.mean()


def peak_sharpen_loss(prob, pseudo_disp, mask=None):
    """
    Peak sharpening loss for disparity probability distributions.
    Encourages concentrated probability at the pseudo-disparity peak.
    
    Args:
        prob: [B, D, H, W] probability distribution over disparities
        pseudo_disp: [B, 1, H, W] pseudo-ground-truth disparity (e.g., from previous frame)
        mask: [B, 1, H, W] optional mask
    Returns:
        loss: scalar loss
    """
    # Round pseudo_disp to valid disparity indices
    idx = pseudo_disp.round().long().clamp_(0, prob.shape[1] - 1)
    
    picked = torch.gather(prob, 1, idx.detach())
    
    # Cross-entropy like loss: maximize probability at peak
    loss_map = -torch.log(picked.clamp_min(1e-8))
    
    if mask is not None:
        return (loss_map * mask).sum() / (mask.sum() + 1e-8)
    return loss_map.mean()

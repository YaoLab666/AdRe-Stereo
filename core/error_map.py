"""Fixed matching-error map and high-error tile selection for sparse TTT."""

import torch
import torch.nn.functional as F


DEFAULT_HIGH_ERROR_THRESHOLD = 0.4


def disparity_entropy(prob, eps=1e-8):
    """
    Compute Shannon entropy of disparity probability distribution.
    
    Args:
        prob: [B, D, H, W] probability distribution over disparities
        eps: small value for numerical stability
        
    Returns:
        entropy: [B, 1, H, W] entropy map
    """
    # Clamp to avoid log(0)
    prob_safe = prob.clamp_min(eps)
    # H(p) = -sum(p * log(p))
    ent = -(prob_safe * torch.log(prob_safe)).sum(dim=1, keepdim=True)
    return ent


def normalized_disparity_entropy(prob, eps=1e-8):
    """Return entropy normalized to ``[0, 1]`` by the disparity count."""
    entropy = disparity_entropy(prob, eps=eps)
    max_entropy = torch.log(
        prob.new_tensor(float(prob.shape[1]), dtype=torch.float32)
    ).clamp_min(eps)
    return (entropy.float() / max_entropy).clamp_(0.0, 1.0)


def normalize_nonnegative_map(value, scale, valid_mask=None, eps=1e-8):
    """Normalize a non-negative signal with an explicit, reproducible scale."""
    if scale <= 0:
        raise ValueError("normalization scale must be positive")
    normalized = (value.float().clamp_min(0.0) / max(float(scale), eps)).clamp_(0.0, 1.0)
    if valid_mask is not None:
        normalized = normalized * valid_mask.to(dtype=normalized.dtype)
    return normalized


def build_matching_error_map(
    prob,
    delta_abs_map=None,
    lr_consistency_map=None,
    entropy_weight=1.0,
    lr_weight=1.0,
    delta_weight=1.0,
    lr_scale=1.0,
    delta_scale=1.0,
    lr_valid_mask=None,
):
    """Build the fixed AdRe-Stereo matching-error map (paper Eq. 22).

    The weighted sum is divided by the sum of active weights, so the returned
    score stays in ``[0, 1]``. Missing optional signals are excluded rather
    than silently replaced by zeros.
    """
    components = {}
    weighted = []
    active_weight = 0.0

    if entropy_weight > 0:
        components["entropy"] = normalized_disparity_entropy(prob)
        weighted.append(float(entropy_weight) * components["entropy"])
        active_weight += float(entropy_weight)

    if lr_weight > 0:
        if lr_consistency_map is None:
            raise ValueError("lr_consistency_map is required when lr_weight > 0")
        components["lr_consistency"] = normalize_nonnegative_map(
            lr_consistency_map, lr_scale, valid_mask=lr_valid_mask
        )
        weighted.append(float(lr_weight) * components["lr_consistency"])
        active_weight += float(lr_weight)

    if delta_weight > 0:
        if delta_abs_map is None:
            raise ValueError("delta_abs_map is required when delta_weight > 0")
        components["warmup_delta"] = normalize_nonnegative_map(delta_abs_map, delta_scale)
        weighted.append(float(delta_weight) * components["warmup_delta"])
        active_weight += float(delta_weight)

    if not weighted or active_weight <= 0:
        raise ValueError("at least one matching-error component must have a positive weight")

    score = sum(weighted) / active_weight
    return score.detach(), {name: value.detach() for name, value in components.items()}


def select_high_error_tiles(score_map, threshold, tile_size=16, return_tile_mask=False):
    """Select every tile whose mean matching error is strictly above ``tau_h``.

    Tiles are non-overlapping on the score-map grid.  Repeating the tile mask
    and cropping, instead of resizing it, preserves the exact tile boundaries
    when the final row or column is smaller than ``tile_size``.
    """
    if score_map.ndim != 4 or score_map.shape[1] != 1:
        raise ValueError("score_map must have shape [B, 1, H, W]")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if threshold is None:
        raise ValueError("high-error threshold (tau_h) must be configured")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("high-error threshold (tau_h) must be in [0, 1]")

    height, width = score_map.shape[-2:]
    kernel = (min(tile_size, height), min(tile_size, width))
    tile_means = F.avg_pool2d(
        score_map.float(), kernel_size=kernel, stride=kernel, ceil_mode=True
    )
    tile_mask = tile_means > threshold
    mask = tile_mask.to(score_map.dtype)
    mask = mask.repeat_interleave(kernel[0], dim=-2).repeat_interleave(kernel[1], dim=-1)
    mask = mask[..., :height, :width]
    if return_tile_mask:
        return mask, tile_mask, tile_means
    return mask


class HighErrorTileSelector:
    """Select recurrent-grid tiles using the fixed matching-error map."""
    
    def __init__(self, threshold, tile_size=16):
        """
        Args:
            threshold: fixed high-error-tile threshold ``tau_h``
            tile_size: size of spatial tiles for coherence
        """
        self.threshold = threshold
        self.tile_size = tile_size

    def __call__(self, prob, delta_abs_map=None, lr_consistency_map=None):
        """
        Compute a high-error mask from the available matching-error signals.
        
        Args:
            prob: [B, D, H, W] disparity probability
            delta_abs_map: [B, 1, H, W] absolute disparity change magnitude (optional)
            lr_consistency_map: [B, 1, H, W] left-right consistency error (optional)
            
        Returns:
            mask: [B, 1, H, W] high-error tile mask
            score: [B, 1, H, W] fixed matching-error score
        """
        # Normalized components keep the fixed threshold meaningful.
        components = [normalized_disparity_entropy(prob)]
        
        # Optional: add update magnitude signal
        if delta_abs_map is not None:
            components.append(normalize_nonnegative_map(delta_abs_map.detach(), 1.0))
        
        # Optional: add left-right consistency signal
        if lr_consistency_map is not None:
            components.append(normalize_nonnegative_map(lr_consistency_map.detach(), 1.0))

        score = sum(components) / len(components)
        
        mask = select_high_error_tiles(
            score, self.threshold, self.tile_size
        )
        
        return mask, score

    def from_score(self, score):
        """Select hard tiles from a precomputed score without changing it."""
        score = score.detach()
        return select_high_error_tiles(
            score, self.threshold, self.tile_size
        ), score

    def selection_stats(self, score):
        """Return exact selected-tile counts for per-pair diagnostics."""
        _, tile_mask, tile_means = select_high_error_tiles(
            score.detach(), self.threshold, self.tile_size,
            return_tile_mask=True,
        )
        selected = tile_mask.flatten(1).sum(dim=1)
        total = tile_mask.new_full(
            (tile_mask.shape[0],), tile_mask[0].numel(), dtype=torch.long
        )
        return {
            "selected_tiles": selected,
            "total_tiles": total,
            "hard_tile_fraction": selected.float() / total.clamp_min(1).float(),
            "tile_means": tile_means.detach(),
        }

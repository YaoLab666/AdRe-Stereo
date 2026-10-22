"""Error-Driven Sparse TTT context construction for AdRe-Stereo."""

import time

import torch

from core.error_map import (
    DEFAULT_HIGH_ERROR_THRESHOLD,
    build_matching_error_map,
)
from core.ttt_losses import left_right_consistency_map


def synchronize_for(tensor):
    """Synchronize only when timing CUDA work."""
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _timed(tensor, function, measure=True):
    if measure:
        synchronize_for(tensor)
    started = time.perf_counter()
    value = function()
    if measure:
        synchronize_for(tensor)
        elapsed = time.perf_counter() - started
    else:
        elapsed = 0.0
    return value, elapsed


def make_aligned_right_view_inputs(image1, image2):
    """Build inputs for predicting a right-view disparity with a left-view matcher.

    IGEV-Stereo searches the second image to the left of the first image and
    therefore represents non-negative left disparities.  Merely swapping the
    stereo pair reverses that search direction.  Flipping both swapped images
    restores the expected direction; the resulting disparity must be flipped
    back before left-right consistency is evaluated.
    """
    return image2.flip(-1), image1.flip(-1)


def restore_aligned_right_view_tensor(value):
    """Map a right-view prediction back to the original image coordinates."""
    if value is None:
        return None
    return value.flip(-1)


def validate_adaptation_args(args):
    """Validate the paper-defined sparse TTT configuration."""
    if not getattr(args, "ttt_reset_each_sample", True):
        raise ValueError("AdRe-Stereo requires per-pair TTGA reset")
    if getattr(args, "ttt_warmup_iters", 2) < 1:
        raise ValueError("--ttt_warmup_iters must be at least 1")
    high_error_threshold = getattr(
        args, "ttt_high_error_threshold", DEFAULT_HIGH_ERROR_THRESHOLD
    )
    if not 0.0 <= high_error_threshold <= 1.0:
        raise ValueError("--ttt_high_error_threshold must be in [0, 1]")
    if getattr(args, "ttt_tile_size", 16) < 1:
        raise ValueError("--ttt_tile_size must be positive")

    weights = (
        getattr(args, "ttt_score_entropy_weight", 1.0),
        getattr(args, "ttt_score_lr_weight", 1.0),
        getattr(args, "ttt_score_delta_weight", 1.0),
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("matching-error weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("at least one matching-error weight must be positive")
    if getattr(args, "ttt_lr_score_scale", 1.0) <= 0:
        raise ValueError("--ttt_lr_score_scale must be positive")
    if getattr(args, "ttt_delta_score_scale", 1.0) <= 0:
        raise ValueError("--ttt_delta_score_scale must be positive")


@torch.no_grad()
def prepare_adaptation_context(model, image1, image2, args, tile_selector, measure_timing=False):
    """Prepare the fixed error map, tile mask, and reusable left-view cache.

    The returned score is calculated exactly once.  Its tensors are detached
    and reused by every inner-loop update and, optionally, final early exit.
    Right-view inference is also performed once when needed.
    """
    validate_adaptation_args(args)
    warmup_iters = getattr(args, "ttt_warmup_iters", 2)
    timings = {}

    left_cache, timings["feature_gev"] = _timed(
        image1,
        lambda: model.forward_features_only(image1, image2),
        measure=measure_timing,
    )
    (warmup_disp, warmup_aux), timings["warmup_left"] = _timed(
        image1,
        lambda: model.forward_from_cache(
            left_cache,
            iters=warmup_iters,
            test_mode=True,
            return_aux=True,
        ),
        measure=measure_timing,
    )

    prob = warmup_aux["prob"].detach()
    init_lowres = warmup_aux["init_disp_lowres"].detach()
    final_lowres = warmup_aux["refine_aux"]["final_disp_lowres"].detach()
    delta_abs_map = (final_lowres - init_lowres).abs()

    score_lr_weight = getattr(args, "ttt_score_lr_weight", 1.0)
    use_lr_loss = (
        getattr(args, "ttt_bidirectional", False)
        and getattr(args, "ttt_lr_consistency_weight", 0.0) > 0
    )
    need_right = score_lr_weight > 0 or use_lr_loss
    right_disp = None
    right_lowres = None
    lr_error = None
    lr_valid = None
    visibility_full = None
    visibility_lowres = None

    if need_right:
        right_image1, right_image2 = make_aligned_right_view_inputs(image1, image2)
        right_cache, timings["right_feature_gev"] = _timed(
            image1,
            lambda: model.forward_features_only(right_image1, right_image2),
            measure=measure_timing,
        )
        (right_disp, right_aux), timings["warmup_right"] = _timed(
            image1,
            lambda: model.forward_from_cache(
                right_cache,
                iters=warmup_iters,
                test_mode=True,
                return_aux=True,
            ),
            measure=measure_timing,
        )
        right_disp = restore_aligned_right_view_tensor(right_disp.detach())
        right_lowres = restore_aligned_right_view_tensor(
            right_aux["refine_aux"]["final_disp_lowres"].detach()
        )
        lr_error, _, lr_valid = left_right_consistency_map(final_lowres, right_lowres)
        full_lr_error, _, full_lr_valid = left_right_consistency_map(
            warmup_disp.detach(), right_disp
        )
        visibility_threshold = getattr(args, "ttt_visibility_thresh", 1.0)
        downsample_factor = float(2 ** getattr(args, "n_downsample", 2))
        visibility_full = full_lr_valid * (full_lr_error <= visibility_threshold).to(full_lr_valid.dtype)
        visibility_lowres = lr_valid * (
            lr_error <= visibility_threshold / downsample_factor
        ).to(lr_valid.dtype)
    else:
        timings["right_feature_gev"] = 0.0
        timings["warmup_right"] = 0.0

    def make_score():
        score, components = build_matching_error_map(
            prob,
            delta_abs_map=delta_abs_map,
            lr_consistency_map=lr_error,
            entropy_weight=getattr(args, "ttt_score_entropy_weight", 1.0),
            lr_weight=score_lr_weight,
            delta_weight=getattr(args, "ttt_score_delta_weight", 1.0),
            lr_scale=getattr(args, "ttt_lr_score_scale", 1.0),
            delta_scale=getattr(args, "ttt_delta_score_scale", 1.0),
            lr_valid_mask=lr_valid,
        )
        mask, score = tile_selector.from_score(score)
        selection = tile_selector.selection_stats(score)
        return mask.detach(), score.detach(), components, selection

    (high_error_mask, matching_error, components, selection), timings["score_selection"] = _timed(
        image1, make_score, measure=measure_timing
    )

    if visibility_lowres is None:
        valid_supervision_lowres = high_error_mask
    else:
        valid_supervision_lowres = high_error_mask * visibility_lowres
    high_error_mask_full = torch.nn.functional.interpolate(
        high_error_mask.float(), size=warmup_disp.shape[-2:], mode="nearest"
    )
    if visibility_full is None:
        valid_supervision_full = high_error_mask_full
    else:
        valid_supervision_full = high_error_mask_full * visibility_full

    return {
        "left_cache": left_cache,
        "high_error_mask": high_error_mask,
        "matching_error": matching_error,
        "error_components": components,
        "high_error_threshold": float(args.ttt_high_error_threshold),
        "selected_tiles": selection["selected_tiles"].detach(),
        "total_tiles": selection["total_tiles"].detach(),
        "high_error_tile_fraction": selection["hard_tile_fraction"].detach(),
        "high_error_pixel_fraction": high_error_mask.float().mean(dim=(1, 2, 3)).detach(),
        "valid_supervision_lowres": valid_supervision_lowres.detach(),
        "valid_supervision_full": valid_supervision_full.detach(),
        "valid_supervision_fraction_lowres": valid_supervision_lowres.float().mean(
            dim=(1, 2, 3)
        ).detach(),
        "valid_supervision_fraction_full": valid_supervision_full.float().mean(
            dim=(1, 2, 3)
        ).detach(),
        "warmup_disp": warmup_disp.detach(),
        "warmup_init_lowres": init_lowres,
        "warmup_final_lowres": final_lowres,
        "right_disp": right_disp,
        "right_disp_lowres": right_lowres,
        "lr_valid_lowres": lr_valid,
        "visibility_full": visibility_full,
        "visibility_lowres": visibility_lowres,
        "timing": timings,
    }


def add_error_driven_ttt_args(parser):
    """Add the paper's fixed matching-error-map options."""
    parser.add_argument("--ttt_warmup_iters", type=int, default=2)
    parser.add_argument("--ttt_tile_size", type=int, default=16)
    parser.add_argument(
        "--ttt_high_error_threshold",
        type=float,
        default=DEFAULT_HIGH_ERROR_THRESHOLD,
        help="fixed high-error tile threshold tau_h (default: 0.4)",
    )
    parser.add_argument("--ttt_score_entropy_weight", type=float, default=1.0)
    parser.add_argument("--ttt_score_lr_weight", type=float, default=1.0)
    parser.add_argument("--ttt_score_delta_weight", type=float, default=1.0)
    parser.add_argument(
        "--ttt_lr_score_scale",
        type=float,
        default=1.0,
        help="LR-error clipping scale in recurrent low-resolution disparity units",
    )
    parser.add_argument(
        "--ttt_delta_score_scale",
        type=float,
        default=1.0,
        help="warm-up update clipping scale in recurrent low-resolution disparity units",
    )
    parser.add_argument(
        "--ttt_visibility_thresh",
        type=float,
        default=1.0,
        help="LR visibility threshold in full-resolution disparity pixels",
    )

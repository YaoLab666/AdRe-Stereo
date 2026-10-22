"""Per-pair Test-Time Geometry Adaptation for AdRe-Stereo."""

import logging
import time

import torch
import torch.nn.functional as F

from core.adapter_utils import (
    clone_ttga_state,
    get_ttga_parameters,
    restore_ttga_state,
)
from core.adaptive_refinement import (
    FixedIterationRefinement,
    PairLevelAdaptiveRefinement,
)
from core.error_driven_ttt import prepare_adaptation_context, synchronize_for
from core.error_map import DEFAULT_HIGH_ERROR_THRESHOLD, HighErrorTileSelector
from core.ttt_losses import (
    left_right_consistency_map,
    lr_consistency_loss,
    peak_sharpen_loss,
    photometric_loss,
    warp_right_to_left,
)


class PerPairTTGA:
    """Reset, adapt, and apply TTGA independently for every stereo pair."""

    UPDATE_STEPS = 2

    def __init__(self, model, args):
        if getattr(model, "ttga", None) is None:
            raise ValueError("PerPairTTGA requires a model constructed with TTGA enabled")

        self.model = model
        self.args = args
        threshold = getattr(
            args, "ttt_high_error_threshold", DEFAULT_HIGH_ERROR_THRESHOLD
        )
        self.tile_selector = HighErrorTileSelector(
            threshold=threshold,
            tile_size=getattr(args, "ttt_tile_size", 16),
        )

        final_refinement_iters = getattr(args, "valid_iters", 32)
        if getattr(args, "adaptive_refinement", False):
            self.refinement_controller = PairLevelAdaptiveRefinement(
                delta_thresh=getattr(args, "update_threshold", 0.02),
                pair_error_thresh=getattr(args, "pair_error_threshold", 0.25),
                min_iters=getattr(args, "min_refinement_iters", 10),
                max_iters=final_refinement_iters,
            )
        else:
            self.refinement_controller = FixedIterationRefinement(
                max_iters=final_refinement_iters
            )

        self.initial_ttga_state = clone_ttga_state(model)
        self.pair_context = None
        self.pair_index = 0
        self.pair_diagnostics = {}
        self.timing_stats = {}

    def _optimizer(self):
        parameters = get_ttga_parameters(self.model)
        if not parameters:
            raise RuntimeError("TTGA has no trainable parameters")
        return torch.optim.AdamW(
            parameters,
            lr=self.args.ttt_lr,
            weight_decay=getattr(self.args, "ttt_weight_decay", 0.0),
        )

    def adapt(self, image_left, image_right, return_timing=False):
        """Run the two-step Error-Driven Sparse TTT phase (paper Sec. 3.3-3.5)."""
        self.timing_stats = {}
        self.pair_context = None
        self.pair_index += 1
        restore_ttga_state(self.model, self.initial_ttga_state)
        self.model.eval()

        requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith("ttga.")

        try:
            self.pair_context = prepare_adaptation_context(
                self.model,
                image_left,
                image_right,
                self.args,
                self.tile_selector,
                measure_timing=return_timing,
            )
            if return_timing:
                self.timing_stats.update(self.pair_context["timing"])

            valid_low = int(
                self.pair_context["valid_supervision_lowres"].count_nonzero().item()
            )
            valid_full = int(
                self.pair_context["valid_supervision_full"].count_nonzero().item()
            )
            self.pair_diagnostics = {
                "pair_index": self.pair_index,
                "high_error_threshold": float(
                    self.pair_context["high_error_threshold"]
                ),
                "selected_tiles": int(
                    self.pair_context["selected_tiles"].sum().item()
                ),
                "total_tiles": int(self.pair_context["total_tiles"].sum().item()),
                "high_error_tile_fraction": float(
                    self.pair_context["high_error_tile_fraction"].mean().item()
                ),
                "valid_supervision_fraction_lowres": float(
                    self.pair_context["valid_supervision_fraction_lowres"].mean().item()
                ),
                "valid_supervision_fraction_full": float(
                    self.pair_context["valid_supervision_fraction_full"].mean().item()
                ),
                "ttga_updates": 0,
                "skipped_no_valid_supervision": valid_low + valid_full == 0,
            }
            if self.pair_diagnostics["skipped_no_valid_supervision"]:
                return dict(self.pair_diagnostics)

            optimizer = self._optimizer()
            high_error_mask = self.pair_context["high_error_mask"]
            last_losses = {}
            step_times = []

            for _ in range(self.UPDATE_STEPS):
                if return_timing:
                    synchronize_for(image_left)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)

                with torch.enable_grad():
                    disparity, auxiliary = self.model.forward_from_cache(
                        self.pair_context["left_cache"],
                        iters=self.args.ttt_inner_iters,
                        test_mode=True,
                        return_aux=True,
                        high_error_mask=high_error_mask,
                    )
                    losses = self._test_time_objective(
                        image_left,
                        image_right,
                        disparity,
                        auxiliary["prob"],
                        high_error_mask,
                    )
                    total_loss = sum(losses.values())
                    total_loss.backward()
                    optimizer.step()

                if return_timing:
                    synchronize_for(image_left)
                    step_times.append(time.perf_counter() - started)
                last_losses = {
                    name: float(value.detach().item()) for name, value in losses.items()
                }
                self.pair_diagnostics["ttga_updates"] += 1

            if return_timing:
                self.timing_stats["ttga_updates"] = step_times
                self.timing_stats["ttga_update_total"] = sum(step_times)
            self.pair_diagnostics["losses"] = last_losses
            return dict(self.pair_diagnostics)
        finally:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = requires_grad[name]

    def _test_time_objective(
        self, image_left, image_right, disparity, probability, high_error_mask
    ):
        """Photometric, left-right, and peak terms from paper Eq. 11."""
        mask_full = F.interpolate(
            high_error_mask.float(), size=disparity.shape[-2:], mode="nearest"
        )
        visibility_full = self.pair_context["visibility_full"]

        warped_right, warp_valid = warp_right_to_left(
            image_right, disparity, return_valid_mask=True
        )
        photo_mask = warp_valid * mask_full
        if visibility_full is not None:
            photo_mask = photo_mask * visibility_full

        losses = {
            "photo": self.args.ttt_photo_weight
            * photometric_loss(
                image_left,
                warped_right,
                mask=photo_mask,
            )
        }

        if self.args.ttt_lr_consistency_weight > 0:
            right_disparity = self.pair_context["right_disp"]
            _, right_to_left, lr_valid = left_right_consistency_map(
                disparity, right_disparity
            )
            lr_mask = lr_valid * mask_full
            if visibility_full is not None:
                lr_mask = lr_mask * visibility_full
            losses["left_right"] = self.args.ttt_lr_consistency_weight * (
                lr_consistency_loss(disparity, right_to_left, mask=lr_mask)
            )

        if self.args.ttt_peak_weight > 0:
            peak_mask = high_error_mask
            visibility_lowres = self.pair_context["visibility_lowres"]
            if visibility_lowres is not None:
                peak_mask = peak_mask * visibility_lowres
            pseudo_disparity = (probability * torch.arange(
                probability.shape[1],
                device=probability.device,
                dtype=probability.dtype,
            ).view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            losses["peak"] = self.args.ttt_peak_weight * peak_sharpen_loss(
                probability,
                pseudo_disparity,
                mask=peak_mask,
            )

        return losses

    @torch.no_grad()
    def infer(self, image_left, image_right, iterations=None, return_timing=False):
        """Apply the adapted TTGA densely, then perform recurrent refinement."""
        if self.pair_context is None:
            raise RuntimeError("adapt() must be called before infer()")
        if return_timing:
            synchronize_for(image_left)
        started = time.perf_counter()

        if iterations is None:
            iterations = getattr(self.args, "valid_iters", 32)
        if iterations < 1:
            raise ValueError("final refinement iterations must be positive")

        controller = (
            self.refinement_controller
            if getattr(self.args, "adaptive_refinement", False)
            else None
        )
        disparity, auxiliary = self.model.forward_from_cache(
            self.pair_context["left_cache"],
            iters=iterations,
            test_mode=True,
            return_aux=True,
            adaptive_cfg=controller,
            matching_error_map_override=self.pair_context["matching_error"],
        )

        if return_timing:
            synchronize_for(image_left)
            self.timing_stats["final_refinement"] = time.perf_counter() - started

        refine = auxiliary["refine_aux"]
        iterations = refine.get("num_iters")
        if torch.is_tensor(iterations):
            iterations = int(iterations.item())
        final_delta = None
        if refine.get("iter_stats"):
            final_delta = refine["iter_stats"][-1].get("delta_abs_mean")
            if torch.is_tensor(final_delta):
                final_delta = float(final_delta.item())
        pair_error = float(
            self.pair_context["matching_error"].float().mean().item()
        )
        self.pair_diagnostics.update(
            {
                "final_refinement_iters": iterations,
                "final_update_magnitude": final_delta,
                "pair_matching_error": pair_error,
                "early_stopped": bool(refine.get("early_stopped", False)),
            }
        )
        auxiliary["matching_error_map"] = self.pair_context["matching_error"]
        auxiliary["high_error_mask"] = self.pair_context["high_error_mask"]
        auxiliary["adre_diagnostics"] = dict(self.pair_diagnostics)
        logging.debug("AdRe-Stereo pair diagnostics: %s", self.pair_diagnostics)
        # The adapted parameters belong only to this stereo pair. Restore the
        # supervised initialization immediately after producing its prediction.
        restore_ttga_state(self.model, self.initial_ttga_state)
        self.pair_context = None
        return disparity, auxiliary

    def get_timing_stats(self):
        return dict(self.timing_stats)

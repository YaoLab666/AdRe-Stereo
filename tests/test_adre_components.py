import argparse
import unittest
from types import SimpleNamespace

import torch

from core.adaptation import PerPairTTGA
from core.adaptive_refinement import PairLevelAdaptiveRefinement
from core.checkpoint import _canonicalize_state_keys
from core.cli import add_model_arguments, add_test_time_adaptation_arguments
from core.error_driven_ttt import (
    make_aligned_right_view_inputs,
    restore_aligned_right_view_tensor,
)
from core.error_map import (
    HighErrorTileSelector,
    build_matching_error_map,
)
from core.ttga import TestTimeGeometryAdapter as TTGA
from core.ttt_losses import peak_sharpen_loss
from evaluate_adre_stereo import _inference
from train_adre_stereo import parse_training_args


class CommandLineTests(unittest.TestCase):
    def test_training_entrypoint_embeds_paper_defaults(self):
        args = parse_training_args([])
        expected = {
            "ttga_rank": 8,
            "ttga_dropout": 0.0,
            "ttga_spatial_branch": True,
            "ttga_disparity_gate": True,
            "batch_size": 8,
            "image_size": [320, 736],
            "num_steps": 300000,
            "lr": 2e-4,
            "wdecay": 1e-5,
            "train_iters": 22,
            "valid_iters": 32,
            "seed": 666,
            "mixed_precision": True,
            "precision_dtype": "float16",
        }
        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value)
        self.assertIsNone(args.restore_ckpt)

    def test_kitti_training_profile_is_automatic(self):
        args = parse_training_args([
            "--restore_ckpt", "sceneflow_best.pth",
            "--train_datasets", "kitti",
        ])
        self.assertEqual(args.num_steps, 50000)
        self.assertEqual(args.train_iters, 12)

    def test_training_checkpoint_policy(self):
        with self.assertRaisesRegex(ValueError, "random initialization"):
            parse_training_args(["--restore_ckpt", "source.pth"])
        with self.assertRaisesRegex(ValueError, "KITTI fine-tuning requires"):
            parse_training_args(["--train_datasets", "kitti"])

    def test_paper_defaults(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        add_test_time_adaptation_arguments(parser)
        args = parser.parse_args([])
        self.assertTrue(args.mixed_precision)
        self.assertEqual(args.ttga_rank, 8)
        self.assertTrue(args.ttga_spatial_branch)
        self.assertTrue(args.ttga_disparity_gate)
        self.assertEqual(args.ttt_photo_weight, 1.0)
        self.assertEqual(args.ttt_lr_consistency_weight, 0.5)
        self.assertEqual(args.ttt_peak_weight, 0.25)
        self.assertEqual(args.ttt_high_error_threshold, 0.4)
        self.assertEqual(PerPairTTGA.UPDATE_STEPS, 2)

        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--enable_ttga", option_strings)
        self.assertNotIn("--ttt_steps", option_strings)

    def test_ttt_inference_uses_requested_refinement_iterations(self):
        class RecordingEngine:
            def __init__(self):
                self.iterations = None

            def adapt(self, image_left, image_right, return_timing=False):
                return None

            def infer(
                self, image_left, image_right, iterations=None,
                return_timing=False,
            ):
                self.iterations = iterations
                return image_left[:, :1], {"adre_diagnostics": {}}

        engine = RecordingEngine()
        image = torch.zeros(1, 3, 4, 4)
        _inference(None, engine, image, image, iterations=16)
        self.assertEqual(engine.iterations, 16)


class TTGATests(unittest.TestCase):
    def test_three_branch_shape_and_gradient(self):
        adapter = TTGA(
            geo_channels=8,
            feat_channels=96,
            adapter_rank=8,
            use_spatial_branch=True,
            use_disp_branch=True,
        )
        geometry = torch.randn(2, 8, 6, 5, 7)
        image_features = torch.randn(2, 96, 5, 7)
        output, auxiliary = adapter(
            geometry, image_features, return_aux=True
        )
        self.assertEqual(output.shape, geometry.shape)
        self.assertIn("disp_gate_mean", auxiliary)
        output.mean().backward()
        self.assertIsNotNone(adapter.disparity_gate.net[1].bias.grad)


class ErrorDrivenSparseTTTTests(unittest.TestCase):
    def test_matching_error_map_is_normalized(self):
        probability = torch.full((1, 4, 4, 4), 0.25)
        delta = torch.full((1, 1, 4, 4), 0.5)
        left_right = torch.ones((1, 1, 4, 4))
        score, components = build_matching_error_map(
            probability,
            delta_abs_map=delta,
            lr_consistency_map=left_right,
        )
        self.assertTrue(torch.all((score >= 0) & (score <= 1)))
        self.assertEqual(set(components), {
            "entropy", "lr_consistency", "warmup_delta"
        })

    def test_tile_threshold_is_strict(self):
        selector = HighErrorTileSelector(threshold=0.4, tile_size=2)
        score = torch.tensor([[[[0.4, 0.4, 0.8, 0.8],
                                [0.4, 0.4, 0.8, 0.8]]]])
        mask, _ = selector.from_score(score)
        self.assertEqual(mask[..., :2].sum().item(), 0)
        self.assertEqual(mask[..., 2:].sum().item(), 4)

    def test_flip_swap_restores_coordinates(self):
        left = torch.arange(12).view(1, 1, 3, 4).float()
        right = left + 10
        flipped_right, flipped_left = make_aligned_right_view_inputs(left, right)
        self.assertTrue(torch.equal(flipped_right, right.flip(-1)))
        self.assertTrue(torch.equal(flipped_left, left.flip(-1)))
        self.assertTrue(
            torch.equal(restore_aligned_right_view_tensor(flipped_right), right)
        )


class ObjectiveAndRefinementTests(unittest.TestCase):
    def test_peak_term_backpropagates_to_distribution(self):
        logits = torch.randn(1, 5, 2, 3, requires_grad=True)
        probability = logits.softmax(dim=1)
        expectation = (
            probability * torch.arange(5).view(1, 5, 1, 1)
        ).sum(dim=1, keepdim=True)
        loss = peak_sharpen_loss(probability, expectation)
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_pair_level_early_exit_requires_both_conditions(self):
        controller = PairLevelAdaptiveRefinement(
            delta_thresh=0.02,
            pair_error_thresh=0.25,
            min_iters=10,
            max_iters=32,
        )
        self.assertFalse(controller.should_stop(8, 0.01, 0.1))
        self.assertFalse(controller.should_stop(9, 0.03, 0.1))
        self.assertFalse(controller.should_stop(9, 0.01, 0.3))
        self.assertTrue(controller.should_stop(9, 0.01, 0.1))

    def test_adapted_state_is_discarded_after_prediction(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ttga = torch.nn.Linear(1, 1, bias=False)

            def forward_from_cache(self, cache, **kwargs):
                disparity = torch.ones(1, 1, 2, 2)
                auxiliary = {
                    "refine_aux": {
                        "num_iters": torch.tensor(1),
                        "iter_stats": [],
                        "early_stopped": False,
                    }
                }
                return disparity, auxiliary

        model = DummyModel()
        engine = PerPairTTGA(
            model,
            SimpleNamespace(adaptive_refinement=False, valid_iters=1),
        )
        initial = model.ttga.weight.detach().clone()
        with torch.no_grad():
            model.ttga.weight.add_(1.0)
        engine.pair_context = {
            "left_cache": {},
            "matching_error": torch.zeros(1, 1, 2, 2),
            "high_error_mask": torch.zeros(1, 1, 2, 2),
        }
        engine.infer(torch.zeros(1), torch.zeros(1), iterations=1)
        self.assertTrue(torch.equal(model.ttga.weight, initial))
        self.assertIsNone(engine.pair_context)


class CheckpointTests(unittest.TestCase):
    def test_source_checkpoint_keys_migrate_to_paper_ttga_names(self):
        tensor = torch.ones(1)
        migrated = _canonicalize_state_keys({
            "module.geo_ttt_adapter.lowrank.up.weight": tensor,
            "module.geo_ttt_adapter.alpha": tensor,
            "feature.weight": tensor,
        })
        self.assertIn("module.ttga.low_rank_residual.up.weight", migrated)
        self.assertIn("module.ttga.a_lr", migrated)
        self.assertIn("feature.weight", migrated)


if __name__ == "__main__":
    unittest.main()

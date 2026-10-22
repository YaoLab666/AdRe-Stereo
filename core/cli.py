"""Shared command-line options for AdRe-Stereo entry points."""

from core.error_driven_ttt import add_error_driven_ttt_args


def add_model_arguments(parser):
    architecture = parser.add_argument_group("model")
    architecture.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    architecture.add_argument("--corr_levels", type=int, default=2)
    architecture.add_argument("--corr_radius", type=int, default=4)
    architecture.add_argument("--n_downsample", type=int, default=2)
    architecture.add_argument("--n_gru_layers", type=int, default=3)
    architecture.add_argument("--max_disp", type=int, default=192)
    architecture.add_argument("--mixed_precision", action="store_true")
    architecture.add_argument(
        "--no_mixed_precision", dest="mixed_precision", action="store_false"
    )
    architecture.add_argument(
        "--precision_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )

    ttga = parser.add_argument_group("Test-Time Geometry Adapter (TTGA)")
    ttga.add_argument(
        "--ttga_rank", type=int, default=8,
    )
    ttga.add_argument(
        "--ttga_dropout", type=float, default=0.0,
    )
    ttga.add_argument(
        "--ttga_spatial_branch", action="store_true",
    )
    ttga.add_argument(
        "--no_ttga_spatial_branch", dest="ttga_spatial_branch",
        action="store_false",
    )
    ttga.add_argument(
        "--ttga_disparity_gate", action="store_true",
    )
    ttga.add_argument(
        "--no_ttga_disparity_gate", dest="ttga_disparity_gate",
        action="store_false",
    )
    parser.set_defaults(
        mixed_precision=True,
        ttga_spatial_branch=True,
        ttga_disparity_gate=True,
    )


def add_dataset_arguments(parser):
    datasets = parser.add_argument_group("dataset roots")
    datasets.add_argument("--sceneflow_root", default=None)
    datasets.add_argument("--kitti_root", default=None)
    datasets.add_argument("--middlebury_root", default=None)
    datasets.add_argument("--eth3d_root", default=None)


def add_test_time_adaptation_arguments(parser):
    ttt = parser.add_argument_group("per-pair test-time adaptation")
    ttt.add_argument("--ttt_lr", type=float, default=1e-4)
    ttt.add_argument("--ttt_weight_decay", type=float, default=0.0)
    ttt.add_argument("--ttt_inner_iters", type=int, default=2)
    ttt.add_argument("--ttt_photo_weight", type=float, default=1.0)
    ttt.add_argument("--ttt_lr_consistency_weight", type=float, default=0.5)
    ttt.add_argument("--ttt_peak_weight", type=float, default=0.25)
    ttt.add_argument(
        "--adaptive_refinement", action="store_true",
    )
    ttt.add_argument(
        "--min_refinement_iters", type=int, default=10,
    )
    ttt.add_argument(
        "--update_threshold", type=float, default=0.02,
    )
    ttt.add_argument(
        "--pair_error_threshold", type=float, default=0.25,
    )
    parser.set_defaults(ttt_reset_each_sample=True)
    add_error_driven_ttt_args(parser)

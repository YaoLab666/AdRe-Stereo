"""Offline supervised joint training of the IGEV backbone and TTGA."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from core.checkpoint import checkpoint_payload, load_checkpoint
from core.cli import add_dataset_arguments, add_model_arguments


PAPER_TRAINING_DEFAULTS = {
    "ttga_rank": 8,
    "ttga_dropout": 0.0,
    "ttga_spatial_branch": True,
    "ttga_disparity_gate": True,
    "batch_size": 8,
    "image_size": [320, 736],
    "lr": 2e-4,
    "wdecay": 1e-5,
    "valid_iters": 32,
    "seed": 666,
    "mixed_precision": True,
    "precision_dtype": "float16",
}

DATASET_TRAINING_PROFILES = {
    "sceneflow": {"num_steps": 300000, "train_iters": 22},
    "kitti": {"num_steps": 50000, "train_iters": 12},
}


def supervised_stereo_loss(
    disparity_predictions,
    initial_prediction,
    ground_truth,
    valid,
    loss_gamma=0.9,
    max_disp=192,
):
    """Initial Smooth-L1 plus weighted recurrent L1 loss (paper Eq. 3)."""
    prediction_count = len(disparity_predictions)
    if prediction_count < 1:
        raise ValueError("at least one recurrent prediction is required")
    magnitude = torch.sum(ground_truth ** 2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (magnitude < max_disp)).unsqueeze(1)

    loss = F.smooth_l1_loss(
        initial_prediction[valid], ground_truth[valid], reduction="mean"
    )
    exponent_scale = 15 / max(prediction_count - 1, 1)
    adjusted_gamma = loss_gamma ** exponent_scale
    for index, prediction in enumerate(disparity_predictions):
        weight = adjusted_gamma ** (prediction_count - index - 1)
        loss = loss + weight * (prediction - ground_truth).abs()[valid].mean()

    endpoint_error = (disparity_predictions[-1] - ground_truth).abs()[valid]
    metrics = {
        "epe": endpoint_error.mean().item(),
        "1px": (endpoint_error < 1).float().mean().item(),
        "3px": (endpoint_error < 3).float().mean().item(),
        "5px": (endpoint_error < 5).float().mean().item(),
    }
    return loss, metrics


def build_optimizer(args, model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


class TrainingLogger:
    summary_frequency = 100

    def __init__(self, logdir, scheduler):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "Training requires tensorboard; run `pip install tensorboard`."
            ) from error
        self.scheduler = scheduler
        self.total_steps = 0
        self.running = {}
        self.writer = SummaryWriter(log_dir=logdir)

    def push(self, metrics):
        self.total_steps += 1
        for name, value in metrics.items():
            self.running[name] = self.running.get(name, 0.0) + value
        if self.total_steps % self.summary_frequency == 0:
            averaged = {
                name: value / self.summary_frequency
                for name, value in self.running.items()
            }
            logging.info(
                "step=%d lr=%.8g metrics=%s",
                self.total_steps,
                self.scheduler.get_last_lr()[0],
                averaged,
            )
            for name, value in averaged.items():
                self.writer.add_scalar(f"train/{name}", value, self.total_steps)
            self.running.clear()

    def validation(self, metrics):
        for name, value in metrics.items():
            self.writer.add_scalar(f"validation/{name}", value, self.total_steps)

    def close(self):
        self.writer.close()


def _validation(model, args):
    from evaluate_adre_stereo import validate_kitti, validate_sceneflow

    if "sceneflow" in args.train_datasets:
        metrics = validate_sceneflow(model, args)
        return metrics, metrics["sceneflow_epe"]
    if any("kitti" in name for name in args.train_datasets):
        metrics_2012 = validate_kitti(model, args, "2012")
        metrics_2015 = validate_kitti(model, args, "2015")
        metrics = {**metrics_2012, **metrics_2015}
        selection_metric = (
            metrics["kitti2012_epe"] + metrics["kitti2015_epe"]
        ) / 2.0
        return metrics, selection_metric
    raise ValueError(f"no validation protocol for {args.train_datasets}")


def _save(path, model, step, selection_metric=None, args=None):
    torch.save(
        checkpoint_payload(
            model,
            step=step,
            selection_metric=selection_metric,
            config=vars(args) if args is not None else None,
        ),
        path,
    )


def train(args):
    import core.stereo_datasets as datasets
    from core.adre_stereo import AdReStereo

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / "run_config.json").open("w") as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)

    model = nn.DataParallel(AdReStereo(args))
    if args.restore_ckpt:
        logging.info("Warm-starting from %s", args.restore_ckpt)
        load_checkpoint(
            model,
            args.restore_ckpt,
            strict_ttga=True,
            require_low_rank_ttga=True,
        )
    else:
        logging.info("Starting Scene Flow training from random initialization")

    ttga_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".ttga." in name
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    logging.info(
        "Parameters: total=%d, TTGA=%d, backbone=%d",
        total_parameters,
        ttga_parameters,
        total_parameters - ttga_parameters,
    )

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = build_optimizer(args, model)
    logger = TrainingLogger(args.logdir, scheduler)
    model.to(args.device).train()
    model.module.freeze_bn()

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.mixed_precision and args.precision_dtype == "float16"
    )
    best_metric = float("inf")
    step = 0

    while step < args.num_steps:
        for _, *batch in tqdm(train_loader, desc=f"train {step}/{args.num_steps}"):
            if step >= args.num_steps:
                break
            image_left, image_right, ground_truth, valid = [
                tensor.to(args.device, non_blocking=True) for tensor in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            initial, predictions = model(
                image_left, image_right, iters=args.train_iters
            )
            loss, metrics = supervised_stereo_loss(
                predictions,
                initial,
                ground_truth,
                valid,
                max_disp=args.max_disp,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            logger.writer.add_scalar("train/loss", loss.item(), step)
            logger.writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            logger.push(metrics)

            if step % args.validation_frequency == 0 or step == args.num_steps:
                periodic = logdir / f"step_{step:06d}.pth"
                _save(periodic, model, step, args=args)
                model.eval()
                metrics, selection_metric = _validation(model.module, args)
                logger.validation(metrics)
                if selection_metric < best_metric:
                    best_metric = selection_metric
                    _save(
                        logdir / "best.pth",
                        model,
                        step,
                        selection_metric=selection_metric,
                        args=args,
                    )
                    logging.info(
                        "New best checkpoint: step=%d metric=%.6f",
                        step,
                        selection_metric,
                    )
                model.train()
                model.module.freeze_bn()

    _save(logdir / "final.pth", model, step, best_metric, args)
    logger.close()
    return str(logdir / "best.pth")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="adre_stereo")
    parser.add_argument("--logdir", default="checkpoints/adre_stereo")
    parser.add_argument("--restore_ckpt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_datasets", nargs="+", default=["sceneflow"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="optimizer steps; defaults to the selected dataset profile",
    )
    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 736])
    parser.add_argument(
        "--train_iters", type=int, default=None,
        help="recurrent training iterations; defaults to the dataset profile",
    )
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--validation_frequency", type=int, default=10000)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--img_gamma", type=float, nargs=2, default=None)
    parser.add_argument("--saturation_range", type=float, nargs=2, default=[0, 1.4])
    parser.add_argument("--do_flip", default=None, choices=["h", "v"])
    parser.add_argument("--spatial_scale", type=float, nargs=2, default=[-0.2, 0.4])
    parser.add_argument("--noyjitter", action="store_true")
    add_model_arguments(parser)
    add_dataset_arguments(parser)
    parser.set_defaults(**PAPER_TRAINING_DEFAULTS)
    return parser


def parse_training_args(argv=None):
    """Parse arguments and apply the dataset-specific training profile."""
    args = build_parser().parse_args(argv)
    if len(args.train_datasets) != 1:
        raise ValueError("AdRe-Stereo training accepts one dataset profile at a time")
    dataset = args.train_datasets[0]
    if dataset not in DATASET_TRAINING_PROFILES:
        raise ValueError(f"unsupported training dataset profile: {dataset}")
    for name, value in DATASET_TRAINING_PROFILES[dataset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if dataset == "sceneflow" and args.restore_ckpt is not None:
        raise ValueError(
            "Scene Flow training starts from random initialization; remove "
            "--restore_ckpt."
        )
    if dataset == "kitti" and args.restore_ckpt is None:
        raise ValueError(
            "KITTI fine-tuning requires --restore_ckpt pointing to a trained "
            "Scene Flow checkpoint."
        )
    return args


def main():
    args = parse_training_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    best_path = train(args)
    logging.info("Training complete; best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()

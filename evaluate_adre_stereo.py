"""Validation and zero-shot evaluation for AdRe-Stereo."""

import argparse
import logging
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from core.adaptation import PerPairTTGA
from core.checkpoint import load_checkpoint
from core.cli import (
    add_dataset_arguments,
    add_model_arguments,
    add_test_time_adaptation_arguments,
)
from core.utils.utils import InputPadder


def _inference(model, engine, image_left, image_right, iterations):
    if engine is None:
        with torch.no_grad():
            return model(
                image_left, image_right, iters=iterations, test_mode=True
            ), {}

    with torch.enable_grad():
        engine.adapt(image_left, image_right, return_timing=False)
    disparity, auxiliary = engine.infer(
        image_left, image_right, iterations=iterations, return_timing=False
    )
    return disparity, auxiliary.get("adre_diagnostics", {})


def _engine(model, args):
    # User-facing evaluation and submission parsers always provide the TTT
    # configuration. Training-time validation does not, and evaluates the
    # supervised initialization without mutating it between training epochs.
    return PerPairTTGA(model, args) if hasattr(args, "ttt_lr") else None


def _prepare_pair(image_left, image_right, device):
    image_left = image_left[None].to(device)
    image_right = image_right[None].to(device)
    padder = InputPadder(image_left.shape, divis_by=32)
    image_left, image_right = padder.pad(image_left, image_right)
    return image_left, image_right, padder


def validate_sceneflow(model, args):
    import core.stereo_datasets as datasets

    dataset = datasets.SceneFlowDatasets(
        root=args.sceneflow_root,
        dstype="frames_finalpass",
        things_test=True,
    )
    engine = _engine(model, args)
    error_sum = 0.0
    bad3_sum = 0
    valid_count = 0

    for index in tqdm(range(len(dataset)), desc="Scene Flow"):
        _, left, right, ground_truth, valid = dataset[index]
        left, right, padder = _prepare_pair(left, right, args.device)
        prediction, _ = _inference(model, engine, left, right, args.valid_iters)
        prediction = padder.unpad(prediction.float()).cpu().squeeze(0)
        mask = (valid >= 0.5) & (ground_truth[0].abs() < args.max_disp)
        errors = (prediction[0] - ground_truth[0]).abs()[mask]
        error_sum += errors.sum().item()
        bad3_sum += (errors > 3.0).sum().item()
        valid_count += errors.numel()

    metrics = {
        "sceneflow_epe": error_sum / valid_count,
        "sceneflow_bad3": 100.0 * bad3_sum / valid_count,
    }
    logging.info("Scene Flow: EPE %.6f, Bad-3 %.4f%%", *metrics.values())
    return metrics


def validate_kitti(model, args, year):
    import core.stereo_datasets as datasets

    dataset = datasets.KITTI(
        root=args.kitti_root, image_set="training", years=(year,)
    )
    engine = _engine(model, args)
    error_sum = 0.0
    bad3_sum = 0
    valid_count = 0
    image_epes = []
    runtimes = []

    for index in tqdm(range(len(dataset)), desc=f"KITTI {year}"):
        _, left, right, ground_truth, valid = dataset[index]
        left, right, padder = _prepare_pair(left, right, args.device)
        if left.is_cuda:
            torch.cuda.synchronize(left.device)
        started = time.perf_counter()
        prediction, _ = _inference(model, engine, left, right, args.valid_iters)
        if left.is_cuda:
            torch.cuda.synchronize(left.device)
        runtimes.append(time.perf_counter() - started)
        prediction = padder.unpad(prediction.float()).cpu().squeeze(0)
        mask = (valid >= 0.5) & (ground_truth[0].abs() < args.max_disp)
        errors = (prediction[0] - ground_truth[0]).abs()[mask]
        error_sum += errors.sum().item()
        bad3_sum += (errors > 3.0).sum().item()
        valid_count += errors.numel()
        image_epes.append(errors.mean().item())

    metrics = {
        f"kitti{year}_epe": float(np.mean(image_epes)),
        f"kitti{year}_bad3": 100.0 * bad3_sum / valid_count,
        f"kitti{year}_runtime_ms": 1000.0 * float(np.mean(runtimes)),
    }
    logging.info(
        "KITTI %s: EPE %.6f, Bad-3 %.4f%%, %.2f ms/pair",
        year, *metrics.values(),
    )
    return metrics


def validate_middlebury(model, args, split):
    import core.stereo_datasets as datasets

    dataset = datasets.Middlebury(root=args.middlebury_root, split=split)
    engine = _engine(model, args)
    error_sum = 0.0
    bad2_sum = 0
    valid_count = 0

    for index in tqdm(range(len(dataset)), desc=f"Middlebury {split}"):
        files, left, right, ground_truth, valid = dataset[index]
        left, right, padder = _prepare_pair(left, right, args.device)
        prediction, _ = _inference(model, engine, left, right, args.valid_iters)
        prediction = padder.unpad(prediction.float()).cpu().squeeze(0)
        non_occluded = torch.from_numpy(
            np.array(
                Image.open(files[0].replace("im0.png", "mask0nocc.png")).convert("L"),
                copy=True,
            )
        ) == 255
        mask = (
            (valid >= 0.5)
            & (ground_truth[0] < args.max_disp)
            & non_occluded
        )
        errors = (prediction[0] - ground_truth[0]).abs()[mask]
        error_sum += errors.sum().item()
        bad2_sum += (errors > 2.0).sum().item()
        valid_count += errors.numel()

    metrics = {
        f"middlebury_{split}_epe": error_sum / valid_count,
        f"middlebury_{split}_bad2": 100.0 * bad2_sum / valid_count,
    }
    logging.info("Middlebury %s: EPE %.6f, Bad-2 %.4f%%", split, *metrics.values())
    return metrics


def validate_eth3d(model, args):
    import core.stereo_datasets as datasets

    dataset = datasets.ETH3D(root=args.eth3d_root, split="training")
    engine = _engine(model, args)
    error_sum = 0.0
    bad1_sum = 0
    valid_count = 0

    for index in tqdm(range(len(dataset)), desc="ETH3D"):
        files, left, right, ground_truth, valid = dataset[index]
        left, right, padder = _prepare_pair(left, right, args.device)
        prediction, _ = _inference(model, engine, left, right, args.valid_iters)
        prediction = padder.unpad(prediction.float()).cpu().squeeze(0)
        non_occluded = torch.from_numpy(
            np.array(
                Image.open(files[2].replace("disp0GT.pfm", "mask0nocc.png")),
                copy=True,
            )
        ) == 255
        mask = (valid >= 0.5) & non_occluded
        errors = (prediction[0] - ground_truth[0]).abs()[mask]
        error_sum += errors.sum().item()
        bad1_sum += (errors > 1.0).sum().item()
        valid_count += errors.numel()

    metrics = {
        "eth3d_epe": error_sum / valid_count,
        "eth3d_bad1": 100.0 * bad1_sum / valid_count,
    }
    logging.info("ETH3D: EPE %.6f, Bad-1 %.4f%%", *metrics.values())
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_ckpt", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "sceneflow", "kitti2012", "kitti2015",
            "middlebury_H", "middlebury_Q", "eth3d",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid_iters", type=int, default=32)
    add_model_arguments(parser)
    add_dataset_arguments(parser)
    add_test_time_adaptation_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()
    from core.adre_stereo import AdReStereo

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    model = AdReStereo(args)
    load_checkpoint(model, args.restore_ckpt, strict_ttga=True)
    model.to(args.device).eval()

    if args.dataset == "sceneflow":
        validate_sceneflow(model, args)
    elif args.dataset.startswith("kitti"):
        validate_kitti(model, args, args.dataset[-4:])
    elif args.dataset.startswith("middlebury_"):
        validate_middlebury(model, args, args.dataset[-1])
    else:
        validate_eth3d(model, args)


if __name__ == "__main__":
    main()

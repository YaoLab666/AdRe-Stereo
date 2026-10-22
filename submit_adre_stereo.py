"""Generate KITTI 2012/2015 official-test disparity submissions."""

import argparse
import csv
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from core.checkpoint import load_checkpoint
from core.cli import (
    add_dataset_arguments,
    add_model_arguments,
    add_test_time_adaptation_arguments,
)
from evaluate_adre_stereo import _engine, _inference, _prepare_pair


def _dataset_root(explicit):
    return str(Path(explicit or os.environ.get("KITTI_ROOT", "datasets/KITTI")).expanduser())


def _kitti_year_root(root, year):
    candidates = [Path(root) / f"KITTI{year}", Path(root) / year, Path(root)]
    for candidate in candidates:
        if (candidate / "testing").is_dir():
            return str(candidate)
    return str(candidates[0])


def _load_image(path):
    image = np.asarray(Image.open(path), dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = torch.from_numpy(image[..., :3].copy()).permute(2, 0, 1).float()
    return image


def _test_pairs(kitti_root, year):
    year_root = Path(_kitti_year_root(kitti_root, year))
    if year == "2012":
        left = sorted((year_root / "testing" / "colored_0").glob("*_10.png"))
        right = sorted((year_root / "testing" / "colored_1").glob("*_10.png"))
    else:
        left = sorted((year_root / "testing" / "image_2").glob("*_10.png"))
        right = sorted((year_root / "testing" / "image_3").glob("*_10.png"))
    if not left or len(left) != len(right):
        raise FileNotFoundError(
            f"no complete KITTI {year} test pairs found below {year_root}"
        )
    return list(zip(left, right))


def generate_submission(model, args):
    year = args.dataset[-4:]
    root = _dataset_root(args.kitti_root)
    pairs = _test_pairs(root, year)
    output = Path(args.output_directory or f"submissions/KITTI{year}")
    output.mkdir(parents=True, exist_ok=True)
    engine = _engine(model, args)
    diagnostics = []

    for left_path, right_path in tqdm(pairs, desc=f"KITTI {year} submission"):
        left, right, padder = _prepare_pair(
            _load_image(left_path), _load_image(right_path), args.device
        )
        prediction, pair_diagnostics = _inference(
            model, engine, left, right, args.valid_iters
        )
        disparity = padder.unpad(prediction.float()).cpu().numpy().squeeze()
        encoded = np.rint(
            np.clip(disparity, 0.0, np.iinfo(np.uint16).max / 256.0) * 256.0
        ).astype(np.uint16)
        imageio.imwrite(output / left_path.name, encoded)
        diagnostics.append({"image": left_path.name, **pair_diagnostics})

    if diagnostics and any(len(row) > 1 for row in diagnostics):
        fieldnames = sorted({key for row in diagnostics for key in row})
        with (output / "adre_diagnostics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diagnostics)

    if args.zip:
        archive = output.with_suffix(".zip")
        with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
            for path in sorted(output.glob("*.png")):
                zip_file.write(path, arcname=path.name)
        print(f"Submission archive: {archive}")
    else:
        print(f"Submission directory: {output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_ckpt", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=["kitti2012", "kitti2015"]
    )
    parser.add_argument("--output_directory", default=None)
    parser.add_argument("--zip", dest="zip", action="store_true")
    parser.add_argument("--no_zip", dest="zip", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid_iters", type=int, default=32)
    add_model_arguments(parser)
    add_dataset_arguments(parser)
    add_test_time_adaptation_arguments(parser)
    parser.set_defaults(zip=True)
    return parser


def main():
    args = build_parser().parse_args()
    from core.adre_stereo import AdReStereo

    model = AdReStereo(args)
    load_checkpoint(model, args.restore_ckpt, strict_ttga=True)
    model.to(args.device).eval()
    generate_submission(model, args)


if __name__ == "__main__":
    main()

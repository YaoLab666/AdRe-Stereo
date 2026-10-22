"""Image and disparity readers required by AdRe-Stereo benchmarks."""

from pathlib import Path
import re

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def readPFM(path):
    """Read a grayscale or RGB Portable Float Map."""
    with open(path, "rb") as stream:
        header = stream.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise ValueError(f"not a PFM file: {path}")

        dimensions = re.match(rb"^(\d+)\s(\d+)\s$", stream.readline())
        if dimensions is None:
            raise ValueError(f"malformed PFM dimensions: {path}")
        width, height = map(int, dimensions.groups())

        scale = float(stream.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if color else 1
        data = np.fromfile(stream, endian + "f")
        expected = height * width * channels
        if data.size != expected:
            raise ValueError(
                f"PFM payload has {data.size} values; expected {expected}: {path}"
            )

    shape = (height, width, 3) if color else (height, width)
    return np.flipud(data.reshape(shape))


def readDispKITTI(path):
    disparity = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if disparity is None:
        raise FileNotFoundError(path)
    disparity = disparity.astype(np.float32) / 256.0
    return disparity, disparity > 0.0


def readDispMiddlebury(path):
    path = Path(path)
    if path.name != "disp0GT.pfm":
        raise ValueError(f"unexpected Middlebury disparity name: {path.name}")
    disparity = readPFM(path).astype(np.float32)
    if disparity.ndim != 2:
        raise ValueError(f"Middlebury disparity must be two-dimensional: {path}")
    mask_path = path.with_name("mask0nocc.png")
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    non_occluded = imageio.imread(mask_path) == 255
    return disparity, non_occluded


def read_gen(path):
    """Read a benchmark image or PFM disparity from ``path``."""
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpeg", ".ppm", ".jpg"}:
        return Image.open(path)
    if suffix == ".pfm":
        value = readPFM(path).astype(np.float32)
        return value if value.ndim == 2 else value[:, :, :-1]
    raise ValueError(f"unsupported AdRe-Stereo data file: {path}")

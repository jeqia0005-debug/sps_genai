"""Shared helpers for turning generated image tensors into PNG bytes."""

import io
import math

import torch
from PIL import Image
from torchvision.utils import make_grid


def tensor_to_png(images: torch.Tensor, upscale: int = 6) -> io.BytesIO:
    """Convert a batch of images in [0, 1] with shape (B, C, H, W) into a PNG buffer.

    A grid is built from the batch and the result is upscaled with nearest-neighbour
    interpolation so that small 32x32 samples are readable in a browser.
    """
    images = images.detach().cpu().clamp(0.0, 1.0)
    nrow = max(1, int(math.ceil(math.sqrt(images.size(0)))))
    grid = make_grid(images, nrow=nrow, padding=2)

    array = (grid * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    pil_image = Image.fromarray(array)

    if upscale > 1:
        pil_image = pil_image.resize(
            (pil_image.width * upscale, pil_image.height * upscale),
            resample=Image.NEAREST,
        )

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer
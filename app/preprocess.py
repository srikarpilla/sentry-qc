"""
app/preprocess.py
─────────────────
SINGLE SOURCE OF TRUTH for image preprocessing.

Both the API (app/model_service.py) and the evaluation script
(scripts/verify_model.py) import preprocess_image_bytes() from here so there
is no train/serve skew.

Pipeline (mirrors training exactly):
  raw bytes
    -> Pillow for EXIF rotation / alpha / mode conversion
    -> numpy float32 array  (H x W x 3, values 0-255)
    -> tf.image.resize (bilinear, 288x288)
    -> cast float32
    -> add batch dim  (1, 288, 288, 3)

CRITICAL: Do NOT divide by 255. EfficientNetB0 has an internal rescaling layer.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

if TYPE_CHECKING:
    pass

IMG_SIZE = (288, 288)   # (height, width)


def preprocess_image_bytes(
    image_bytes: bytes,
    use_tta: bool = False,
) -> "np.ndarray":
    """
    Convert raw image bytes into a float32 numpy array ready for the model.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG / PNG / BMP / WebP image.
    use_tta : bool
        If True, return a batch of 3 augmented views:
        [original, h-flipped, v-flipped], shape (3, 288, 288, 3).
        If False, return a single image batch, shape (1, 288, 288, 3).

    Returns
    -------
    np.ndarray
        float32 array, pixel values 0-255.
    """
    img = _load_and_normalise_pil(image_bytes)
    arr = _pil_to_tensor(img)                # (288, 288, 3) float32

    if use_tta:
        h_flip = tf.image.flip_left_right(arr).numpy()
        v_flip = tf.image.flip_up_down(arr).numpy()
        return np.stack([arr, h_flip, v_flip], axis=0)  # (3, 288, 288, 3)
    else:
        return np.expand_dims(arr, axis=0)              # (1, 288, 288, 3)


def _load_and_normalise_pil(image_bytes: bytes) -> Image.Image:
    """
    Open the image with Pillow, apply EXIF rotation, and convert to RGB.

    Handles:
    - EXIF-rotated JPEGs (ImageOps.exif_transpose)
    - RGBA / LA (alpha channel stripped)
    - Grayscale (L, 1)
    - CMYK / other modes
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                          # raises on corrupt data
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Re-open after verify() (verify() exhausts the file pointer)
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)        # honour EXIF orientation
    img = img.convert("RGB")                  # always 3-channel
    return img


def _pil_to_tensor(img: Image.Image) -> np.ndarray:
    """
    Resize to IMG_SIZE with bilinear interpolation and return float32 array.
    Values stay in 0-255 range.
    """
    arr = np.array(img, dtype=np.float32)                       # H x W x 3
    tensor = tf.constant(arr)                                   # keep dtype
    tensor = tf.image.resize(tensor, IMG_SIZE, method="bilinear")
    return tensor.numpy()                                       # (288, 288, 3)

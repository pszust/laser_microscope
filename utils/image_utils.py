import cv2
import numpy as np

import utils.consts as cst


def get_objective_mag(obj: str):
    mag = int(obj.replace("X", ""))
    return mag


def scale_mm2px(dist_mm: float, obj: str) -> int:
    px_in_mm = cst.SCALE_PX2MM_AT_1X * get_objective_mag(obj)
    return int(px_in_mm * dist_mm)


def scale_px2mm(dist_px: int, obj: str) -> float:
    px_in_mm = cst.SCALE_PX2MM_AT_1X * get_objective_mag(obj)
    return dist_px / px_in_mm


def add_image(base_image: np.ndarray, new_image: np.ndarray, coords: tuple[int, int]) -> np.ndarray:
    """
    Paste `new_image` onto `base_image` at top-left pixel `coords=(x, y)`.
    If the paste exceeds base bounds, the output canvas is expanded and
    the empty area is filled with black (0,0,0) for color or 0 for grayscale.

    - Accepts grayscale (HxW), 3-channel (HxWx3), and 4-channel (HxWx4) arrays.
    - If `new_image` has an alpha channel, it is used to alpha-composite onto the canvas.
    - Dtypes are preserved via numpy.result_type of both inputs.
    - `coords` may be negative to paste above/left of the base image's origin.

    Returns
    -------
    np.ndarray
        The combined image with shape large enough to contain both.
    """
    x, y = int(coords[0]), int(coords[1])

    base = np.asarray(base_image)
    new = np.asarray(new_image)

    # Common dtype
    out_dtype = np.result_type(base.dtype, new.dtype)

    def split_alpha(img):
        if img.ndim == 3 and img.shape[2] == 4:
            return img[..., :3], img[..., 3]
        return img, None

    # Separate color and alpha if present
    b_rgb, _ = split_alpha(base)
    n_rgb, n_a = split_alpha(new)

    # Decide output color plane count (1 or 3). If any is color, use 3.
    want_rgb = (
        3 if ((b_rgb.ndim == 3 and b_rgb.shape[2] == 3) or (n_rgb.ndim == 3 and n_rgb.shape[2] == 3)) else 1
    )

    def to_channels(img, ch):
        if img.ndim == 2 and ch == 3:
            return np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 3 and ch == 1:
            # Convert color → gray (simple luminance approximation)
            # Works for either RGB or BGR ordering; weights are generic.
            return (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(img.dtype)
        else:
            return img

    b_rgb = to_channels(b_rgb, want_rgb)
    n_rgb = to_channels(n_rgb, want_rgb)

    # Cast to common dtype
    b_rgb = b_rgb.astype(out_dtype, copy=False)
    n_rgb = n_rgb.astype(out_dtype, copy=False)

    Hb, Wb = b_rgb.shape[:2]
    Hn, Wn = n_rgb.shape[:2]

    # Compute bounding box of union (base at (0,0), new at (x,y))
    min_x = min(0, x)
    min_y = min(0, y)
    max_x = max(Wb, x + Wn)
    max_y = max(Hb, y + Hn)

    out_W = max_x - min_x
    out_H = max_y - min_y

    # Create black canvas
    if want_rgb == 1:
        canvas = np.zeros((out_H, out_W), dtype=out_dtype)
    else:
        canvas = np.zeros((out_H, out_W, 3), dtype=out_dtype)

    # Offsets for placing images into the canvas
    bx, by = -min_x, -min_y
    nx, ny = x - min_x, y - min_y

    # Paste base image
    if want_rgb == 3:
        canvas[by : by + Hb, bx : bx + Wb, :] = b_rgb
    else:
        canvas[by : by + Hb, bx : bx + Wb] = b_rgb

    # Paste new image (with optional alpha compositing)
    if n_a is not None:
        # Ensure alpha is float in [0,1]
        alpha = n_a.astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        if want_rgb == 3:
            alpha = alpha[..., None]

        y1, y2 = ny, ny + Hn
        x1, x2 = nx, nx + Wn

        roi = canvas[y1:y2, x1:x2]  # view
        roi_f = roi.astype(np.float32)
        new_f = n_rgb.astype(np.float32)

        blended = new_f * alpha + roi_f * (1.0 - alpha)
        canvas[y1:y2, x1:x2] = blended.astype(out_dtype)
    else:
        if want_rgb == 3:
            canvas[ny : ny + Hn, nx : nx + Wn, :] = n_rgb
        else:
            canvas[ny : ny + Hn, nx : nx + Wn] = n_rgb

    return canvas

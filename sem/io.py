"""Inventory, sha256, TIFF tag pixel-size parsing, grouping (SPEC §0, §3 upload)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tifffile

KNOWN_DETECTORS = {"BSE", "ETD", "Inlens", "SE"}

# TIFF tags that may carry microscope calibration (SPEC §0)
ZEISS_CZ_SEM_TAG = 34118
FEI_TAG = 34682
IMAGEDESCRIPTION_TAG = 270

FNAME_RE = re.compile(r"^img_(?P<gid>[A-Za-z0-9]+)_(?P<det>[A-Za-z]+)\.tiff?$")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _try_parse_px_nm_from_text(text: str) -> float | None:
    """Search free text (ImageDescription / CZ_SEM payload) for a pixel-size
    specification. Returns nm/px or None. Never invents a value."""
    if not text:
        return None
    # common spellings: "Pixel Size = 25.0 nm", "PixelSize=2.5e-8 m"
    m = re.search(
        r"pixel[_ ]?size\D{0,12}?([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*(nm|um|µm|m)\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(
            r"ImagePixelSize\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", text
        )
        if m:
            # Zeiss stores metres
            return float(m.group(1)) * 1e9
        return None
    val, unit = float(m.group(1)), m.group(2).lower()
    if unit == "nm":
        return val
    if unit in ("um", "µm"):
        return val * 1e3
    if unit == "m":
        return val * 1e9
    return None


def parse_pixel_size_nm(tif: tifffile.TiffFile) -> float | None:
    """Try the tags named in SPEC §0: Zeiss CZ_SEM (34118), FEI (34682),
    ImageDescription (270). Returns None when nothing usable is present."""
    page = tif.pages[0]
    for code in (ZEISS_CZ_SEM_TAG, FEI_TAG):
        try:
            tag = page.tags.get(code)
        except Exception:
            tag = None
        if tag is not None:
            nm = _try_parse_px_nm_from_text(str(tag.value))
            if nm is not None:
                return nm
    tag = page.tags.get(IMAGEDESCRIPTION_TAG)
    if tag is not None:
        return _try_parse_px_nm_from_text(str(tag.value))
    return None


def read_image_gray(path: str | Path) -> np.ndarray:
    """Read a TIFF as 2D uint8 grayscale (RGB -> luminance if needed)."""
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[..., :3]
        img = (
            0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        ).astype(np.uint8)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def parse_filename(fname: str) -> tuple[str, str] | None:
    m = FNAME_RE.match(Path(fname).name)
    if not m:
        return None
    return m.group("gid"), m.group("det")


def build_inventory(data_dir: str | Path) -> dict:
    """Walk Batch_* dirs, return inventory dict:
    {images: [{path, sha256, batch, group_id, detector, image_id, shape, px_size_nm}], ...}
    """
    data_dir = Path(data_dir)
    images = []
    for tif_path in sorted(data_dir.rglob("*.tif*")):
        parsed = parse_filename(tif_path.name)
        if parsed is None:
            continue
        gid, det = parsed
        batch = tif_path.parent.name
        with tifffile.TiffFile(tif_path) as tif:
            shape = list(tif.pages[0].shape)
            px = parse_pixel_size_nm(tif)
        images.append(
            {
                "path": str(tif_path),
                "sha256": sha256_file(tif_path),
                "batch": batch,
                "group_id": gid,
                "detector": det,
                "image_id": f"{batch}/{gid}/{det}",
                "shape": shape,
                "px_size_nm": px,
            }
        )
    return {"images": images, "n_images": len(images)}


def save_inventory(inv: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(inv, f, indent=2, sort_keys=True)


def load_inventory(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def groups_of(inv: dict, detector: str = "BSE") -> dict[str, list[dict]]:
    """group_id -> list of image entries for the given detector."""
    out: dict[str, list[dict]] = {}
    for e in inv["images"]:
        if e["detector"] == detector:
            out.setdefault(e["group_id"], []).append(e)
    return out


def image_path_for(inv: dict, image_id: str) -> str:
    for e in inv["images"]:
        if e["image_id"] == image_id:
            return e["path"]
    raise KeyError(image_id)

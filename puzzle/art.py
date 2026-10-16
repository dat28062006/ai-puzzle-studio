from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSET_KEYS = ("floor", "wall", "goal", "box", "player")
ASSET_LABELS = {"floor": "Nền", "wall": "Tường", "goal": "Đích", "box": "Thùng", "player": "Robot"}


def normalize_asset(data: bytes, key: str, size: int = 64) -> bytes:
    if key not in ASSET_KEYS:
        raise ValueError("Loại asset không hợp lệ.")
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Mỗi ảnh cần nhỏ hơn 10 MB.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width > 4096 or source.height > 4096:
                raise ValueError("Ảnh cần nhỏ hơn hoặc bằng 4096×4096.")
            image = ImageOps.exif_transpose(source).convert("RGBA")
            if key in ("floor", "wall"):
                image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
            else:
                image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
                canvas = Image.new("RGBA", (size, size))
                canvas.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
                image = canvas
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Không đọc được ảnh này.") from exc
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def load_assets(overrides: dict[str, bytes] | None = None, size: int = 88) -> dict[str, Image.Image]:
    result = {}
    overrides = overrides or {}
    for key in ASSET_KEYS:
        path = ASSET_DIR / "tiles" / f"{key}.png"
        data = overrides.get(key)
        if data is None and path.exists():
            data = path.read_bytes()
        if data:
            result[key] = Image.open(BytesIO(data)).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    return result


def asset_manifest(overrides: dict[str, bytes] | None = None) -> dict:
    path = ASSET_DIR / "prompts.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"theme": "Orbital Warehouse", "assets": {}}
    # Copy via JSON so replacing a tile never mutates cached default metadata.
    manifest = json.loads(json.dumps(manifest))
    for key in overrides or {}:
        manifest.setdefault("assets", {})[key] = {
            "source": "user_upload", "normalization": "PNG RGBA 64×64; preserve aspect ratio for objects",
            "generation_prompt": None,
        }
    return manifest

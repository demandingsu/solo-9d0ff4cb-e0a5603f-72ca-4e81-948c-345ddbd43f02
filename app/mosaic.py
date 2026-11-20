"""Tile validation, effective-tile selection and deterministic mosaic composition."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


def quote_field(value: str) -> str:
    """Minimal RFC 4180 quoting for the two free-text CSV columns."""
    if any(ch in value for ch in ',"\r\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


TILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
MAX_TILE_ID_LEN = 64


class TileValidationError(ValueError):
    """Raised when an uploaded tile or canvas setting is invalid."""


class LockedVersionError(RuntimeError):
    """Raised when a locked published version cannot be used for rendering."""


@dataclass(frozen=True)
class RegionLock:
    """One rectangle that keeps using pixels/provenance of a published version.

    The rectangle follows image conventions: left/top are inclusive,
    right/bottom are exclusive.  Coordinates are absolute canvas pixels.
    """

    version_id: str
    seq: int
    dir_rel: str
    x0: int
    y0: int
    x1: int
    y1: int
    canvas_width: int
    canvas_height: int
    fingerprint: str
    png_sha256: str = ""
    csv_sha256: str = ""

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_json(self) -> str:
        import json

        return json.dumps(
            {
                "version_id": self.version_id,
                "seq": self.seq,
                "dir_rel": self.dir_rel,
                "x0": self.x0,
                "y0": self.y0,
                "x1": self.x1,
                "y1": self.y1,
                "canvas_width": self.canvas_width,
                "canvas_height": self.canvas_height,
                "fingerprint": self.fingerprint,
                "png_sha256": self.png_sha256,
                "csv_sha256": self.csv_sha256,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "RegionLock":
        import json

        data = json.loads(raw)
        return cls(
            version_id=str(data["version_id"]),
            seq=int(data["seq"]),
            dir_rel=str(data["dir_rel"]),
            x0=int(data["x0"]),
            y0=int(data["y0"]),
            x1=int(data["x1"]),
            y1=int(data["y1"]),
            canvas_width=int(data["canvas_width"]),
            canvas_height=int(data["canvas_height"]),
            fingerprint=str(data["fingerprint"]),
            png_sha256=str(data.get("png_sha256", "")),
            csv_sha256=str(data.get("csv_sha256", "")),
        )


def validate_lock_rect(
    x0: str | int,
    y0: str | int,
    x1: str | int,
    y1: str | int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    """Validate a draft-canvas rectangle; right/bottom bounds are exclusive."""
    left = validate_int(x0, "锁区左边界")
    top = validate_int(y0, "锁区上边界")
    right = validate_int(x1, "锁区右边界")
    bottom = validate_int(y1, "锁区下边界")
    if left < 0 or top < 0 or left >= canvas_width or top >= canvas_height:
        raise TileValidationError("锁区左上角必须位于画幅内。")
    if right > canvas_width or bottom > canvas_height:
        raise TileValidationError("锁区右下角不能超出画幅。")
    if right <= left or bottom <= top:
        raise TileValidationError(
            "锁区必须非空（右、下边界不含，因此须大于左、上边界）。"
        )
    return left, top, right, bottom


def lock_matches_canvas(lock: RegionLock, width: int, height: int) -> bool:
    return (
        lock.canvas_width == width
        and lock.canvas_height == height
        and lock.x1 <= width
        and lock.y1 <= height
    )


@dataclass(frozen=True)
class TileRecord:
    id: int
    tile_id: str
    revision: int
    x: int
    y: int
    width: int
    height: int
    image_rel: str
    mask_rel: str
    image_sha256: str
    mask_sha256: str
    cloud_pixels: int
    clear_pixels: int
    submitted_at: str
    status: str = "active"

    @property
    def sort_key(self) -> tuple[datetime, int]:
        return (parse_timestamp(self.submitted_at), self.id)


def parse_timestamp(value: str) -> datetime:
    """Parse our stored ISO-8601 timestamps as timezone-aware UTC datetimes."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def validate_tile_id(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise TileValidationError("瓦片号不能为空。")
    if len(value) > MAX_TILE_ID_LEN:
        raise TileValidationError("瓦片号最多 64 个字符。")
    if not TILE_ID_RE.match(value):
        raise TileValidationError(
            "瓦片号只能包含字母、数字、下划线、点和连字符，且须以字母或数字开头。"
        )
    return value


def validate_revision(raw: str | int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise TileValidationError("修订号必须是整数。") from None
    if value < 1:
        raise TileValidationError("修订号必须是正整数（从 1 开始）。")
    return value


def validate_int(raw: str | int, label: str) -> int:
    try:
        text = str(raw).strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError
        return int(text)
    except (TypeError, ValueError):
        raise TileValidationError(f"{label}必须是整数。") from None


def validate_canvas(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise TileValidationError("画幅宽高必须为正整数。")
    if width > 20000 or height > 20000:
        raise TileValidationError("画幅宽高不能超过 20000 像素。")
    if width * height > 16_000_000:
        raise TileValidationError("画幅像素总量不能超过 1600 万（逐像素来源 CSV 会很庞大）。")


@dataclass
class ValidatedTile:
    tile_id: str
    revision: int
    x: int
    y: int
    width: int
    height: int
    image_bytes: bytes
    mask_bytes: bytes
    image_sha256: str
    mask_sha256: str
    cloud_pixels: int
    clear_pixels: int


def _decode_png(raw: bytes, label: str) -> Image.Image:
    if not raw:
        raise TileValidationError(f"{label}为空。")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise TileValidationError(f"{label}不是可读取的 PNG 图像。") from None
    if image.format != "PNG":
        raise TileValidationError(f"{label}必须是 PNG 格式。")
    return image


def validate_upload(
    *,
    tile_id: str,
    revision: str | int,
    x: str | int,
    y: str | int,
    image_bytes: bytes,
    mask_bytes: bytes,
) -> ValidatedTile:
    """Validate one RGB image plus same-size binary cloud mask.

    Cloud pixels are marks with any non-zero mask value.  The mask is
    interpreted as fully opaque cloud coverage: a cloudy pixel may never be
    used as a colour source.
    """
    tile_id = validate_tile_id(tile_id)
    revision = validate_revision(revision)
    x = validate_int(x, "X 位置")
    y = validate_int(y, "Y 位置")

    image = _decode_png(image_bytes, "瓦片图像")
    mask = _decode_png(mask_bytes, "云掩膜")

    if image.mode != "RGB":
        raise TileValidationError(f"瓦片图像必须是 RGB PNG（当前为 {image.mode}）。")
    if mask.mode not in ("1", "L", "P"):
        raise TileValidationError(
            f"云掩膜必须是同样尺寸的二值 PNG（当前为 {mask.mode}）。"
        )
    if image.size != mask.size:
        raise TileValidationError(
            "云掩膜尺寸必须与瓦片图像一致"
            f"（图像 {image.width}x{image.height}，掩膜 {mask.width}x{mask.height}）。"
        )

    width, height = image.size
    if width < 1 or height < 1:
        raise TileValidationError("瓦片尺寸必须为正。")

    if mask.mode != "L":
        mask = mask.convert("L")
    histogram = mask.histogram()
    clear_pixels = int(histogram[0])
    cloud_pixels = width * height - clear_pixels

    return ValidatedTile(
        tile_id=tile_id,
        revision=revision,
        x=x,
        y=y,
        width=width,
        height=height,
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        cloud_pixels=cloud_pixels,
        clear_pixels=clear_pixels,
    )


def effective_tiles(records: Iterable[TileRecord]) -> list[TileRecord]:
    """Newest active revision per tile id, ordered by submission time."""
    newest: dict[str, TileRecord] = {}
    for record in records:
        if record.status != "active":
            continue
        current = newest.get(record.tile_id)
        if current is None or (record.revision, record.sort_key) > (
            current.revision,
            current.sort_key,
        ):
            newest[record.tile_id] = record
    return sorted(newest.values(), key=lambda r: r.sort_key)


def compute_fingerprint(
    canvas_width: int,
    canvas_height: int,
    tiles: Sequence[TileRecord],
    lock: RegionLock | None = None,
) -> str:
    """Stable digest of the exact inputs a publish preview was built from."""
    digest = hashlib.sha256()
    digest.update(b"mosaic-preview-v2\0")
    digest.update(f"{canvas_width}x{canvas_height}\0".encode())
    if lock is not None:
        digest.update(
            (
                "lock\0"
                f"{lock.version_id}\0{lock.seq}\0{lock.dir_rel}\0"
                f"{lock.x0}\0{lock.y0}\0{lock.x1}\0{lock.y1}\0"
                f"{lock.canvas_width}\0{lock.canvas_height}\0"
                f"{lock.fingerprint}\0"
            ).encode()
        )
    for tile in sorted(tiles, key=lambda t: (t.tile_id, t.revision, t.id)):
        digest.update(
            (
                f"{tile.id}\0{tile.tile_id}\0{tile.revision}\0{tile.x}\0{tile.y}\0"
                f"{tile.width}\0{tile.height}\0{tile.image_sha256}\0"
                f"{tile.mask_sha256}\0"
            ).encode()
        )
    return digest.hexdigest()


@dataclass
class _Crop:
    tile: TileRecord
    x0: int
    y0: int
    x1: int
    y1: int
    rank: int
    clear: bool = False

    @property
    def placement(self) -> tuple[int, int]:
        return (self.x0 - self.tile.x, self.y0 - self.tile.y)


@dataclass
class RenderStats:
    width: int
    height: int
    covered_pixels: int
    transparent_pixels: int
    # (tile_id, revision) labels actually used inside a locked region, if any.
    lock_sources: list[tuple[str, int]] = field(default_factory=list)


LOCKED_PNG_NAME = "mosaic.png"
LOCKED_CSV_NAME = "sources.csv"


@dataclass
class _LockedData:
    rgba: Image.Image
    labels: list[tuple[str, str]]
    indices: array


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lock_files(lock: RegionLock, versions_root: Path) -> None:
    """Cheap check that the locked version files look present and decodable.

    The full pixel/provenance read and checksum verification happen at render
    time; here we only make sure a user cannot lock a version whose files are
    already unusable.  Reading the PNG header does not decode the pixels.
    """
    base = versions_root / lock.dir_rel
    png_path = base / LOCKED_PNG_NAME
    csv_path = base / LOCKED_CSV_NAME
    if not png_path.is_file() or not csv_path.is_file():
        raise LockedVersionError("锁定版本缺少 PNG 或来源 CSV 文件。")
    try:
        with Image.open(png_path) as image:
            if image.size != (lock.canvas_width, lock.canvas_height):
                raise LockedVersionError("锁定版本 PNG 尺寸与发布记录不一致。")
    except LockedVersionError:
        raise
    except Exception:
        raise LockedVersionError("锁定版本 PNG 无法读取。") from None
    try:
        with open(csv_path, "rb") as handle:
            header = handle.readline().decode("utf-8", "replace").strip()
    except OSError:
        raise LockedVersionError("锁定版本来源 CSV 无法读取。") from None
    if header != "x,y,tile_id,revision":
        raise LockedVersionError("锁定版本来源 CSV 格式不正确。")


def _load_locked_region(lock: RegionLock, versions_root: Path) -> _LockedData:
    """Read the exact pixels and per-pixel provenance of a lock rectangle.

    The published PNG and CSV are immutable once a version directory becomes
    visible, so a snapshot taken here is stable for the whole render.  Any
    missing, unreadable, size-mismatched or checksum-mismatched file aborts
    the render instead of falling back to draft pixels.
    """
    base = versions_root / lock.dir_rel
    png_path = base / LOCKED_PNG_NAME
    csv_path = base / LOCKED_CSV_NAME
    if not png_path.is_file() or not csv_path.is_file():
        raise LockedVersionError(
            f"锁定版本 #{lock.seq} 缺少 PNG 或来源 CSV 文件，已中止生成。"
        )
    try:
        png_sha = _file_sha256(png_path)
        csv_sha = _file_sha256(csv_path)
    except OSError:
        raise LockedVersionError(
            f"锁定版本 #{lock.seq} 的文件无法读取，已中止生成。"
        ) from None
    if lock.png_sha256 and png_sha != lock.png_sha256:
        raise LockedVersionError(f"锁定版本 #{lock.seq} 的 PNG 校验失败，已中止生成。")
    if lock.csv_sha256 and csv_sha != lock.csv_sha256:
        raise LockedVersionError(
            f"锁定版本 #{lock.seq} 的来源 CSV 校验失败，已中止生成。"
        )

    try:
        with Image.open(png_path) as image:
            image.load()
            if image.size != (lock.canvas_width, lock.canvas_height):
                raise LockedVersionError("锁定版本 PNG 尺寸与发布记录不一致。")
            rgba = image.convert("RGBA").crop((lock.x0, lock.y0, lock.x1, lock.y1))
    except LockedVersionError:
        raise
    except Exception:
        raise LockedVersionError(
            f"锁定版本 #{lock.seq} 的 PNG 无法读取，已中止生成。"
        ) from None

    width = lock.width
    height = lock.height
    rect_pixels = width * height
    labels: list[tuple[str, str]] = [("", "")]
    label_index: dict[tuple[str, str], int] = {("", ""): 0}
    indices = array("i", [0]) * rect_pixels

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                raise LockedVersionError("锁定版本来源 CSV 为空。") from None
            if header != ["x", "y", "tile_id", "revision"]:
                raise LockedVersionError("锁定版本来源 CSV 表头不正确。")
            total = lock.canvas_width * lock.canvas_height
            count = 0
            for row in reader:
                if count >= total:
                    raise LockedVersionError("锁定版本来源 CSV 行数超出画幅。")
                try:
                    x = int(row[0])
                    y = int(row[1])
                except (ValueError, IndexError):
                    raise LockedVersionError("锁定版本来源 CSV 坐标不正确。") from None
                if len(row) != 4 or (
                    x != count % lock.canvas_width
                    or y != count // lock.canvas_width
                ):
                    raise LockedVersionError("锁定版本来源 CSV 的像素顺序不正确。")
                count += 1
                if (lock.x0 <= x < lock.x1) and (lock.y0 <= y < lock.y1):
                    key = (row[2], row[3])
                    idx = label_index.get(key)
                    if idx is None:
                        idx = len(labels)
                        label_index[key] = idx
                        labels.append(key)
                    indices[(y - lock.y0) * width + (x - lock.x0)] = idx
            if count != total:
                raise LockedVersionError("锁定版本来源 CSV 行数与画幅不一致。")
    except LockedVersionError:
        raise
    except (OSError, UnicodeDecodeError, csv.Error):
        raise LockedVersionError(
            f"锁定版本 #{lock.seq} 的来源 CSV 无法读取，已中止生成。"
        ) from None

    return _LockedData(rgba=rgba, labels=labels, indices=indices)


def _flat_pixels(image: Image.Image):
    """Pillow 13 renamed getdata(); support both old and new releases."""
    getter = getattr(image, "get_flattened_data", None)
    if getter is not None:
        return getter()
    return image.getdata()


def render_mosaic(
    canvas_width: int,
    canvas_height: int,
    tiles: Sequence[TileRecord],
    storage_root: Path,
    png_path: Path,
    csv_path: Path,
    lock: RegionLock | None = None,
    versions_root: Path | None = None,
) -> RenderStats:
    """Compose the canvas and provenance CSV into the given files.

    Cloud pixels are transparent.  For the remaining pixels the most recently
    submitted tile (by submission timestamp, not revision number) wins, as
    long as it has a cloud-free value there; otherwise the earlier tiles act
    as fallbacks.  Pixels with no clear source stay transparent.

    When `lock` is given, the rectangle is overwritten (not blended) with the
    exact RGBA pixels and per-pixel provenance of the selected published
    version.  Transparent pixels keep their empty source.  The locked files
    are read in full before anything is composed; an unreadable or
    checksum-mismatched locked version aborts the render.
    """
    validate_canvas(canvas_width, canvas_height)
    locked: _LockedData | None = None
    if lock is not None:
        if not lock_matches_canvas(lock, canvas_width, canvas_height):
            raise LockedVersionError("锁区与当前画幅不匹配，已中止生成。")
        versions_root = versions_root if versions_root is not None else storage_root
        locked = _load_locked_region(lock, versions_root)

    # Newest first: index 0 has the highest priority.
    ordered = sorted(tiles, key=lambda t: t.sort_key, reverse=True)
    crops: list[_Crop] = []
    for rank, tile in enumerate(ordered):
        x0 = max(tile.x, 0)
        y0 = max(tile.y, 0)
        x1 = min(tile.x + tile.width, canvas_width)
        y1 = min(tile.y + tile.height, canvas_height)
        if x1 <= x0 or y1 <= y0:
            continue
        crops.append(_Crop(tile=tile, x0=x0, y0=y0, x1=x1, y1=y1, rank=rank))

    canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    alpha = Image.new("L", (canvas_width, canvas_height), 0)
    owner = Image.new("I", (canvas_width, canvas_height), 0xFFFFFFFF)
    sentinel = 0xFFFFFFFF

    # Apply oldest first so newer tiles paint over them.
    for crop in reversed(crops):
        tile_path = storage_root / crop.tile.image_rel
        mask_path = storage_root / crop.tile.mask_rel
        with Image.open(tile_path) as image, Image.open(mask_path) as mask:
            image = image.convert("RGB")
            mask = mask.convert("L")
            if image.size != (crop.tile.width, crop.tile.height):
                raise RuntimeError(
                    f"瓦片 {crop.tile.tile_id} r{crop.tile.revision} 文件尺寸已变化。"
                )
            left, top = crop.placement
            box = (left, top, left + (crop.x1 - crop.x0), top + (crop.y1 - crop.y0))
            image_crop = image.crop(box)
            mask_crop = mask.crop(box)
        clear_mask = mask_crop.point(lambda v: 255 if v == 0 else 0)
        mask_pixels = list(_flat_pixels(clear_mask))
        if not any(mask_pixels):
            continue
        crop.clear = True
        canvas.paste(image_crop, (crop.x0, crop.y0), clear_mask)
        alpha.paste(255, (crop.x0, crop.y0), clear_mask)
        # Stream the owner map row by row (rank + 1, so 0 stays "no source").
        crop_width = crop.x1 - crop.x0
        overlay = Image.new("I", (crop_width, crop.y1 - crop.y0), 0)
        overlay_rows = []
        for start in range(0, len(mask_pixels), crop_width):
            overlay_rows.extend(
                crop.rank + 1 if value else 0
                for value in mask_pixels[start:start + crop_width]
            )
        overlay.putdata(overlay_rows)
        owner.paste(overlay, (crop.x0, crop.y0), clear_mask)

    lock_sources: list[tuple[str, int]] = []
    if locked is not None and lock is not None:
        # The lock rectangle wins over every draft tile, including where the
        # draft has a newer clear pixel: the acceptance already fixed here.
        # Only the RGBA plane is pasted; provenance is resolved in the CSV
        # loop by coordinates, so a full-canvas marker map is not built.
        canvas.paste(locked.rgba.convert("RGB"), (lock.x0, lock.y0))
        alpha.paste(locked.rgba.getchannel("A"), (lock.x0, lock.y0))

        used: list[tuple[str, int]] = []
        seen_labels: set[int] = set()
        # Fast path: a fully transparent locked region cannot use any source.
        if locked.rgba.getchannel("A").getextrema()[1] != 0:
            for index, opacity in zip(
                locked.indices, _flat_pixels(locked.rgba.getchannel("A"))
            ):
                if opacity == 0 or index in seen_labels:
                    continue
                label = locked.labels[index]
                if label == ("", ""):
                    raise LockedVersionError(
                        "锁定版本来源 CSV 与 PNG 不一致"
                        f"（{lock.seq} 版有可见像元却无来源标记），已中止生成。"
                    )
                seen_labels.add(index)
                try:
                    revision = int(label[1])
                except ValueError:
                    raise LockedVersionError(
                        "锁定版本来源 CSV 含有非整数修订号，已中止生成。"
                    )
                used.append((label[0], revision))
        lock_sources = used

    rgba = canvas.convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(png_path, format="PNG")

    rank_to_tile = {crop.rank: crop.tile for crop in crops if crop.clear}
    alphas = _flat_pixels(alpha)
    covered = 0
    label_for_rank = {
        rank: (tile.tile_id, str(tile.revision))
        for rank, tile in rank_to_tile.items()
    }
    blank = ("", "")
    # Locked provenance is copied verbatim, so a transparent locked pixel
    # keeps its empty source and a labelled pixel keeps its label.
    locked_indices = locked.indices if locked is not None else None
    locked_labels = locked.labels if locked is not None else []
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("x,y,tile_id,revision\n")
        rows = []
        y = 0
        x = 0
        owners = _flat_pixels(owner)
        for index, (marker, opacity) in enumerate(zip(owners, alphas)):
            if opacity != 0:
                covered += 1
            in_lock = (
                locked is not None
                and lock.x0 <= x < lock.x1
                and lock.y0 <= y < lock.y1
            )
            if in_lock:
                idx = locked_indices[(y - lock.y0) * lock.width + (x - lock.x0)]
                # A transparent locked pixel always keeps an empty source, even
                # if the stored CSV were inconsistent with the PNG.
                label = (
                    blank
                    if opacity == 0
                    else locked_labels[idx]
                    if 0 <= idx < len(locked_labels)
                    else blank
                )
            elif opacity == 0 or marker == sentinel:
                label = blank
            else:
                label = label_for_rank.get(marker - 1, blank)
            rows.append(f"{x},{y},{quote_field(label[0])},{label[1]}")
            x += 1
            if x == canvas_width:
                x = 0
                y += 1
            if len(rows) >= 20000:
                handle.write("\n".join(rows) + "\n")
                rows.clear()
        if rows:
            handle.write("\n".join(rows) + "\n")

    total = canvas_width * canvas_height
    return RenderStats(
        width=canvas_width,
        height=canvas_height,
        covered_pixels=covered,
        transparent_pixels=total - covered,
        lock_sources=lock_sources,
    )

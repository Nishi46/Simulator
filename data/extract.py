"""Safe zip extraction for downloaded GSS / PUMS archives.

Never uses `ZipFile.extractall`. Each member is validated, then streamed with
byte counting into a staging directory that is moved into place only if the
whole archive passes, so a rejected archive leaves nothing behind.
"""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from data.paths import DATA_DIR, UnsafePathError, safe_join

CHUNK = 64 * 1024
MAX_NAME_LEN = 255
MAX_MEMBERS = 10_000
MIN_RATIO_WINDOW = 1024  # tiny members may compress "well" without being a bomb

_NESTED_EXT = {".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".jar", ".z"}
_NESTED_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"7z\xbc\xaf\x27\x1c")


class ExtractError(ValueError):
    """Raised when an archive is rejected or cannot be extracted safely."""


def _validate_member(info: zipfile.ZipInfo) -> tuple[str, bool]:
    """Return (clean relative name, is_dir) or raise ExtractError."""
    name = info.filename
    is_dir = name.endswith("/")
    clean = name.rstrip("/")
    if not clean or len(clean) > MAX_NAME_LEN * 4 or any(len(p) > MAX_NAME_LEN for p in clean.split("/")):
        raise ExtractError(f"bad member name length: {name[:60]!r}")
    if info.flag_bits & 0x1:
        raise ExtractError(f"encrypted member not allowed: {name[:60]!r}")
    mode = info.external_attr >> 16
    if mode:
        kind = stat.S_IFMT(mode)
        if kind == stat.S_IFLNK:
            raise ExtractError(f"symlink member not allowed: {name[:60]!r}")
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ExtractError(f"special file member not allowed: {name[:60]!r}")
        if (kind == stat.S_IFDIR) != is_dir and kind != 0:
            raise ExtractError(f"member type does not match its name: {name[:60]!r}")
    if not is_dir and os.path.splitext(clean.lower())[1] in _NESTED_EXT:
        raise ExtractError(f"nested archive not allowed: {name[:60]!r}")
    return clean, is_dir


def safe_extract(
    rel_zip: str,
    rel_out: str,
    *,
    max_total: int,
    max_ratio: int = 100,
    max_members: int = MAX_MEMBERS,
    base_dir: str | Path = DATA_DIR / "raw",
) -> list[str]:
    """Extract `base_dir/rel_zip` into the new directory `base_dir/rel_out`.

    Returns the sorted relative paths of extracted files. `rel_out` must not
    exist (or be empty). On any problem nothing is left behind.
    """
    for label, v in (("max_total", max_total), ("max_ratio", max_ratio), ("max_members", max_members)):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ExtractError(f"{label} must be a positive integer")

    base = Path(base_dir)
    zpath = safe_join(base, rel_zip)
    out = safe_join(base, rel_out)
    if not zpath.is_file():
        raise ExtractError(f"archive not found: {rel_zip}")
    if out == base.resolve() or out == zpath:
        raise ExtractError("output directory must be a new subdirectory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ExtractError("output directory exists and is not empty")
    out.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}.", suffix=".staging"))
    try:
        files = _extract_into(zpath, staging, max_total, max_ratio, max_members)
        if out.exists():
            out.rmdir()  # empty, checked above
        os.replace(staging, out)
        return files
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError, UnsafePathError, NotImplementedError, RuntimeError) as e:
        if isinstance(e, ExtractError):
            raise
        raise ExtractError(f"extraction failed ({type(e).__name__}): {e}") from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _extract_into(zpath: Path, staging: Path, max_total: int, max_ratio: int, max_members: int) -> list[str]:
    with zipfile.ZipFile(zpath) as zf:
        infos = zf.infolist()
        if not infos:
            raise ExtractError("archive is empty")
        if len(infos) > max_members:
            raise ExtractError(f"archive has more than {max_members} members")

        seen: set[str] = set()
        plan: list[tuple[zipfile.ZipInfo, str, bool]] = []
        declared = comp_total = 0
        for info in infos:
            clean, is_dir = _validate_member(info)
            key = clean.casefold()  # also catches collisions on case-insensitive filesystems
            if key in seen:
                raise ExtractError(f"duplicate member name: {clean[:60]!r}")
            seen.add(key)
            declared += info.file_size
            comp_total += info.compress_size
            plan.append((info, clean, is_dir))
        if declared > max_total:
            raise ExtractError(f"declared uncompressed size {declared} exceeds {max_total}")
        if declared > max_ratio * max(comp_total, MIN_RATIO_WINDOW):
            raise ExtractError("compression ratio too high (possible zip bomb)")
        if not any(not d for _, _, d in plan):
            raise ExtractError("archive contains no files")

        written_total = 0
        files: list[str] = []
        for info, clean, is_dir in plan:
            target = safe_join(staging, clean)
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
                os.chmod(target, 0o755)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_limit = max_ratio * max(info.compress_size, MIN_RATIO_WINDOW)
            n = 0
            with zf.open(info) as src, open(target, "xb") as dst:
                while chunk := src.read(CHUNK):
                    if n == 0 and chunk.startswith(_NESTED_MAGIC):
                        raise ExtractError(f"nested archive content not allowed: {clean[:60]!r}")
                    n += len(chunk)
                    written_total += len(chunk)
                    if written_total > max_total:
                        raise ExtractError(f"extracted data exceeds {max_total} bytes")
                    if n > member_limit:
                        raise ExtractError(f"member {clean[:60]!r} expands too much (possible zip bomb)")
                    dst.write(chunk)
            if n != info.file_size:
                raise ExtractError(f"member {clean[:60]!r} size does not match its header")
            os.chmod(target, 0o644)  # ignore any mode bits (setuid, executable) from the archive
            files.append(clean)
        return sorted(files)

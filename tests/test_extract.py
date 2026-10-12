import os
import stat
import struct
import warnings
import zipfile

import pytest

from data.extract import ExtractError, safe_extract
from data.paths import UnsafePathError

CD_SIG = b"PK\x01\x02"


def make_zip(path, members, compression=zipfile.ZIP_DEFLATED):
    """members: list of (name | ZipInfo, bytes)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate-name warnings
        with zipfile.ZipFile(path, "w", compression) as z:
            for name, data in members:
                z.writestr(name, data)
    return path


def info(name, mode=None):
    zi = zipfile.ZipInfo(name)
    if mode is not None:
        zi.external_attr = mode << 16
    zi.compress_type = zipfile.ZIP_DEFLATED
    return zi


def patch_central(path, field_offset, fmt, value):
    """Patch a field in every central-directory header."""
    raw = bytearray(path.read_bytes())
    i = 0
    while (i := raw.find(CD_SIG, i)) != -1:
        struct.pack_into(fmt, raw, i + field_offset, value)
        i += 4
    path.write_bytes(bytes(raw))


def run(tmp_path, members=None, zip_name="a.zip", out="out", **kw):
    if members is not None:
        make_zip(tmp_path / zip_name, members)
    kw.setdefault("max_total", 10_000_000)
    return safe_extract(zip_name, out, base_dir=tmp_path, **kw)


def clean(tmp_path, out="out"):
    """Rejected archive leaves no output and no staging dirs."""
    assert not (tmp_path / out).exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".staging")]


def test_normal_archive(tmp_path):
    files = run(tmp_path, [("gss.dta", b"abc" * 10), ("docs/readme.txt", b"hi"), ("docs/", b"")])
    assert files == ["docs/readme.txt", "gss.dta"]
    assert (tmp_path / "out/gss.dta").read_bytes() == b"abc" * 10
    assert (tmp_path / "out/docs/readme.txt").read_bytes() == b"hi"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".staging")]


def test_unicode_and_empty_file_ok(tmp_path):
    assert run(tmp_path, [("caf\u00e9.csv", b"x"), ("empty.txt", b"")]) == ["caf\u00e9.csv", "empty.txt"]
    assert (tmp_path / "out/empty.txt").read_bytes() == b""


def test_stored_not_just_deflated(tmp_path):
    make_zip(tmp_path / "a.zip", [("f.txt", b"hello")], compression=zipfile.ZIP_STORED)
    assert run(tmp_path) == ["f.txt"]


@pytest.mark.parametrize("name", [
    "../evil.txt", "a/../../evil.txt", "/abs/evil.txt", "..", "a\\..\\evil.txt",
    "C:/evil.txt", "a\x01b.txt", "a/\n.txt",
])
def test_unsafe_member_names_rejected(tmp_path, name):
    with pytest.raises((ExtractError, UnsafePathError)):
        run(tmp_path, [("ok.txt", b"x"), (info(name), b"evil")])
    clean(tmp_path)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_symlink_member_rejected(tmp_path):
    with pytest.raises(ExtractError, match="symlink"):
        run(tmp_path, [("ok.txt", b"x"), (info("link", stat.S_IFLNK | 0o777), b"/etc/passwd")])
    clean(tmp_path)


@pytest.mark.parametrize("kind", [stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK])
def test_special_file_rejected(tmp_path, kind):
    with pytest.raises(ExtractError, match="special"):
        run(tmp_path, [(info("dev", kind | 0o666), b"")])
    clean(tmp_path)


def test_mode_bits_ignored(tmp_path):
    run(tmp_path, [(info("run.sh", stat.S_IFREG | 0o4777), b"#!/bin/sh")])
    assert stat.S_IMODE((tmp_path / "out/run.sh").stat().st_mode) == 0o644


def test_encrypted_member_rejected(tmp_path):
    make_zip(tmp_path / "a.zip", [("f.txt", b"x")])
    patch_central(tmp_path / "a.zip", 8, "<H", 0x1)
    with pytest.raises(ExtractError, match="encrypted"):
        run(tmp_path)
    clean(tmp_path)


@pytest.mark.parametrize("name", ["inner.zip", "x.tar", "x.GZ", "x.tgz", "x.7z", "x.rar", "x.bz2", "x.xz"])
def test_nested_archive_by_extension_rejected(tmp_path, name):
    with pytest.raises(ExtractError, match="nested"):
        run(tmp_path, [("ok.txt", b"x"), (name, b"data")])
    clean(tmp_path)


@pytest.mark.parametrize("magic", [b"PK\x03\x04rest", b"\x1f\x8brest", b"BZh9rest", b"7z\xbc\xaf\x27\x1crest"])
def test_nested_archive_by_content_rejected(tmp_path, magic):
    with pytest.raises(ExtractError, match="nested"):
        run(tmp_path, [("disguised.dat", magic)])
    clean(tmp_path)


def test_real_nested_zip_content_disguised(tmp_path):
    inner = tmp_path / "inner.zip"
    make_zip(inner, [("x.txt", b"x")])
    with pytest.raises(ExtractError, match="nested"):
        run(tmp_path, [("data.dta", inner.read_bytes())])
    clean(tmp_path)


def test_zip_bomb_by_ratio_rejected(tmp_path):
    with pytest.raises(ExtractError, match="ratio|expands"):
        run(tmp_path, [("zeros.bin", b"\0" * 5_000_000)], max_total=100_000_000)
    clean(tmp_path)


def test_many_medium_members_total_ratio_rejected(tmp_path):
    members = [(f"z{i}.bin", b"\0" * 90_000) for i in range(40)]
    with pytest.raises(ExtractError, match="ratio|expands"):
        run(tmp_path, members, max_total=100_000_000)
    clean(tmp_path)


def test_total_size_cap(tmp_path):
    members = [("a.bin", os.urandom(600)), ("b.bin", os.urandom(600))]
    with pytest.raises(ExtractError, match="exceeds"):
        run(tmp_path, members, max_total=1000)
    clean(tmp_path)
    assert run(tmp_path, members, max_total=1200) == ["a.bin", "b.bin"]


def test_declared_size_larger_than_cap_rejected(tmp_path):
    make_zip(tmp_path / "a.zip", [("f.bin", os.urandom(100))])
    patch_central(tmp_path / "a.zip", 24, "<I", 2_000_000_000)
    with pytest.raises(ExtractError):
        run(tmp_path, max_total=1_000_000)
    clean(tmp_path)


def test_header_lies_smaller_than_real_size_rejected(tmp_path):
    make_zip(tmp_path / "a.zip", [("f.bin", os.urandom(500))])
    patch_central(tmp_path / "a.zip", 24, "<I", 10)
    with pytest.raises(ExtractError):
        run(tmp_path)
    clean(tmp_path)


def test_duplicate_names_rejected(tmp_path):
    with pytest.raises(ExtractError, match="duplicate"):
        run(tmp_path, [("a.txt", b"1"), ("a.txt", b"2")])
    clean(tmp_path)


def test_case_collision_rejected(tmp_path):
    with pytest.raises(ExtractError, match="duplicate"):
        run(tmp_path, [("A.txt", b"1"), ("a.TXT", b"2")])
    clean(tmp_path)


def test_file_directory_conflict_rejected(tmp_path):
    with pytest.raises(ExtractError):
        run(tmp_path, [("a", b"file"), ("a/b.txt", b"x")])
    clean(tmp_path)


def test_too_many_members(tmp_path):
    with pytest.raises(ExtractError, match="more than"):
        run(tmp_path, [(f"f{i}.txt", b"x") for i in range(6)], max_members=5)
    clean(tmp_path)


def test_overlong_names(tmp_path):
    with pytest.raises(ExtractError, match="length"):
        run(tmp_path, [("x" * 300 + ".txt", b"x")])
    clean(tmp_path)


def test_empty_and_dir_only_archives(tmp_path):
    with pytest.raises(ExtractError, match="empty"):
        run(tmp_path, [])
    clean(tmp_path)
    with pytest.raises(ExtractError, match="no files"):
        run(tmp_path, [("d/", b"")])
    clean(tmp_path)


def test_not_a_zip_and_corrupt(tmp_path):
    (tmp_path / "a.zip").write_bytes(os.urandom(200))
    with pytest.raises(ExtractError):
        run(tmp_path)
    clean(tmp_path)
    make_zip(tmp_path / "b.zip", [("f.bin", os.urandom(300))])
    raw = bytearray((tmp_path / "b.zip").read_bytes())
    raw[40] ^= 0xFF  # corrupt compressed data -> CRC/decompress error
    (tmp_path / "b.zip").write_bytes(bytes(raw))
    with pytest.raises(ExtractError):
        run(tmp_path, zip_name="b.zip")
    clean(tmp_path)


def test_missing_archive_and_directory_as_archive(tmp_path):
    with pytest.raises(ExtractError, match="not found"):
        run(tmp_path, zip_name="nope.zip")
    (tmp_path / "d.zip").mkdir()
    with pytest.raises(ExtractError, match="not found"):
        run(tmp_path, zip_name="d.zip")


@pytest.mark.parametrize("rel_zip,rel_out", [("../a.zip", "out"), ("a.zip", "../out"), ("/a.zip", "out"), ("a.zip", "/tmp/out")])
def test_path_traversal_in_arguments(tmp_path, rel_zip, rel_out):
    make_zip(tmp_path / "a.zip", [("f.txt", b"x")])
    with pytest.raises(UnsafePathError):
        safe_extract(rel_zip, rel_out, max_total=1000, base_dir=tmp_path)


def test_output_dir_rules(tmp_path):
    make_zip(tmp_path / "a.zip", [("f.txt", b"x")])
    (tmp_path / "full").mkdir()
    (tmp_path / "full/keep.txt").write_text("keep")
    with pytest.raises(ExtractError, match="not empty"):
        safe_extract("a.zip", "full", max_total=1000, base_dir=tmp_path)
    assert (tmp_path / "full/keep.txt").read_text() == "keep"
    (tmp_path / "empty").mkdir()
    assert safe_extract("a.zip", "empty", max_total=1000, base_dir=tmp_path) == ["f.txt"]
    with pytest.raises(ExtractError, match="new subdirectory"):
        safe_extract("a.zip", ".", max_total=1000, base_dir=tmp_path)
    (tmp_path / "afile").write_text("x")
    with pytest.raises(ExtractError, match="not empty"):
        safe_extract("a.zip", "afile", max_total=1000, base_dir=tmp_path)


def test_output_symlink_escape_rejected(tmp_path):
    base, outside = tmp_path / "base", tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    make_zip(base / "a.zip", [("f.txt", b"x")])
    os.symlink(outside, base / "link")
    with pytest.raises(UnsafePathError):
        safe_extract("a.zip", "link/out", max_total=1000, base_dir=base)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("kw", [{"max_total": 0}, {"max_total": -1}, {"max_total": True}, {"max_total": 1.5},
                                {"max_total": 10, "max_ratio": 0}, {"max_total": 10, "max_members": 0}])
def test_bad_parameters(tmp_path, kw):
    make_zip(tmp_path / "a.zip", [("f.txt", b"x")])
    with pytest.raises(ExtractError):
        safe_extract("a.zip", "out", base_dir=tmp_path, **kw)


def test_deterministic(tmp_path):
    members = [("b.txt", b"2"), ("a.txt", b"1")]
    first = run(tmp_path, members, out="o1")
    second = run(tmp_path, None, out="o2")
    assert first == second == ["a.txt", "b.txt"]

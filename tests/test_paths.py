import os

import pytest

from data.paths import DATA_DIR, UnsafePathError, safe_join


def test_simple_and_nested_paths(tmp_path):
    assert safe_join(tmp_path, "a.csv") == tmp_path.resolve() / "a.csv"
    assert safe_join(tmp_path, "raw", "gss", "x.dta") == tmp_path.resolve() / "raw/gss/x.dta"
    assert safe_join(tmp_path, "raw/gss/x.dta") == tmp_path.resolve() / "raw/gss/x.dta"


def test_no_parts_returns_base(tmp_path):
    assert safe_join(tmp_path) == tmp_path.resolve()


def test_inner_dot_segments_that_stay_inside_are_ok(tmp_path):
    assert safe_join(tmp_path, "a/./b") == tmp_path.resolve() / "a/b"
    assert safe_join(tmp_path, "..foo") == tmp_path.resolve() / "..foo"  # a name, not traversal


@pytest.mark.parametrize("part", [
    "..", "../x", "a/../../x", "a/..", "../../etc/passwd", "./../x",
])
def test_parent_traversal_rejected(tmp_path, part):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, part)


def test_traversal_split_across_parts_rejected(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "a", "..", "..", "x")


@pytest.mark.parametrize("part", ["/etc/passwd", "/", "C:\\Windows", "C:foo", "c:/x"])
def test_absolute_rejected(tmp_path, part):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, part)


def test_absolute_in_later_part_rejected(tmp_path):
    # Path.joinpath would silently discard everything before an absolute part.
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "ok", "/etc/passwd")


@pytest.mark.parametrize("part", ["a\\b", "..\\x", "a\\..\\..\\x"])
def test_backslash_rejected(tmp_path, part):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, part)


@pytest.mark.parametrize("part", ["a\x00b", "a\nb", "a\tb", "\x7f", "a\rb"])
def test_control_chars_rejected(tmp_path, part):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, part)


@pytest.mark.parametrize("part", ["", "   ", "\u00a0"[:0]])
def test_empty_rejected(tmp_path, part):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, part)


@pytest.mark.parametrize("part", [None, 5, b"x", ["a"], Ellipsis])
def test_non_string_rejected(tmp_path, part):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, part)


def test_symlink_pointing_outside_rejected(tmp_path):
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    os.symlink(outside, base / "link")
    with pytest.raises(UnsafePathError):
        safe_join(base, "link")
    with pytest.raises(UnsafePathError):
        safe_join(base, "link", "secret.txt")
    with pytest.raises(UnsafePathError):
        safe_join(base, "link", "new_file_to_create.txt")


def test_dangling_symlink_pointing_outside_rejected(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    os.symlink(tmp_path / "does_not_exist", base / "dangling")
    with pytest.raises(UnsafePathError):
        safe_join(base, "dangling")


def test_symlink_inside_base_allowed(tmp_path):
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "alias")
    assert safe_join(tmp_path, "alias", "f.txt") == tmp_path.resolve() / "real" / "f.txt"


def test_symlinked_base_is_resolved(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    os.symlink(real, tmp_path / "linkbase")
    assert safe_join(tmp_path / "linkbase", "x") == real.resolve() / "x"


def test_sibling_with_shared_prefix_rejected(tmp_path):
    base = tmp_path / "data"
    sibling = tmp_path / "data_evil"
    base.mkdir()
    sibling.mkdir()
    os.symlink(sibling, base / "link")
    with pytest.raises(UnsafePathError):
        safe_join(base, "link", "x")


def test_base_must_exist_and_be_dir(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path / "missing", "x")
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(UnsafePathError):
        safe_join(f, "x")


def test_nonexistent_target_allowed(tmp_path):
    assert safe_join(tmp_path, "new", "dir", "file.txt").parent.name == "dir"


def test_unicode_names_ok(tmp_path):
    assert safe_join(tmp_path, "caf\u00e9.csv").name == "caf\u00e9.csv"


def test_default_data_dir_exists():
    assert DATA_DIR.is_dir()
    assert safe_join(DATA_DIR, "raw", "x") == DATA_DIR.resolve() / "raw" / "x"

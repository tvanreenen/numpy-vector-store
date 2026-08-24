"""Tests for release version agreement."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.verify_release_version import ReleaseVersionError, verify_release_version


def _metadata(version: str) -> bytes:
    return f"Metadata-Version: 2.4\nName: example\nVersion: {version}\n".encode()


def _write_release_files(
    tmp_path: Path, *, version: str = "0.7.0"
) -> tuple[Path, Path]:
    source_file = tmp_path / "__init__.py"
    source_file.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    distribution_directory = tmp_path / "dist"
    distribution_directory.mkdir()

    wheel = distribution_directory / f"example-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr(f"example-{version}.dist-info/METADATA", _metadata(version))

    sdist = distribution_directory / f"example-{version}.tar.gz"
    metadata = _metadata(version)
    member = tarfile.TarInfo(name=f"example-{version}/PKG-INFO")
    member.size = len(metadata)
    with tarfile.open(sdist, mode="w:gz") as archive:
        archive.addfile(member, io.BytesIO(metadata))

    return source_file, distribution_directory


def test_release_versions_agree(tmp_path):
    """Test matching source, artifact, and tag versions pass."""
    source_file, distribution_directory = _write_release_files(tmp_path)

    version = verify_release_version(
        source_file=source_file,
        distribution_directory=distribution_directory,
        tag="v0.7.0",
    )

    assert version == "0.7.0"


def test_testpypi_verification_does_not_require_a_tag(tmp_path):
    """Test manually dispatched candidate builds can omit a release tag."""
    source_file, distribution_directory = _write_release_files(tmp_path)

    version = verify_release_version(
        source_file=source_file,
        distribution_directory=distribution_directory,
    )

    assert version == "0.7.0"


@pytest.mark.parametrize(
    ("tag", "message"),
    [
        ("0.7.0", "v<version>"),
        ("v0.7.1", "source=0.7.0, wheel=0.7.0, sdist=0.7.0, tag=0.7.1"),
    ],
)
def test_release_tag_must_follow_and_match_the_source(tmp_path, tag, message):
    """Test malformed and stale release tags fail with useful context."""
    source_file, distribution_directory = _write_release_files(tmp_path)

    with pytest.raises(ReleaseVersionError, match=message):
        verify_release_version(
            source_file=source_file,
            distribution_directory=distribution_directory,
            tag=tag,
        )


def test_built_metadata_must_match_the_source(tmp_path):
    """Test a stale wheel is identified before publication."""
    source_file, distribution_directory = _write_release_files(tmp_path)
    wheel = next(distribution_directory.glob("*.whl"))
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("example-0.7.0.dist-info/METADATA", _metadata("0.6.0"))

    with pytest.raises(
        ReleaseVersionError,
        match="source=0.7.0, wheel=0.6.0, sdist=0.7.0",
    ):
        verify_release_version(
            source_file=source_file,
            distribution_directory=distribution_directory,
        )


def test_distribution_directory_requires_one_artifact_of_each_kind(tmp_path):
    """Test missing or leftover builds cannot make verification ambiguous."""
    source_file, distribution_directory = _write_release_files(tmp_path)
    next(distribution_directory.glob("*.whl")).unlink()

    with pytest.raises(ReleaseVersionError, match="exactly one wheel; found 0"):
        verify_release_version(
            source_file=source_file,
            distribution_directory=distribution_directory,
        )

"""Verify that source, tag, wheel, and sdist versions agree."""

from __future__ import annotations

import argparse
import ast
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import IO


class ReleaseVersionError(ValueError):
    """Raised when release versions or artifacts do not meet the contract."""


def _read_source_version(source_file: Path) -> str:
    try:
        module = ast.parse(source_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        raise ReleaseVersionError(
            f"Cannot read the source version from {source_file}: {error}"
        ) from error

    versions: list[str] = []
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        if not isinstance(statement.value, ast.Constant) or not isinstance(
            statement.value.value, str
        ):
            raise ReleaseVersionError(
                f"{source_file} must assign __version__ to a string literal"
            )
        versions.append(statement.value.value)

    if len(versions) != 1:
        raise ReleaseVersionError(
            f"{source_file} must contain exactly one __version__ assignment"
        )
    return versions[0]


def _read_metadata_version(metadata: IO[bytes], *, artifact: Path) -> str:
    message = BytesParser(policy=compat32).parse(metadata, headersonly=True)
    version = message.get("Version")
    if version is None or not version.strip():
        raise ReleaseVersionError(f"{artifact} metadata has no Version field")
    return version.strip()


def _read_wheel_version(wheel: Path) -> str:
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_files = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ReleaseVersionError(
                    f"{wheel} must contain exactly one .dist-info/METADATA file"
                )
            with archive.open(metadata_files[0]) as metadata:
                return _read_metadata_version(metadata, artifact=wheel)
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseVersionError(f"Cannot read wheel {wheel}: {error}") from error


def _read_sdist_version(sdist: Path) -> str:
    try:
        with tarfile.open(sdist, mode="r:gz") as archive:
            metadata_files = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith("/PKG-INFO")
            ]
            if len(metadata_files) != 1:
                raise ReleaseVersionError(
                    f"{sdist} must contain exactly one top-level PKG-INFO file"
                )
            metadata = archive.extractfile(metadata_files[0])
            if metadata is None:
                raise ReleaseVersionError(f"Cannot read metadata from {sdist}")
            with metadata:
                return _read_metadata_version(metadata, artifact=sdist)
    except (OSError, tarfile.TarError) as error:
        raise ReleaseVersionError(f"Cannot read sdist {sdist}: {error}") from error


def _single_artifact(paths: list[Path], *, kind: str, directory: Path) -> Path:
    if len(paths) != 1:
        raise ReleaseVersionError(
            f"{directory} must contain exactly one {kind}; found {len(paths)}"
        )
    return paths[0]


def verify_release_version(
    *, source_file: Path, distribution_directory: Path, tag: str | None = None
) -> str:
    """Return the agreed release version or raise with the mismatch details."""
    wheel = _single_artifact(
        sorted(distribution_directory.glob("*.whl")),
        kind="wheel",
        directory=distribution_directory,
    )
    sdist = _single_artifact(
        sorted(distribution_directory.glob("*.tar.gz")),
        kind="source distribution",
        directory=distribution_directory,
    )

    versions = {
        "source": _read_source_version(source_file),
        "wheel": _read_wheel_version(wheel),
        "sdist": _read_sdist_version(sdist),
    }
    if tag is not None:
        if not tag.startswith("v") or len(tag) == 1:
            raise ReleaseVersionError(
                f"Release tag must use the v<version> form; received {tag!r}"
            )
        versions["tag"] = tag.removeprefix("v")

    if len(set(versions.values())) != 1:
        details = ", ".join(f"{name}={version}" for name, version in versions.items())
        raise ReleaseVersionError(f"Release versions do not agree: {details}")
    return versions["source"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify source and distribution versions before publication."
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=Path("src/numpy_vector_store/__init__.py"),
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--tag")
    return parser


def main() -> None:
    """Run release version verification from the command line."""
    parser = _build_parser()
    arguments = parser.parse_args()
    try:
        version = verify_release_version(
            source_file=arguments.source_file,
            distribution_directory=arguments.dist_dir,
            tag=arguments.tag,
        )
    except ReleaseVersionError as error:
        parser.error(str(error))
    print(f"Release version verified: {version}")


if __name__ == "__main__":
    main()

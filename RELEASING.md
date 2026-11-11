# Releasing and repository transfer

This document explains how NumPy Vector Store validates and publishes distributions, and records the ownership and trust boundaries maintainers must preserve.

## Release trust model

The project builds distributions in the same `Checks` workflow that runs its source tests. A successful distribution job produces exactly one wheel and one source distribution, exercises both in isolated environments, and uploads them as one short-lived GitHub Actions artifact.

```text
tagged source commit
        |
        v
Checks: test source, build once, test wheel and sdist
        |
        v
immutable artifact ID for the verified wheel and sdist
        |
        v
top-level publish job: download by ID, then upload with OIDC
```

The PyPI and TestPyPI jobs do not check out source, install build tools, or rebuild the package. They receive only `id-token: write`, download the artifact identified by the completed validation workflow, and pass that directory to `pypa/gh-action-pypi-publish`. This keeps package construction outside the job that can obtain a publishing credential.

Publication stays in `.github/workflows/publish-pypi.yml` and `.github/workflows/publish-testpypi.yml`. PyPI Trusted Publishers identify a top-level workflow by its exact owner, repository, workflow filename, and optional GitHub environment. PyPI does not support using a reusable workflow as the publishing identity. See the [PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) and [PyPI troubleshooting guide](https://docs.pypi.org/trusted-publishers/troubleshooting/).

## What `Checks` proves

Every pull request and push to `main` runs these independent jobs:

- linting, formatting, and static type checks;
- the full suite on supported Python versions 3.11 through 3.14 on Linux;
- the full suite on Python 3.13 on Windows;
- the full suite on Python 3.11 with the declared minimum NumPy 1.23.2;
- a wheel and source-distribution build using the pinned Hatchling backend;
- source and distribution version agreement;
- strict distribution metadata validation with pinned Twine;
- isolated installation and smoke testing of the wheel and source distribution.

Remote Actions use immutable commit references, release tools use exact versions, jobs have time limits, and verified distributions are retained for seven days. The normal source-test jobs still use `uv.lock`; artifact installation deliberately resolves only the distribution and its declared runtime dependencies so it catches missing or incorrect package metadata.

The artifact smoke test checks the installed package rather than the source checkout. It verifies package-version metadata, basic exact search, metadata return values, save and open behavior, and archive contents after reopening. It is a packaging boundary check, not a replacement for the full source test suite.

## Normal release procedure

1. Prepare a release pull request that updates `src/numpy_vector_store/__init__.py`, `CHANGELOG.md`, `ROADMAP.md`, README compatibility text when needed, and any release-specific migration guidance.
2. Confirm that the version follows the intended semantic-versioning scope and that all `Checks` jobs pass on the pull request.
3. Squash merge the release pull request into `main`, then confirm the `main` checks pass.
4. Create a GitHub release targeting the merged commit with a `v<version>` tag and release notes that explain behavior, motivation, compatibility, user impact, and verification.
5. Publish the GitHub release. The release event calls `Checks` again for the tagged commit and supplies the tag to the distribution-version verifier.
6. Confirm that validation finishes before the PyPI job starts, then verify the new version, files, metadata, and attestations on PyPI.

The release workflow requires the source version, wheel metadata, source-distribution metadata, and Git tag to agree. A mismatch stops before the OIDC publishing job. Published PyPI files are immutable; do not delete or replace a release to correct code. Prepare a new semantic version instead.

TestPyPI publication is a separate manual workflow. It builds and validates the selected commit through the same `Checks` workflow, but it has no release tag to compare. Use it for an end-to-end publishing probe when the selected version is not already present on TestPyPI.

## Ownership and publishing environments

The canonical repository is `tvr-works/numpy-vector-store`. The PyPI and TestPyPI projects remain owned by the individual `tvanreenen` account. GitHub organization ownership and Python package index ownership are independent; a PyPI organization, shared account, or additional human owner is not required for this project's solo-maintainer model.

Both indexes trust the GitHub organization repository through environment-bound publishers:

| Index | Workflow | Environment | Allowed ref |
|---|---|---|---|
| TestPyPI | `publish-testpypi.yml` | `testpypi` | `main` branch |
| PyPI | `publish-pypi.yml` | `pypi` | tags matching `v*` |

The environments contain no publishing token or other secret. PyPI exchanges the workflow's GitHub OIDC identity for a short-lived upload token. The environment name is part of that identity and must match the corresponding Trusted Publisher exactly. See [GitHub deployment environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments) and [PyPI Trusted Publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

The environments intentionally have no required reviewer and do not prevent self-review. Requiring another approver would deadlock the current solo maintainer. Their branch and tag policies still prevent an upload job from using an unintended ref.

Do not remove or replace a working publisher record until its replacement has completed an authorized upload. TestPyPI can qualify configuration with an otherwise legitimate unused package version. Production configuration is qualified only by a planned semantic release; never create, delete, replace, or republish a production release solely to probe OIDC configuration.

## Runner boundary

Linux validation and distribution construction use the organization-managed `blacksmith-4vcpu-ubuntu-2404` runner. Windows coverage remains on `windows-latest`. The small OIDC publishing jobs remain on GitHub-hosted `ubuntu-latest`; they only download and upload verified artifacts, so the external build runner never receives the short-lived publishing credential.

If Blacksmith availability changes, move the validation and build jobs back to an explicit GitHub-hosted Ubuntu image in a reviewed pull request. Do not move the OIDC upload jobs to a third-party runner.

## Repository transfer record

The existing repository was transferred from `tvanreenen/numpy-vector-store` to `tvr-works/numpy-vector-store`; it was not copied or recreated. The transfer preserved repository identity and history, while GitHub redirects the old web and Git URLs. Active clones and maintained metadata should use the canonical URL directly:

```bash
git remote set-url origin https://github.com/tvr-works/numpy-vector-store.git
git remote -v
```

Do not recreate the old owner/name combination or depend on its redirect for automation and OIDC identity. PyPI Trusted Publishers must name `tvr-works` directly because OIDC owner identity does not follow redirects.

When auditing the migration, verify the repository ID, default branch, tags, GitHub releases and assets, pull requests, visibility, maintainer access, workflow permissions, environments, publisher records, and a green `main` run under the organization owner. Do not rewrite Git history or rename the Python distribution or import package.

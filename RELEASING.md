# Releasing and repository transfer

This document explains how NumPy Vector Store validates and publishes distributions, then records the staged plan for moving maintenance from `tvanreenen` to `tvr-works`. The transfer steps are a runbook, not a record that the external changes have already happened.

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

## Why environments are deferred

Protected GitHub environments can require reviewer approval, restrict deployment branches or tags, and bind PyPI trust to a more specific identity. They are the intended final state for `testpypi` and `pypi`.

They are not added in the initial CI hardening change. The current workflows do not name environments, and repository inspection cannot confirm the authenticated publisher records stored on PyPI or TestPyPI. Adding `environment: pypi` only in GitHub would change the OIDC claim and could make the next release fail. The environment name must be configured on both sides as one staged change. See [GitHub deployment environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments) and [PyPI Trusted Publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Transfer to `tvr-works`

The GitHub and PyPI transfers are related operationally, but they are separate ownership systems. Moving the GitHub repository does not move the PyPI project, and placing the PyPI project in an organization does not rename `numpy-vector-store`, change imports, or create a package namespace. PyPI organization accounts do not provide namespaces.

### 1. Recheck the baseline

Immediately before the transfer, record the default branch, tags, open pull requests and issues, releases, Actions variables and secrets, environments, rulesets, branch protections, deploy keys, webhooks, installed apps, and current Trusted Publisher records on both PyPI services. This repository had no GitHub environments, Actions secrets or variables, self-hosted runners, branch protections, or rulesets when this runbook was written, but that evidence can become stale.

Confirm that `tvr-works/numpy-vector-store` is available, the organization permits the transfer, the maintainer has permission to create repositories there, and the required teams and outside collaborators are known. Do not rely on redirected URLs for automation or security identities.

### 2. Prepare PyPI organization ownership

Create or obtain approval for the appropriate PyPI organization account and define its owner and maintainer teams. PyPI community organizations may qualify for free service; corporate organizations are billed per active member. Review the current [organization pricing](https://docs.pypi.org/organization-accounts/pricing-and-payments/) before registration.

An existing project owner can transfer `numpy-vector-store` from the individual account to the organization without changing its project name, release history, install command, or import path. Follow PyPI's [project transfer procedure](https://docs.pypi.org/organization-accounts/actions/project-actions/) and keep at least two trusted human owners during the transition.

This PyPI ownership move can be completed before the GitHub transfer because it does not change the GitHub OIDC claim. Confirm team permissions and the existing publisher record after the project move.

### 3. Transfer the GitHub repository

Transfer the repository through GitHub's documented ownership flow, preserving its name and full Git history. GitHub redirects old repository web and Git URLs after a transfer, but update active clones and integrations to the new URL instead of treating the redirect as permanent configuration. Follow the [GitHub repository transfer documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository).

After the transfer, verify the default branch, tags, releases, issues, pull requests, Actions history, collaborator access, organization policy, and workflow permissions before merging new work. Run the nonpublishing `Checks` workflow on `tvr-works/numpy-vector-store` and resolve organization-policy or runner failures before attempting a package upload.

### 4. Rebind Trusted Publishers

OIDC does not follow GitHub's repository redirect. Update the Trusted Publisher records on both PyPI and TestPyPI so their owner is `tvr-works`, their repository is `numpy-vector-store`, and their workflow filenames remain `publish-pypi.yml` and `publish-testpypi.yml`.

Do this after the GitHub transfer and before the next upload. Confirm each field exactly; owner, repository, workflow filename, and environment matching is case-sensitive and exact. Keep the first post-transfer publisher records without an environment if that matches the current working records. This restores the existing trust shape with only the required owner change.

### 5. Update repository references

In a normal reviewed pull request, replace maintained `github.com/tvanreenen/numpy-vector-store` links with `github.com/tvr-works/numpy-vector-store`. At minimum, inspect `pyproject.toml`, the README clone command, changelog and release links, badges, funding configuration, and external service settings. Do not rewrite historical Git commits or rename the package.

Update local remotes explicitly:

```bash
git remote set-url origin https://github.com/tvr-works/numpy-vector-store.git
git remote -v
```

### 6. Add protected publishing environments

Make the environment migration after the transferred repository can run `Checks` and after its owner-only Trusted Publisher records are confirmed.

1. Create `testpypi` and `pypi` environments in GitHub.
2. Restrict their deployment branches or tags and configure required reviewers. For production, prevent self-review when the available maintainer model supports it.
3. Add new Trusted Publisher records on TestPyPI and PyPI with the matching environment names. Keep the working no-environment records during the transition.
4. Update the corresponding workflow jobs to declare `environment: testpypi` and `environment: pypi`.
5. Validate the complete path on TestPyPI first.
6. Use the protected production environment for the next planned release, verify the upload, then remove superseded no-environment publisher records.

Do not remove a working publisher record before its replacement has completed a real upload. Do not use a production release solely as a configuration probe.

### 7. Adopt `tvr-works` runners

The personal repository remains on GitHub-hosted runners because it has no confirmed Blacksmith runner access. After transfer, first run a harmless organization-level probe that proves the expected label is available to this repository.

Once verified, change eligible Linux validation and distribution-build jobs from `ubuntu-latest` to the established `tvr-works` label `blacksmith-4vcpu-ubuntu-2404`. Keep Windows coverage on `windows-latest`. Keep the small OIDC publishing jobs on `ubuntu-latest`; they only download and upload artifacts, so moving them would add an external runner to the credential-bearing boundary without a useful performance benefit.

Change runner labels in their own pull request. The build commands and checks should remain the same so any behavior or timing difference can be attributed to runner infrastructure.

## Transfer completion criteria

The transfer is complete when all of the following are true:

- the canonical GitHub repository is `tvr-works/numpy-vector-store` and normal checks pass there;
- the existing PyPI project remains named `numpy-vector-store`, with its release history intact and team ownership confirmed;
- PyPI and TestPyPI Trusted Publisher records match the transferred repository and top-level workflows;
- protected `testpypi` and `pypi` environments have completed their staged validation;
- maintained repository links and integrations use the new GitHub owner;
- eligible Linux jobs pass on the verified organization runner, while Windows and OIDC upload jobs retain their intended runners;
- obsolete publisher records and temporary transfer access have been removed only after their replacements are proven.

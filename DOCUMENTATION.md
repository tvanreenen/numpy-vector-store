# Documentation development

The documentation is a static Zensical site in the library repository. It has
no service, authentication, paid search, JavaScript package manager, or runtime
Python dependency beyond the library's existing NumPy requirement. Documentation
tools and the published package used by the site are pinned independently in
`tools/docs-env/pyproject.toml` and its `uv.lock`.

## How the pieces fit

Library development and public documentation need different copies of NVS.
Library tests use the code being developed; the public reference and guide
examples use the published release that readers can install.

`tools/docs-env/` is the documentation's toolbox. Its `pyproject.toml` lists
the released NVS package and documentation tools, and its `uv.lock` records
their exact dependencies and artifact hashes. uv installs them into a generated,
ignored `.venv/` inside that folder. This is a build-only environment, not
another library to publish. The documentation itself still lives in `docs/`.

`tools/docs.py` supplies the build instructions. It checks that the installed
package matches the documented release, builds the site, checks links and
metadata, and runs the displayed examples. Its `serve` command previews the
site locally; its `pages` command also checks publication eligibility and
prepares preview indexing rules.

Your laptop, GitHub Actions, and Cloudflare use the same toolbox and script.
The build writes static files to `site/`. Visitors receive those files; neither
Python nor the documentation environment runs when they visit the website.

## Build and preview

```sh
uv sync --project tools/docs-env --locked
uv run --project tools/docs-env --locked python tools/docs_tests.py
uv run --project tools/docs-env --locked python tools/docs.py check
uv run --project tools/docs-env --locked python tools/docs.py serve
```

The preview listens on `http://127.0.0.1:8000`. Its environment is
`tools/docs-env/.venv`, separate from the library's development environment.
`site/` is generated and ignored. After changing configuration or theme overrides,
restart the live server if necessary. Run `tools/docs.py check` for a clean
publication build; the development server is not the verification gate.

`check` verifies the edition, runs a clean strict build, checks all rendered
local links and anchors, inspects public API rendering and search coverage,
checks titles/descriptions/canonical/Open Graph URLs and sitemap/robots, and
executes the Python examples extracted from the built guide HTML. Their printed
outputs must match the outputs displayed on the page. Old migration examples
are intentionally not executed against the current API. External links are not
network-checked in CI. Installing the locked environment needs access to the
pinned artifacts unless they are already cached. Installation failures stop the
build; there is no fallback to checkout source or another release.

With uv installed, `python tools/docs.py check` and `python tools/docs.py serve`
also enter the isolated environment automatically. The launcher removes inherited
Python import paths and environment-location overrides before invoking the
locked docs project. Documentation builds do not install the root library
project or its development dependencies. There is no root `docs` dependency
group; select `tools/docs-env` when invoking uv directly.

## Content and canonical sources

| Content | Canonical source | How the site uses it |
| --- | --- | --- |
| Introduction and guides | `docs/*.md` | Authored Markdown |
| Quick start | `README.md`, `quickstart` snippet | Included on the home page; executed from rendered HTML |
| Reference measurements | `README.md`, `performance` snippet | Included in `docs/performance.md` without copying numbers |
| Guide examples | `docs/examples/*.py` | Included in guide code fences; executed from rendered HTML |
| Supported API signatures/docstrings | Verified PyPI wheel pinned in `tools/docs-env/` | Installed-package discovery with an explicit member allowlist in `docs/reference.md` |
| Compatibility | `COMPATIBILITY.md` | Snippet wrapper at `docs/COMPATIBILITY.md` |
| Historical migration | `MIGRATION.md` | Snippet wrapper at `docs/MIGRATION.md` |
| Release history | `CHANGELOG.md` | Introduction and published-history snippets at `docs/CHANGELOG.md` |
| Benchmark methodology | `benchmarks/README.md` | Snippet wrapper at `docs/benchmarks/README.md` |
| Theme and metadata | `zensical.toml`, `docs/assets/`, `docs/overrides/` | Native modern theme plus small CSS/template overrides |

Uppercase policy/history paths preserve the relative links shared by the
repository documents. Existing GitHub file links remain useful. The documentation
first appeared at `https://nvs.tvr.works/`; that address redirects to the
canonical `https://vectors.tvr.works/` site and should remain available for
existing links. The README remains a standalone package introduction,
installation guide, executable quick start, benchmark record, and usage summary
for GitHub and PyPI. Detailed guide prose belongs in `docs/`; summaries in the
README should stay short.

The built site includes only the curated `docs/` tree. Theme override sources
are excluded. Maintainer release, ownership-transfer, and roadmap notes are not
copied into the site. Docs source, output, tooling, and this contributor guide
are excluded from the source distribution, and the wheel remains package-only.
`pyproject.toml` remains in the source distribution as build metadata. The
independent docs project and its lockfile are excluded; docs tools are not
runtime dependencies of the library.

## Edition and framework qualification

The public edition describes the published **0.7.0** release. The docs project
pins its exact PyPI wheel URL; the independent lockfile records the wheel's
SHA-256 hash and the resolved documentation dependencies. `docs_version`,
`docs_status = "released"`, and `api_source_sha256` in `zensical.toml` record the
expected package identity. The source digest covers each filename, a NUL byte,
and its file bytes for `__init__.py`, `vector_store.py`, and `py.typed`, in that
order.

The build verifies the installed package version and source digest and rejects
checkout imports or explicit source paths in the API renderer. Reference
generation and rendered examples use that same installed release. Changing the
development source or version does not advance or invalidate the public edition.
Library checks continue to exercise the checkout independently.

Keep public prose, shared README snippets, policy/history documents, and examples
aligned with the pinned release. Examples test executable claims, but cannot
prove every statement in prose. Do not include unreleased behavior in these
sources merely because the development implementation supports it. The public
build does not accept a development edition or automatically select the latest
package from PyPI.

Release preparation can add a new entry to the repository changelog above its
`published` snippet. The website includes only the introduction and the history
inside that snippet, so preparing release notes does not announce an unpublished
package on the released site. Keep the snippet's opening marker immediately
before the newest release covered by the public documentation edition.

After an independently authorized release has uploaded and passed verification:

1. Update the exact version and wheel URL in `tools/docs-env/pyproject.toml`.
2. Regenerate its lockfile with `uv lock --project tools/docs-env` and compare
   the wheel hash with PyPI's published metadata.
3. Verify the wheel's package source, then update `docs_version` and
   `api_source_sha256`. Keep `docs_status = "released"`.
4. Review release-specific wording in the guides, reference, README, changelog,
   and compatibility policy. Move the changelog's `published` opening marker
   above the newly verified release, and mark its roadmap entry complete. Run
   the docs tooling tests and full build before merging the edition update.

The previous edition remains public until this update is deployed. Roll back an
edition with its matching pin, lockfile, digest, and prose, not by pointing an
old label at a different package. No historical-version selector is required.

The qualified tools are **Zensical 0.1.1** and **mkdocstrings-python 2.0.10**.
The real landing page and public reference passed a strict static build, API
deep-link checks, and a client-side search interaction before the other pages
were added. Zensical renders meaningful static HTML and a local `search.json`;
search requires no remote service. See the official
[mkdocstrings compatibility notes](https://zensical.org/docs/compatibility/mkdocs/plugins/#mkdocstrings),
[strict build documentation](https://zensical.org/docs/usage/build/), and
[search documentation](https://zensical.org/docs/setup/search/).

The modern theme is the design baseline. System fonts avoid external font
requests. A plain repository-link override avoids live GitHub statistics and
latest-tag requests, which could otherwise disagree with the labeled edition.
The small logo is an SVG asset. Search, code-copy behavior, palettes, navigation,
and keyboard interactions use the theme's existing components.

## Documentation CI

`.github/workflows/docs.yml` is independent of the library checks and publishing
workflows. It installs the independent locked docs project and published package,
tests the publication gate offline, builds the site strictly, verifies the
generated content, executes guide examples, and retains the verified static output as an
artifact. Remote Actions use the same pinned revisions as the existing checks.

It runs on PRs and every branch push. This is intentionally a small docs-only
workflow without path filters: a future Pages build of any source revision can
always require successful CI for that exact SHA. A package-only edit does not
change the public API reference. The library release workflows do not call the
docs workflow; browser checks and external-link checks do not become package
release gates.

For visual changes, separately test Chromium and WebKit at desktop and mobile
widths, a 320px reflow width, 200% zoom, light/dark/system palettes, keyboard
navigation, search, empty search, code copying, deep links, and 404 recovery.
Check focus, contrast, reduced motion, console errors, and horizontal overflow.
Browser QA is local for this first milestone; there is no committed browser
dependency or flaky browser step in package CI.

## Cloudflare Pages publication

No hosting project, DNS record, GitHub secret, package publisher, or account
setting is created by the local build. The canonical URL is
`https://vectors.tvr.works/`. If it changes, update `site_url` in `zensical.toml`
and the sitemap URL in `docs/robots.txt`, then rerun the checks.

The Cloudflare Pages project is `nvs`, with the provider hostname
`nvs-1nz.pages.dev`. It connects to the existing public
`tvr-works/numpy-vector-store` repository through the Cloudflare GitHub app.
The app must have access to this repository for automatic push deployments.
Use these settings in both production and preview environments:

| Setting | Value |
| --- | --- |
| Framework preset | None |
| Root directory | Repository root |
| Production branch | `main` |
| Preview branches | All non-production branches in this repository |
| Build output directory | `site` |
| Python environment variable | `PYTHON_VERSION=3.13.5` |
| Dependency-install environment variable | `SKIP_DEPENDENCY_INSTALL=1` |
| Build command | Command below |

```sh
python -m pip install uv==0.13.7 && python -m uv run --project tools/docs-env --locked python tools/docs.py pages
```

Pages starts directly in the documentation environment, without first installing
the root library project. `uv run --locked` installs or reuses the locked
documentation dependencies before running the script; a separate sync step is
not needed.
Invoke uv through the same Python interpreter that installed it: newly installed
executables may not appear on the Pages shell's PATH until its shims refresh.

The build command is a project-wide setting shared by production and previews.
When adopting this command, first push a branch containing `tools/docs-env/`,
then update the Pages setting and verify that branch's exact-commit preview.
Merge that branch to make the command work for subsequent `main` builds too.
Between the setting change and merge, builds of older revisions without the
docs project will fail; the existing deployed website stays available. Rebuilding
an older revision requires restoring its matching build command. Rolling back
to an existing successful deployment does not require rebuilding it.

Pages supplies `CF_PAGES_COMMIT_SHA` and `CF_PAGES_BRANCH`. The `pages` command
checks the checkout SHA and tracked cleanliness, then polls the public GitHub
workflow-runs endpoint for `docs.yml` with matching SHA, branch, repository, and
`push` event. Only the newest matching run/attempt may authorize a build, and it
must finish with `success`. An older success cannot hide a newer failure or
pending rerun. Missing context, timeout, cancellation, skipped CI, HTTP errors,
rate limits, and malformed responses fail closed. The timeout is ten minutes;
retry a Pages build after CI is green or a public API rate limit has cleared.

Using only a successful local Pages rebuild would leave GitHub CI independent
of publication. This gate first requires the exact-source CI result and then
reruns the deterministic docs checks in Pages. It uses no GitHub credential and
requires no Cloudflare token in GitHub. Fork PRs can run docs CI, but cannot
substitute for a same-repository branch push when authorizing a Pages preview.

The generated top-level `404.html` supports Pages' normal static 404 behavior.
Production canonical tags always point to the canonical documentation domain.
The `pages` command adds `X-Robots-Tag: noindex, nofollow` and a disallowing
`robots.txt` to branch previews. Production keeps the indexed robots/sitemap.

After a hosting, branch-control, or custom-domain change, verify a main
production build, a same-repository branch preview, the live exact-SHA gate,
preview noindex headers, and custom-domain/DNS behavior. These hosted acceptance
checks are outside the local build. Use the pinned static artifact or an earlier
known-good Pages deployment for rollback; package publishing remains separate.

Cloudflare documents the relevant
[build variables and output configuration](https://developers.cloudflare.com/pages/configuration/build-configuration/),
[Git integration](https://developers.cloudflare.com/pages/configuration/git-integration/),
and [branch controls](https://developers.cloudflare.com/pages/configuration/branch-build-controls/).
The gate uses GitHub's public
[workflow-runs API](https://docs.github.com/en/rest/actions/workflow-runs#list-workflow-runs-for-a-workflow).

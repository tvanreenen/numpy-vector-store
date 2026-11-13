# Documentation development

The documentation is a static Zensical site in the library repository. It has
no service, authentication, paid search, JavaScript package manager, or runtime
Python dependency beyond the library's existing NumPy requirement. Documentation
tools are pinned in the `docs` dependency group and `uv.lock`.

## Build and preview

```sh
uv sync --locked --no-default-groups --group docs
uv run --locked --no-default-groups --group docs python tools/docs_tests.py
uv run --locked --no-default-groups --group docs python tools/docs.py check
uv run --locked --no-default-groups --group docs zensical serve
```

The preview listens on `http://127.0.0.1:8000`. For an isolated environment,
prefix the commands with `UV_PROJECT_ENVIRONMENT=.venv-docs`. `site/` is generated
and ignored. After changing configuration or theme overrides, restart the live
server if necessary. Run `tools/docs.py check` for a clean publication build;
the development server is not the verification gate.

`check` verifies the edition, runs a clean strict build, checks all rendered
local links and anchors, inspects public API rendering and search coverage,
checks titles/descriptions/canonical/Open Graph URLs and sitemap/robots, and
executes the Python examples extracted from the built guide HTML. Their printed
outputs must match the outputs displayed on the page. Old migration examples
are intentionally not executed against the current API. External links are not
network-checked in CI.

## Content and canonical sources

| Content | Canonical source | How the site uses it |
| --- | --- | --- |
| Introduction and guides | `docs/*.md` | Authored Markdown |
| Quick start | `README.md`, `quickstart` snippet | Included on the home page; executed from rendered HTML |
| Reference measurements | `README.md`, `performance` snippet | Included in `docs/performance.md` without copying numbers |
| Guide examples | `docs/examples/*.py` | Included in guide code fences; executed from rendered HTML |
| Supported API signatures/docstrings | `src/numpy_vector_store/` | mkdocstrings directives with an explicit member allowlist in `docs/reference.md` |
| Compatibility | `COMPATIBILITY.md` | Snippet wrapper at `docs/COMPATIBILITY.md` |
| Historical migration | `MIGRATION.md` | Snippet wrapper at `docs/MIGRATION.md` |
| Release history | `CHANGELOG.md` | Snippet wrapper at `docs/CHANGELOG.md` |
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
`pyproject.toml` and its docs dependency-group declaration remain in the source
distribution as build metadata; docs dependencies are not `Requires-Dist`.

## Edition and framework qualification

This first edition describes the published **0.7.0** release. Its Python source
(`__init__.py`, `vector_store.py`, and `py.typed`) was compared byte-for-byte
with the public PyPI 0.7.0 wheel. The `docs_version`, `docs_status`, and
`api_source_sha256` settings record that boundary. A released build fails if the
package version or verified source changes, preventing unreleased code from
silently appearing under a stable label.

For a future development edition, set `docs_status = "development"`, update
`docs_version` to the source version, and revise version claims in the home page,
getting-started page, reference, and persistence guide. For the next released
edition, verify the published wheel again before updating the source digest and
returning the status to `released`. The digest is SHA-256 over each filename,
a NUL byte, and its file bytes, in the order listed above. No historical-version
selector is needed for this first edition.

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
workflows. It installs only the docs group plus the local package, tests the
publication gate offline, builds the site strictly, verifies the generated
content, executes guide examples, and retains the verified static output as an
artifact. Remote Actions use the same pinned revisions as the existing checks.

It runs on PRs and every branch push. This is intentionally a small docs-only
workflow without path filters: a future Pages build of any source revision can
always require successful CI for that exact SHA. A package-only edit can change
API rendering too. The library release workflows do not call the docs workflow;
browser checks and external-link checks do not become package release gates.

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
python -m pip install uv==0.13.7 && python -m uv sync --locked --no-default-groups --group docs && python -m uv run --locked --no-default-groups --group docs python tools/docs.py pages
```

Disabling automatic dependency installation lets the build command install the
pinned uv version before reading the lockfile and install only the docs group
plus the local package.
Invoke uv through the same Python interpreter that installed it: newly installed
executables may not appear on the Pages shell's PATH until its shims refresh.

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

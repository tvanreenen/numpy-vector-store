"""Build and verify the documentation without changing package behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS_PROJECT = ROOT / "tools" / "docs-env"
EXAMPLE_PAGES = (
    "index.md",
    "getting-started.md",
    "search.md",
    "persistence.md",
    "ownership.md",
    "deployment.md",
)
PUBLIC_MEMBERS = (
    "dimensions",
    "normalize",
    "file_path",
    "vectors",
    "metadata",
    "open",
    "add",
    "reload",
    "save",
    "cosine_search",
    "dot_search",
    "euclidean_search",
    "get",
    "clear",
    "__len__",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass
class Page(HTMLParser):
    """Read only the HTML contracts checked by this static-site verifier."""

    ids: set[str] = field(default_factory=set)
    links: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    canonical: str = ""
    title: str = ""
    text: str = ""
    h1_count: int = 0
    blocks: list[tuple[str, str]] = field(default_factory=list)
    _article: bool = False
    _title: bool = False
    _pre: bool = False
    _language: str = ""
    _code: str = ""

    def __post_init__(self) -> None:
        HTMLParser.__init__(self, convert_charrefs=True)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        identifier = attr.get("id")
        if identifier:
            require(identifier not in self.ids, f"Duplicate HTML id: {identifier}")
            self.ids.add(identifier)
        for key in ("href", "src"):
            if attr.get(key):
                self.links.append(str(attr[key]))
        if tag == "meta":
            self.meta[str(attr.get("name") or attr.get("property"))] = str(
                attr.get("content", "")
            )
        if tag == "link" and attr.get("rel") == "canonical":
            self.canonical = str(attr.get("href"))
        if tag == "title":
            self._title = True
        if tag == "article":
            self._article = True
        if self._article:
            self.h1_count += tag == "h1"
            classes = str(attr.get("class", "")).split()
            for value in classes:
                if value.startswith("language-"):
                    self._language = value.removeprefix("language-")
            if tag == "pre":
                self._pre = True
                self._code = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._title = False
        if tag == "article":
            self._article = False
        if tag == "pre" and self._pre:
            self.blocks.append((self._language, self._code))
            self._pre = False
            self._language = ""

    def handle_data(self, data: str) -> None:
        if self._title:
            self.title += data
        if self._article:
            self.text += data
        if self._pre:
            self._code += data


def read_page(path: Path) -> Page:
    page = Page()
    page.feed(path.read_text())
    return page


def verify_edition() -> None:
    """Only the pinned installed release may supply public reference and examples."""
    import numpy_vector_store

    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    extra = config["extra"]
    require(
        config["plugins"]["mkdocstrings"]["handlers"]["python"]["paths"] == [],
        "API rendering must use the installed release, not explicit source paths",
    )
    require(
        extra["docs_version"]
        == numpy_vector_store.__version__
        == version("numpy-vector-store"),
        "Documentation and installed release versions do not agree",
    )
    require(
        extra["docs_status"] == "released", "Public docs require a released edition"
    )
    require(numpy_vector_store.__file__ is not None, "Package has no source file")
    package = Path(str(numpy_vector_store.__file__)).resolve().parent
    require(
        package == Path(sysconfig.get_path("purelib")).resolve() / "numpy_vector_store",
        f"Documentation must use the installed release, not checkout source: {package}",
    )
    digest = hashlib.sha256()
    for name in ("__init__.py", "vector_store.py", "py.typed"):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((package / name).read_bytes())
    require(
        digest.hexdigest() == extra["api_source_sha256"],
        "Installed package source differs from the verified published release",
    )
    print(
        f"Documentation edition: {extra['docs_version']} ({extra['docs_status']}) from {package}"
    )


def verify_site() -> None:
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    base = config["site_url"]
    pages = {
        path.relative_to(SITE).as_posix(): read_page(path)
        for path in SITE.rglob("*.html")
    }
    require("index.html" in pages and "404.html" in pages, "Missing home or 404 page")
    for filename, page in pages.items():
        route = filename.removesuffix("index.html")
        require(
            bool(page.title.strip()) and bool(page.meta.get("description")),
            f"Missing title/description: {filename}",
        )
        require(
            bool(page.meta.get("og:title")) and bool(page.meta.get("og:description")),
            f"Missing Open Graph metadata: {filename}",
        )
        require(
            "--8<--" not in page.text and "::: numpy_vector_store" not in page.text,
            f"Unrendered directive: {filename}",
        )
        if filename != "404.html":
            require(
                page.h1_count == 1 and len(page.text.strip()) > 300,
                f"Missing crawlable article: {filename}",
            )
            require(
                page.canonical == urljoin(base, route),
                f"Incorrect canonical URL: {filename}",
            )
            require(
                page.meta.get("og:url") == page.canonical,
                f"Incorrect Open Graph URL: {filename}",
            )
        else:
            require("noindex" in page.meta.get("robots", ""), "404 must be noindex")
        for link in page.links:
            url = urlsplit(urljoin(urljoin(base, route), link))
            if (
                url.scheme not in ("http", "https")
                or url.netloc != urlsplit(base).netloc
            ):
                continue
            relative = unquote(url.path).lstrip("/")
            destination = SITE / relative
            if destination.is_dir():
                destination /= "index.html"
            require(destination.is_file(), f"Broken local link in {filename}: {link}")
            if url.fragment and destination.suffix == ".html":
                target = destination.relative_to(SITE).as_posix()
                require(
                    unquote(url.fragment) in pages[target].ids,
                    f"Broken anchor in {filename}: {link}",
                )
    reference = pages["reference/index.html"]
    identifiers = {"numpy_vector_store.VectorStore", "numpy_vector_store.VectorHit"}
    identifiers.update(
        f"numpy_vector_store.VectorStore.{name}" for name in PUBLIC_MEMBERS
    )
    identifiers.update(
        f"numpy_vector_store.VectorHit.{name}"
        for name in ("index", "value", "metadata")
    )
    require(
        identifiers <= reference.ids,
        f"Missing supported API: {identifiers - reference.ids}",
    )
    actual = {name for name in reference.ids if name.startswith("numpy_vector_store.")}
    require(actual == identifiers, f"Unexpected API exposed: {actual - identifiers}")
    index = json.loads((SITE / "search.json").read_text())
    locations = {item["location"] for item in index["items"]}
    require(
        all(f"reference/#{identifier}" in locations for identifier in identifiers),
        "API missing from search index",
    )
    require(
        all(
            filename.removesuffix("index.html") in locations
            for filename in pages
            if filename != "404.html"
        ),
        "A page is missing from search",
    )
    sitemap = ElementTree.parse(SITE / "sitemap.xml")
    urls = {element.text for element in sitemap.iter() if element.tag.endswith("}loc")}
    expected = {
        urljoin(base, name.removesuffix("index.html"))
        for name in pages
        if name != "404.html"
    }
    require(urls == expected, "Sitemap does not match public pages")
    require(
        f"Sitemap: {base}sitemap.xml" in (SITE / "robots.txt").read_text(),
        "Incorrect robots sitemap",
    )
    for path in SITE.rglob("*"):
        require(
            not set(path.relative_to(SITE).parts)
            & {"overrides", "tools", "RELEASING.md", "ROADMAP.md", ".git", ".github"},
            f"Non-public content emitted: {path}",
        )
    print(
        f"Verified {len(pages)} HTML pages, local links/anchors, API search, metadata, sitemap, and robots"
    )


def verify_examples() -> None:
    count = 0
    with TemporaryDirectory(prefix="nvs-docs-examples-") as directory:
        for filename in EXAMPLE_PAGES:
            html = (
                "index.html"
                if filename == "index.md"
                else filename.removesuffix(".md") + "/index.html"
            )
            blocks = read_page(SITE / html).blocks
            for index, (language, source) in enumerate(blocks):
                if language != "python":
                    continue
                expected = ""
                if index + 1 < len(blocks) and blocks[index + 1][0] == "text":
                    expected = blocks[index + 1][1]
                result = subprocess.run(
                    [sys.executable, "-c", source],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                require(
                    result.returncode == 0,
                    f"{filename} example failed:\n{result.stderr}",
                )
                require(
                    result.stdout.strip() == expected.strip(),
                    f"{filename} output differs:\nExpected {expected!r}\nGot {result.stdout!r}",
                )
                require(
                    not result.stderr,
                    f"{filename} example emitted warnings: {result.stderr}",
                )
                count += 1
    require(count >= 6, "Expected runnable examples on all authored guide pages")
    print(f"Executed {count} rendered Python examples; displayed outputs match")


def run_checks() -> None:
    verify_edition()
    subprocess.run(
        [sys.executable, "-m", "zensical", "build", "--clean", "--strict"],
        cwd=ROOT,
        check=True,
    )
    verify_site()
    verify_examples()


def prepare_pages_output(site: Path, branch: str) -> None:
    """Prevent search indexing of branch previews without changing canonicals."""
    if branch != "main":
        (site / "_headers").write_text("/*\n  X-Robots-Tag: noindex, nofollow\n")
        (site / "robots.txt").write_text("User-agent: *\nDisallow: /\n")


def enter_docs_environment(command: str) -> None:
    """Keep legacy callers working without importing their editable package."""
    if Path(sys.prefix).resolve() != (DOCS_PROJECT / ".venv").resolve():
        environment = os.environ.copy()
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "UV_PROJECT_ENVIRONMENT",
            "UV_ISOLATED",
            "UV_NO_SYNC",
            "UV_FROZEN",
            "UV_NO_PROJECT",
        ):
            environment.pop(key, None)
        uv = os.environ.get("UV") or shutil.which("uv")
        require(uv is not None, "Install uv before building documentation")
        result = subprocess.run(
            [
                str(uv),
                "run",
                "--project",
                str(DOCS_PROJECT),
                "--locked",
                "--no-env-file",
                "python",
                str(Path(__file__).resolve()),
                command,
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "site", "examples", "pages", "serve"),
        default="check",
        nargs="?",
    )
    args = parser.parse_args()
    enter_docs_environment(args.command)
    if args.command == "serve":
        verify_edition()
        subprocess.run(
            [sys.executable, "-m", "zensical", "serve"], cwd=ROOT, check=True
        )
        return
    if args.command == "site":
        verify_site()
    elif args.command == "examples":
        verify_edition()
        verify_examples()
    else:
        if args.command == "pages":
            from docs_gate import wait_for_docs_ci

            sha = os.environ.get("CF_PAGES_COMMIT_SHA", "")
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            require(sha == head, "Pages commit does not match the local checkout")
            require(
                not subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=ROOT,
                    text=True,
                ).strip(),
                "Pages checkout has tracked changes",
            )
            wait_for_docs_ci(sha, os.environ.get("CF_PAGES_BRANCH", ""))
        run_checks()
        if args.command == "pages":
            prepare_pages_output(SITE, os.environ["CF_PAGES_BRANCH"])


if __name__ == "__main__":
    main()

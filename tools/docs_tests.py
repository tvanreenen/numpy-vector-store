"""Offline tests of the documentation publication boundary and HTML checks."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tomllib
import unittest
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from docs_gate import REPOSITORY, latest_run, wait_for_docs_ci

from docs import (
    DOCS_PROJECT,
    ROOT,
    Page,
    enter_docs_environment,
    prepare_pages_output,
    verify_edition,
)

SHA = "a" * 40


def run_record(**overrides: Any) -> dict[str, Any]:
    return {
        "id": 1,
        "run_number": 1,
        "run_attempt": 1,
        "head_sha": SHA,
        "head_branch": "main",
        "event": "push",
        "head_repository": {"full_name": REPOSITORY},
        "status": "completed",
        "conclusion": "success",
        **overrides,
    }


def response(*runs: dict[str, Any]) -> io.BytesIO:
    return io.BytesIO(json.dumps({"workflow_runs": runs}).encode())


class DocsGateTests(unittest.TestCase):
    def test_exact_source_success_uses_scoped_public_request(self) -> None:
        with patch("docs_gate.urlopen", return_value=response(run_record())) as fetch:
            wait_for_docs_ci(SHA, "main")
        request = fetch.call_args.args[0]
        url = urlsplit(request.full_url)
        self.assertEqual(
            url.path, f"/repos/{REPOSITORY}/actions/workflows/docs.yml/runs"
        )
        self.assertEqual(
            parse_qs(url.query),
            {
                "head_sha": [SHA],
                "branch": ["main"],
                "event": ["push"],
                "per_page": ["100"],
            },
        )
        self.assertNotIn("Authorization", request.headers)

    def test_invalid_context_makes_no_request(self) -> None:
        with patch("docs_gate.urlopen") as fetch:
            for sha, branch, timeout in (
                ("abc", "main", 1),
                (SHA, "", 1),
                (SHA, " main", 1),
                (SHA, "main", 0),
            ):
                with (
                    self.subTest(sha=sha, branch=branch, timeout=timeout),
                    self.assertRaises(ValueError),
                ):
                    wait_for_docs_ci(sha, branch, timeout=timeout)
            fetch.assert_not_called()

    def test_other_sha_branch_event_or_repository_cannot_substitute(self) -> None:
        for change in (
            {"head_sha": "b" * 40},
            {"head_branch": "other"},
            {"event": "pull_request"},
            {"head_repository": {"full_name": "other/fork"}},
        ):
            with self.subTest(change=change):
                self.assertIsNone(latest_run([run_record(**change)], SHA, "main"))

    def test_newer_failure_or_rerun_cannot_hide_behind_older_success(self) -> None:
        for change in ({"run_number": 2}, {"run_attempt": 2}):
            runs = [run_record(), run_record(**change, conclusion="failure")]
            for ordered in (runs, runs[::-1]):
                with (
                    self.subTest(change=change),
                    patch("docs_gate.urlopen", return_value=response(*ordered)),
                    self.assertRaisesRegex(RuntimeError, "refusing deployment"),
                ):
                    wait_for_docs_ci(SHA, "main")

    def test_pending_then_success_waits(self) -> None:
        with (
            patch(
                "docs_gate.urlopen",
                side_effect=[
                    response(run_record(status="in_progress", conclusion=None)),
                    response(run_record()),
                ],
            ) as fetch,
            patch("docs_gate.time.sleep") as sleep,
        ):
            wait_for_docs_ci(SHA, "main")
            self.assertEqual(fetch.call_count, 2)
            sleep.assert_called_once()

    def test_absent_run_times_out(self) -> None:
        with (
            patch("docs_gate.time.monotonic", side_effect=[0, 0, 0, 2, 2]),
            patch("docs_gate.urlopen", return_value=response()),
            self.assertRaises(TimeoutError),
        ):
            wait_for_docs_ci(SHA, "main", timeout=1)

    def test_non_success_conclusions_fail_closed(self) -> None:
        for conclusion in (
            "failure",
            "cancelled",
            "skipped",
            "timed_out",
            "neutral",
            "action_required",
            None,
        ):
            with (
                self.subTest(conclusion=conclusion),
                patch(
                    "docs_gate.urlopen",
                    return_value=response(run_record(conclusion=conclusion)),
                ),
                self.assertRaisesRegex(RuntimeError, "refusing deployment"),
            ):
                wait_for_docs_ci(SHA, "main")

    def test_http_and_malformed_responses_fail_closed(self) -> None:
        with (
            patch(
                "docs_gate.urlopen",
                side_effect=HTTPError(
                    "https://api.github.com", 403, "Rate limited", Message(), None
                ),
            ),
            self.assertRaises(HTTPError),
        ):
            wait_for_docs_ci(SHA, "main")
        with (
            patch("docs_gate.urlopen", return_value=io.BytesIO(b"{}")),
            self.assertRaisesRegex(RuntimeError, "refusing deployment"),
        ):
            wait_for_docs_ci(SHA, "main")


class ReleasedEditionTests(unittest.TestCase):
    def test_installed_release_matches_published_identity(self) -> None:
        verify_edition()

    def test_unreleased_checkout_source_does_not_change_the_public_edition(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / "zensical.toml").write_text(
                (ROOT / "zensical.toml").read_text()
            )
            package = checkout / "src" / "numpy_vector_store"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                '__version__ = "99.0.0"\nraise RuntimeError("unreleased source imported")\n'
            )
            with patch("docs.ROOT", checkout):
                verify_edition()

    def test_version_status_source_and_renderer_mismatches_are_rejected(self) -> None:
        original = tomllib.loads((ROOT / "zensical.toml").read_text())
        for key, value, message in (
            ("docs_version", "99.0.0", "versions do not agree"),
            ("docs_status", "development", "released edition"),
            ("api_source_sha256", "0" * 64, "differs from the verified"),
            ("paths", ["src"], "explicit source paths"),
        ):
            config = json.loads(json.dumps(original))
            if key == "paths":
                config["project"]["plugins"]["mkdocstrings"]["handlers"]["python"][
                    key
                ] = value
            else:
                config["project"]["extra"][key] = value
            with (
                self.subTest(key=key),
                patch("docs.tomllib.loads", return_value=config),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                verify_edition()

    def test_checkout_import_is_rejected_even_when_source_and_version_match(
        self,
    ) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", "from docs import verify_edition; verify_edition()"],
            cwd=ROOT / "tools",
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not checkout source", result.stderr)

    def test_launcher_preserves_uv_path_and_cleans_environment_overrides(self) -> None:
        with (
            patch("docs.sys.prefix", "/not-the-docs-environment"),
            patch.dict(
                os.environ,
                {
                    "UV": "/uv/uv",
                    "PYTHONPATH": "/source",
                    "UV_PROJECT_ENVIRONMENT": "/source/.venv",
                },
            ),
            patch(
                "docs.subprocess.run", return_value=subprocess.CompletedProcess([], 7)
            ) as run,
            self.assertRaises(SystemExit) as exit_result,
        ):
            enter_docs_environment("pages")
        self.assertEqual(exit_result.exception.code, 7)
        args, kwargs = run.call_args
        self.assertEqual(args[0][0], "/uv/uv")
        self.assertIn("--locked", args[0])
        self.assertIn("--no-env-file", args[0])
        self.assertEqual(args[0][-1], "pages")
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertNotIn("UV_PROJECT_ENVIRONMENT", kwargs["env"])

    def test_launcher_does_not_fall_back_after_isolated_install_failure(self) -> None:
        with (
            patch("docs.sys.prefix", "/not-the-docs-environment"),
            patch.dict(os.environ, {"UV": "/uv/uv"}),
            patch(
                "docs.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "uv"),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            enter_docs_environment("check")

    def test_launcher_does_not_reenter_the_docs_environment(self) -> None:
        with (
            patch("docs.sys.prefix", str(DOCS_PROJECT / ".venv")),
            patch("docs.subprocess.run") as run,
        ):
            enter_docs_environment("check")
        run.assert_not_called()

    def test_launcher_requires_uv_instead_of_using_local_source(self) -> None:
        with (
            patch("docs.sys.prefix", "/not-the-docs-environment"),
            patch.dict(os.environ, {"UV": ""}),
            patch("docs.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "Install uv"),
        ):
            enter_docs_environment("check")


class RenderedHTMLTests(unittest.TestCase):
    def test_preview_output_disallows_indexing_and_production_keeps_robots(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            site = Path(directory)
            robots = site / "robots.txt"
            robots.write_text("User-agent: *\nAllow: /\n")
            prepare_pages_output(site, "main")
            self.assertIn("Allow: /", robots.read_text())
            self.assertFalse((site / "_headers").exists())
            prepare_pages_output(site, "docs/example")
            self.assertIn("Disallow: /", robots.read_text())
            self.assertIn(
                "X-Robots-Tag: noindex, nofollow", (site / "_headers").read_text()
            )

    def test_highlighted_example_preserves_code_and_expected_output(self) -> None:
        page = Page()
        page.feed(
            '<article><div class="language-python highlight"><pre><code><span>print</span>("x &amp; y")\n</code></pre></div><div class="language-text"><pre>x &amp; y\n</pre></div></article>'
        )
        self.assertEqual(
            page.blocks, [("python", 'print("x & y")\n'), ("text", "x & y\n")]
        )

    def test_duplicate_anchor_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Duplicate HTML id"):
            Page().feed('<h1 id="same">Title</h1><h2 id="same">Other</h2>')


if __name__ == "__main__":
    unittest.main()

"""Offline tests of the documentation publication boundary and HTML checks."""

from __future__ import annotations

import io
import json
import unittest
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from docs_gate import REPOSITORY, latest_run, wait_for_docs_ci

from docs import Page, prepare_pages_output

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

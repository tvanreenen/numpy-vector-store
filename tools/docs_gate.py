"""Require successful push CI for the exact source Pages is about to build."""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPOSITORY = "tvr-works/numpy-vector-store"
WORKFLOW = "docs.yml"


def latest_run(
    runs: list[dict[str, Any]], sha: str, branch: str
) -> dict[str, Any] | None:
    """An older success cannot hide a newer pending or failed matching run."""
    matches = [
        run
        for run in runs
        if run.get("head_sha") == sha
        and run.get("head_branch") == branch
        and run.get("event") == "push"
        and run.get("head_repository", {}).get("full_name") == REPOSITORY
    ]
    return max(
        matches,
        key=lambda run: (run["run_number"], run["run_attempt"], run["id"]),
        default=None,
    )


def wait_for_docs_ci(sha: str, branch: str, *, timeout: float = 600) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("CF_PAGES_COMMIT_SHA must be a full lowercase Git SHA")
    if not branch or branch.strip() != branch or any(char.isspace() for char in branch):
        raise ValueError("CF_PAGES_BRANCH must be a nonempty Git branch name")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    query = urlencode(
        {"head_sha": sha, "branch": branch, "event": "push", "per_page": 100}
    )
    request = Request(
        f"https://api.github.com/repos/{REPOSITORY}/actions/workflows/{WORKFLOW}/runs?{query}",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "numpy-vector-store-docs-gate",
        },
    )
    deadline = time.monotonic() + timeout
    print(f"Waiting for {WORKFLOW} push CI: {branch} at {sha}", flush=True)
    while time.monotonic() < deadline:
        # Public read-only endpoint; no GitHub or Cloudflare credential is used.
        # HTTP, rate-limit, transport, and malformed-response failures fail closed.
        with urlopen(
            request, timeout=min(15, max(0.1, deadline - time.monotonic()))
        ) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("workflow_runs"), list
        ):
            raise RuntimeError(
                "Invalid GitHub workflow-runs response; refusing deployment"
            )
        run = latest_run(payload["workflow_runs"], sha, branch)
        if run is not None and run.get("status") == "completed":
            if run.get("conclusion") != "success":
                raise RuntimeError(
                    f"Docs CI concluded {run.get('conclusion')}; refusing deployment"
                )
            print(
                f"Exact-source docs CI passed: run {run['id']}, attempt {run['run_attempt']}"
            )
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(10, remaining))
    raise TimeoutError(
        "Timed out waiting for exact-source docs CI; refusing deployment"
    )

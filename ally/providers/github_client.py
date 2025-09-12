from __future__ import annotations
import os
import json
import zipfile
import io
import re
import requests
from typing import Dict, List, Optional

API = "https://api.github.com"


class GH:
    def __init__(self, owner: str, repo: str, token: Optional[str] = None):
        self.owner, self.repo = owner, repo
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.sess = requests.Session()
        if self.token:
            self.sess.headers["Authorization"] = f"Bearer {self.token}"
        self.sess.headers["Accept"] = "application/vnd.github+json"

    def pr_by_number(self, n: int) -> Dict:
        r = self.sess.get(f"{API}/repos/{self.owner}/{self.repo}/pulls/{n}")
        r.raise_for_status()
        return r.json()

    def list_check_runs(self, sha: str) -> List[Dict]:
        r = self.sess.get(f"{API}/repos/{self.owner}/{self.repo}/commits/{sha}/check-runs")
        r.raise_for_status()
        return r.json().get("check_runs", [])

    def list_workflow_runs(self, branch: str) -> List[Dict]:
        r = self.sess.get(f"{API}/repos/{self.owner}/{self.repo}/actions/runs", params={"branch": branch})
        r.raise_for_status()
        return r.json().get("workflow_runs", [])

    def list_run_artifacts(self, run_id: int) -> List[Dict]:
        r = self.sess.get(f"{API}/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/artifacts")
        r.raise_for_status()
        return r.json().get("artifacts", [])

    def download_artifact_json(self, artifact_id: int) -> Dict:
        r = self.sess.get(f"{API}/repos/{self.owner}/{self.repo}/actions/artifacts/{artifact_id}/zip")
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # pick first *.json inside
        for name in z.namelist():
            if name.endswith(".json"):
                return json.loads(z.read(name).decode("utf-8"))
        return {}

    def comment_pr(self, pr_number: int, body: str) -> None:
        r = self.sess.post(f"{API}/repos/{self.owner}/{self.repo}/issues/{pr_number}/comments", json={"body": body})
        r.raise_for_status()

    def set_check(self, sha: str, name: str, status: str, conclusion: Optional[str], summary: str) -> None:
        # Uses Checks API via REST (fallback to status if checks are restricted)
        r = self.sess.post(
            f"{API}/repos/{self.owner}/{self.repo}/check-runs",
            json={
                "name": name,
                "head_sha": sha,
                "status": status,
                "conclusion": conclusion,
                "output": {"title": name, "summary": summary}
            }
        )
        # If insufficient permission for checks, silently ignore
        if r.status_code not in (201, 202):
            pass
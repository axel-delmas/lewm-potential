"""Save results to a GitHub Release of this repo (no Hugging Face needed).

Git pushes reject files > 100 MB, so checkpoints go to a (draft, private) GitHub Release
instead: release assets can be up to 2 GB each and do not bloat git history.

  - results.tar.gz           everything in out/ except *.pt (evals, metrics, logs, report, plots,
                              manifest, calibration), re-uploaded on every sync
  - <job>__model_ep5.pt ...  milestone / final weights, uploaded once each
  - <job>__full_ep20.pt      final weights + optimizer/scheduler/RNG state (resume / extend)
  - <job>__resume.pt         latest full checkpoint of an unfinished job (replaced on every sync,
                              deleted once that job's training is done)
  - release description      report.md

Auth: GITHUB_TOKEN (fine-grained token, repository "Contents: Read and write"), else the
token embedded in the git remote URL (https://<token>@github.com/...). Repo: GITHUB_REPO
(owner/name), else parsed from the git remote.

CLI:
  python -m fm.github_sync check                    # can we write? (creates nothing)
  python -m fm.github_sync download --dest DIR [--tag TAG]   # rebuild an out/ dir from a release
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
MAX_ASSET = 2 * 1000**3 - 1
TAG_PREFIX = "fm-results-"


def _remote_url():
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def resolve_auth():
    url = _remote_url()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        m = re.match(r"https://(?:[^:@/]+:)?([^@/]+)@github\.com/", url)
        token = m.group(1) if m else None
    repo = os.environ.get("GITHUB_REPO")
    if not repo:
        m = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
        repo = f"{m.group(1)}/{m.group(2)}" if m else None
    return token, repo


class GitHubError(RuntimeError):
    pass


class GitHubSync:
    def __init__(self, out_dir, log=print):
        self.out = Path(out_dir)
        self.log = log
        self.token, self.repo = resolve_auth()
        self.release = None
        self.state_path = self.out / ".github_sync_state.json"

    def enabled(self):
        return bool(self.token and self.repo)

    # ------------------------------------------------------------------ http
    def _req(self, method, url, data=None, headers=None, raw=False, timeout=120):
        h = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
             "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "fm-jepa-sync"}
        h.update(headers or {})
        if data is not None and not isinstance(data, (bytes, bytearray)) and not hasattr(data, "read"):
            data = json.dumps(data).encode()
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if raw:
                    return body
                return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            raise GitHubError(f"{method} {url.split('?')[0]} -> HTTP {e.code}: {msg}") from None

    # ------------------------------------------------------------------ release
    def check(self):
        """Verify the token can write to the repo (no side effects)."""
        if not self.enabled():
            raise GitHubError("no GitHub token/repo (set GITHUB_TOKEN, or clone with https://<token>@github.com/...)")
        info = self._req("GET", f"{API}/repos/{self.repo}")
        perms = info.get("permissions") or {}
        if not (perms.get("push") or perms.get("admin") or perms.get("maintain")):
            raise GitHubError(f"token has no write access to {self.repo} (needs 'Contents: Read and write')")
        return info

    def ensure_release(self, tag):
        if self.release and self.release.get("tag_name") == tag:
            return self.release
        # drafts are listed with a placeholder tag ("untagged-..."), so match on the name (== tag)
        for rel in self._req("GET", f"{API}/repos/{self.repo}/releases?per_page=100") or []:
            if rel.get("name") == tag or rel.get("tag_name") == tag:
                self.release = rel
                return rel
        self.release = self._req("POST", f"{API}/repos/{self.repo}/releases", {
            "tag_name": tag, "name": tag, "draft": True,
            "body": "Results and checkpoints uploaded automatically by fm/orchestrate.py."})
        self.log(f"created draft GitHub release '{tag}' on {self.repo}")
        return self.release

    def assets(self):
        out, page = {}, 1
        while True:
            batch = self._req("GET", f"{API}/repos/{self.repo}/releases/{self.release['id']}/assets?per_page=100&page={page}")
            for a in batch or []:
                out[a["name"]] = a
            if not batch or len(batch) < 100:
                return out
            page += 1

    def delete_asset(self, asset):
        self._req("DELETE", f"{API}/repos/{self.repo}/releases/assets/{asset['id']}")

    def upload(self, path_or_bytes, name, existing=None):
        if existing is not None:
            self.delete_asset(existing)
        if isinstance(path_or_bytes, (bytes, bytearray)):
            data, size = bytes(path_or_bytes), len(path_or_bytes)
        else:
            size = Path(path_or_bytes).stat().st_size
            data = open(path_or_bytes, "rb")
        if size > MAX_ASSET:
            raise GitHubError(f"{name} is {size / 1e9:.2f} GB > 2 GB release-asset limit")
        url = f"{UPLOADS}/repos/{self.repo}/releases/{self.release['id']}/assets?name={urllib.parse.quote(name)}"
        try:
            return self._req("POST", url, data=data, timeout=3600,
                             headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)})
        finally:
            if hasattr(data, "close"):
                data.close()

    # ------------------------------------------------------------------ sync
    def _state(self):
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {}

    def _save_state(self, st):
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(st))
        os.replace(tmp, self.state_path)

    def results_tar(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tar:
            for p in sorted(self.out.rglob("*")):
                if not p.is_file() or p.suffix == ".pt" or ".tmp" in p.name or p.name.startswith(".github_sync"):
                    continue
                try:
                    tar.add(p, arcname=str(Path("out") / p.relative_to(self.out)))
                except OSError:
                    continue
        return buf.getvalue()

    def sync(self, tag, final=False):
        """Upload new/changed results. Returns a short summary string."""
        t0 = time.time()
        self.ensure_release(tag)
        assets = self.assets()
        st = self._state()
        n_up, bytes_up = 0, 0

        blob = self.results_tar()
        self.upload(blob, "results.tar.gz", existing=assets.get("results.tar.gz"))
        n_up += 1
        bytes_up += len(blob)

        for jd in sorted((self.out / "jobs").glob("*")):
            ck = jd / "ckpt"
            if not ck.is_dir():
                continue
            done = (jd / "TRAIN_DONE").exists()
            permanent = list(ck.glob("model_ep*.pt"))
            if done:
                permanent += list(ck.glob("full_ep*.pt"))
            for p in permanent:
                name = f"{jd.name}__{p.name}"
                a = assets.get(name)
                if a and a.get("size") == p.stat().st_size and a.get("state") == "uploaded":
                    continue
                self.upload(p, name, existing=a)
                n_up += 1
                bytes_up += p.stat().st_size
            rname = f"{jd.name}__resume.pt"
            if done:
                if rname in assets:
                    self.delete_asset(assets[rname])
                st.pop(rname, None)
                continue
            fulls = [p for p in ck.glob("full_*.pt") if ".tmp" not in p.name]
            if not fulls:
                continue
            latest = max(fulls, key=lambda p: p.stat().st_mtime)
            key = f"{latest.name}:{latest.stat().st_mtime_ns}:{latest.stat().st_size}"
            if st.get(rname) == key and rname in assets:
                continue
            self.upload(latest, rname, existing=assets.get(rname))
            st[rname] = key
            n_up += 1
            bytes_up += latest.stat().st_size
        self._save_state(st)

        report = self.out / "report.md"
        if report.exists():
            body = report.read_text()
            if len(body) > 120_000:
                body = body[:120_000] + "\n\n…(truncated; full report in results.tar.gz)"
            self._req("PATCH", f"{API}/repos/{self.repo}/releases/{self.release['id']}", {"body": body})
        return f"{n_up} file(s), {bytes_up / 1e9:.2f} GB in {time.time() - t0:.0f}s"

    def release_url(self):
        return f"https://github.com/{self.repo}/releases (draft '{(self.release or {}).get('name', '?')}')"

    # ------------------------------------------------------------------ download / restore
    def download(self, dest, tag=None):
        """Rebuild an out/ directory (results + checkpoints) from a release."""
        rels = [r for r in (self._req("GET", f"{API}/repos/{self.repo}/releases?per_page=100") or [])
                if (r.get("name") or "").startswith(TAG_PREFIX)]
        if tag:
            rels = [r for r in rels if r["name"] == tag]
        if not rels:
            raise GitHubError(f"no '{TAG_PREFIX}*' release found on {self.repo}" + (f" with tag {tag}" if tag else ""))
        self.release = max(rels, key=lambda r: r["created_at"])
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        assets = self.assets()
        print(f"downloading release '{self.release['name']}' ({len(assets)} files) -> {dest}", flush=True)
        for name, a in sorted(assets.items()):
            data = self._req("GET", f"{API}/repos/{self.repo}/releases/assets/{a['id']}", raw=True, timeout=3600,
                             headers={"Accept": "application/octet-stream"})
            if name == "results.tar.gz":
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                    for m in tar.getmembers():
                        if not m.isfile() or ".." in Path(m.name).parts or not m.name.startswith("out/"):
                            continue
                        target = dest / Path(m.name).relative_to("out")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(tar.extractfile(m).read())
            elif "__" in name and name.endswith(".pt"):
                job, fname = name.split("__", 1)
                if fname == "resume.pt":
                    fname = "full_resume.pt"  # picked up by train_fm's resume (newest global_step wins)
                target = dest / "jobs" / job / "ckpt" / fname
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            print(f"  {name} ({len(data) / 1e6:.1f} MB)", flush=True)
        return self.release["name"]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    d = sub.add_parser("download")
    d.add_argument("--dest", required=True)
    d.add_argument("--tag", default=None)
    args = p.parse_args()
    g = GitHubSync(Path("."))
    try:
        if args.cmd == "check":
            info = g.check()
            print(f"OK: can write to {g.repo} ({'private' if info.get('private') else 'public'})")
        else:
            g.download(args.dest, args.tag)
    except GitHubError as e:
        print(f"GitHub error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

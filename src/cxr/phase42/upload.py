"""Opt-in private incremental results branch plus uncompressed inference weights."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from cxr.batch.artifacts import commit
from cxr.batch.plan import write_json
from cxr.phase42.protocol import digest


def upload(root: Path, settings: dict) -> dict:
    if not settings.get("enabled"):
        raise ValueError("Friend has not authorized result upload")
    repo, branch, tag = settings["repository"], settings["branch"], settings["release_tag"]
    if (repo != "Zoey-711/cxr-multilabel-challenge" or branch != "results/phase42-pilots" or
            tag != "phase42-pilots-v1"):
        raise ValueError("Dedicated private Phase42 result branch and Release required")
    root = Path(root).resolve(); checkout = Path(settings["checkout"]).resolve()
    if checkout == root or checkout in root.parents or root in checkout.parents:
        raise ValueError("Upload checkout overlaps output directory")

    def gh(*args):
        return subprocess.check_output(["gh", *args], text=True).strip()

    meta = json.loads(gh("repo", "view", repo, "--json", "isPrivate,viewerPermission"))
    if not meta["isPrivate"] or meta["viewerPermission"] not in ("ADMIN", "WRITE", "MAINTAIN"):
        raise ValueError("Private repository write permission required")
    remote = "https://github.com/" + repo + ".git"
    checkout.mkdir(parents=True, exist_ok=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()

    if not (checkout / ".git").exists():
        git("init"); git("remote", "add", "origin", remote)
        if git("ls-remote", "--heads", "origin", branch):
            git("fetch", "origin", branch); git("checkout", "-b", branch, "FETCH_HEAD")
            existing = checkout / 'results/phase42_batch/export_manifest.json'
            if not existing.exists() or json.loads(existing.read_text()).get('phase') != 'phase42_v1':
                raise ValueError('Result branch name collision; refusing to reuse unrelated history')
        else:
            git("checkout", "--orphan", branch)
    if git("remote", "get-url", "origin") != remote or git("branch", "--show-current") != branch:
        raise ValueError("Upload checkout branch or repository changed")
    if any(not line[3:].startswith("results/phase42_batch") for line in git("status", "--porcelain").splitlines()):
        raise ValueError("Unrelated local changes in result checkout")

    folders = [p.parent for p in (root / "experiments").glob("*/DONE.json")]

    if not folders:
        return dict(state="EMPTY")
    for folder in folders:
        marker = folder / ("DONE.json" if (folder / "DONE.json").exists() else "REUSED.json")
        hashes = json.loads(marker.read_text())["sha256"]
        if any(digest(folder / name) != value for name, value in hashes.items()):
            raise ValueError(f"Completed artifact changed: {folder.name}")

    release = subprocess.run(["gh", "release", "view", tag, "--repo", repo], capture_output=True, text=True)
    note = 'Phase42 frozen pilots/confirmation inference weights; no raw images or labels.'
    if release.returncode:
        gh("release", "create", tag, "--repo", repo, "--target", commit(), "--title", tag,
           "--notes", note)
    else:
        existing_release = json.loads(gh('release', 'view', tag, '--repo', repo, '--json', 'body'))
        if existing_release['body'].strip() != note:
            raise ValueError('Release tag collision; refusing unrelated release')
    assets = {a["name"]: a for a in json.loads(gh("api", f"repos/{repo}/releases/tags/{tag}"))["assets"]}
    binaries = []
    for folder in folders:
        weight = folder / "ema_best.pth"
        name = folder.name + "__ema_best.pth"
        sha = digest(weight); label = "sha256:" + sha
        if name in assets:
            if assets[name]["size"] != weight.stat().st_size or assets[name].get("label") != label:
                raise ValueError("Existing inference weight differs; refusing clobber")
        else:
            with tempfile.TemporaryDirectory(prefix="phase42_weight_", dir=root) as temp:
                temporary = Path(temp) / name
                shutil.copyfile(weight, temporary)
                gh("release", "upload", tag, str(temporary) + "#" + label, "--repo", repo)
        binaries.append(dict(model_id=folder.name, asset=name, sha256=sha,
                             bytes=weight.stat().st_size, release_tag=tag))

    target = checkout / "results/phase42_batch"
    sources = []
    for folder in folders:
        sources.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in
                       (".json", ".csv", ".yaml", ".sha256", ".log") and not p.name.endswith(".pid.json"))
    for name in ("reports", "analysis", "oof", "validation", "reference_baselines"):
        sources.extend(p for p in (root / name).rglob("*") if p.is_file() and p.suffix in (".json", ".csv", ".md", ".yaml", ".log", ".sha256"))
    sources.extend(p for p in root.glob('queue_*.json') if p.is_file())
    hashes = {}
    for source in sources:
        if source.stat().st_size > 32 * 2**20:
            raise ValueError("Unexpected large text result")
        text = source.read_text(encoding="utf-8-sig")
        if re.search(r"github_pat_|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY|password\s*[:=]", text, re.I):
            raise ValueError("Possible credential in result")
        relative = source.relative_to(root); destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) == digest(source):
            pass
        else:
            shutil.copyfile(source, destination)
        hashes[relative.as_posix()] = digest(destination)
    write_json(target / "binary_release_manifest.json", dict(repository=repo, release_tag=tag, assets=binaries))
    write_json(target / "export_manifest.json", dict(phase='phase42_v1', code_commit=commit(), files=hashes,
               raw_images="EXCLUDED", raw_labels="EXCLUDED", optimizer_checkpoints="EXCLUDED"))
    git("add", "--", "results/phase42_batch")
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", "Publish verified incremental Phase42 results")
    git("push", "origin", f"HEAD:refs/heads/{branch}")
    return dict(state="UPLOADED", models=len(folders), branch=branch, release_tag=tag)

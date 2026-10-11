"""Clone third-party repositories at the pinned commits of configs/third_party.lock.yaml.

    python scripts/clone_third_party.py            # all
    python scripts/clone_third_party.py DEIMv2 D-FINE
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(list(cmd), cwd=cwd)


def main(names):
    lock = yaml.safe_load((ROOT / "configs" / "third_party.lock.yaml").read_text(encoding="utf-8"))
    for name, e in lock.items():
        if names and name not in names:
            continue
        d = ROOT / "third_party" / name
        if not (d / ".git").exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            run("git", "init", "-q", str(d))
            run("git", "remote", "add", "origin", e["url"], cwd=d)
        run("git", "fetch", "-q", "--depth", "1", "origin", e["commit"], cwd=d)
        run("git", "checkout", "-q", "--detach", e["commit"], cwd=d)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=d, text=True).strip()
        assert head == e["commit"], f"{name}: HEAD {head} != pinned {e['commit']}"
        print(f"{name}: {head} OK")


if __name__ == "__main__":
    main(sys.argv[1:])

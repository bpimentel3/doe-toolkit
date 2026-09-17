import json
import re
import subprocess
import sys

REQUIRED = (0, 9, 2)


def parse_version(text):
    parts = [int(x) for x in re.findall(r"\d+", text or "")[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def main():
    try:
        proc = subprocess.run(
            ["conda", "list", "-n", "base", "conda-pack", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        print("Failed to run 'conda'. Run this build from a conda prompt.")
        return 1
    try:
        records = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        records = []
    pkg = next((r for r in records if r.get("name") == "conda-pack"), None)
    if pkg is None:
        print("conda-pack is NOT installed in base.")
        print("Install it with:")
        print("  conda install -n base -c conda-forge conda-pack=0.9.2 --freeze-installed")
        return 1
    version = pkg.get("version", "")
    if parse_version(version) < REQUIRED:
        print("conda-pack %s found in base, but this build requires 0.9.2+." % version)
        print("Older versions corrupt Python source files in the packaged")
        print("environment on Windows during unpack (they strip the extended-path")
        print("prefix '\\\\?\\' from file contents), which breaks the app.")
        print()
        print("Upgrade with:")
        print("  conda install -n base -c conda-forge conda-pack=0.9.2 --freeze-installed")
        return 1
    print("conda-pack %s OK" % version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
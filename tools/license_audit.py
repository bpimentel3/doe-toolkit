"""License audit for the conda-pack DOE-Toolkit output (stdlib only).

Scans a built environment (conda-managed metadata from ``conda-meta`` plus
pip-installed distributions from ``site-packages``) and classifies each
component's license so the packaged app can be redistributed safely. Also
emits a ``THIRD_PARTY_NOTICES.txt`` file to ship with the app.

Usage:
    python tools/license_audit.py --env dist\\DOE-Toolkit\\env
    python tools/license_audit.py --env <env> --emit-notices
    python tools/license_audit.py --env <env> --file src

Exit code is 1 if any strong-copyleft license (GPL/AGPL/SSPL) is found,
otherwise 0. Weak-copyleft (LGPL/MPL/EPL) and copyleft-with-exception
(e.g. GCC runtime library exception) components are reported but do not
fail the audit.
"""

import argparse
import collections
import glob
import json
import importlib.metadata as im
import os
import re
import sys


SPDX_FULLTEXT_HINTS = [
    ("gcc runtime library exception", "GPL-3.0 WITH GCC-exception-3.1"),
    ("lesser general public license", "LGPL"),
    ("gnu general public license", "GPL"),
    ("affero", "AGPL"),
    ("mozilla public license", "MPL"),
    ("eclipse public license", "EPL"),
    ("apache license", "Apache-2.0"),
    ("mit license", "MIT"),
    ("python software foundation", "PSF"),
]


def find_site_packages(env):
    win = os.path.join(env, "Lib", "site-packages")
    if os.path.isdir(win):
        return win
    matches = sorted(glob.glob(os.path.join(env, "lib", "python*", "site-packages")))
    if matches:
        return matches[-1]
    return None


def extract_license(metadata):
    expr = metadata.get_all("License-Expression") or []
    if expr:
        buf = " ".join(x.strip() for x in expr if x.strip())
        if buf:
            return buf[:400]

    classifiers = [
        c for c in (metadata.get_all("Classifier") or []) if c.startswith("License ::")
    ]
    if classifiers:
        parts = []
        for c in classifiers:
            value = c.rsplit("::", 1)[-1].strip()
            if value and value not in parts:
                parts.append(value)
        if parts:
            return "; ".join(parts)

    lic = metadata.get("License")
    if lic:
        txt = lic.strip()
        if txt and txt.upper() not in ("UNKNOWN", "NONE"):
            first = txt.splitlines()[0].strip() if txt else ""
            if len(first) < 200 and not first.lower().startswith("copyright"):
                return first
            low = txt.lower()
            for marker, canonical in SPDX_FULLTEXT_HINTS:
                if marker in low:
                    return canonical
            return "see License-File (full text)"

    return "UNKNOWN"


def categorize(license_text):
    if not license_text:
        return "unknown"
    s = license_text.lower()
    if "agpl" in s or "affero" in s or "sspl" in s:
        return "strong-copyleft"
    if ("gpl" in s or "gnu general public" in s) and "lgpl" not in s and "lesser" not in s:
        if "exception" in s or "runtime" in s:
            return "copyleft-exception"
        return "strong-copyleft"
    if any(k in s for k in (
        "lgpl", "lesser general public", "mpl", "mozilla public",
        "epl", "eclipse public", "cddl", "cc-by-sa", "eupl",
    )):
        return "weak-copyleft"
    if any(k in s for k in (
        "bsd", "mit", "apache", "psf", "python software foundation",
        "python-2.0", "0bsd", "isc", "zlib", "zope", "unlicense",
        "public domain", "expat", "x11", "wtfpl", "cc0", "openssl", "tcl",
    )):
        return "permissive"
    if s.startswith("bzip2-"):
        return "permissive"
    return "unknown"


def read_conda(env):
    rows = []
    for fp in glob.glob(os.path.join(env, "conda-meta", "*.json")):
        try:
            with open(fp, encoding="utf-8") as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            continue
        rows.append({
            "name": j.get("name") or "?",
            "version": str(j.get("version") or ""),
            "license": j.get("license") or "UNKNOWN",
            "channel": j.get("channel") or "unknown",
        })
    return rows


def read_pip(sp):
    rows = []
    if not sp:
        return rows
    for dist in im.distributions(path=[sp]):
        try:
            m = dist.metadata
        except Exception:
            continue
        rows.append({
            "name": m.get("Name") or getattr(dist, "key") or "?",
            "version": str(m.get("Version") or getattr(dist, "version", "") or ""),
            "license": extract_license(m),
            "channel": "pypi",
        })
    return rows


def channel_origin(channel):
    if "repo.anaconda.com" in channel:
        return "defaults (repo.anaconda.com)" if "/pkgs/main" in channel else "anaconda"
    if "conda-forge" in channel:
        return "conda-forge"
    if not channel or channel == "unknown":
        return "unknown"
    return channel.split("/")[2] if channel.count("/") >= 2 else channel


def report(env, rows, sp):
    total = len(rows)
    by_cat = collections.Counter(categorize(r["license"]) for r in rows)
    channel_counts = collections.Counter(channel_origin(r["channel"]) for r in rows)

    print("=" * 72)
    print("DOE-Toolkit license audit")
    print("env             : %s" % env)
    print("site-packages   : %s" % (sp or "NOT FOUND"))
    print("components found: %d" % total)
    print("categories      : %s" % ", ".join(
        "%s=%s" % (k, v) for k, v in sorted(by_cat.items())
    ))
    print("conda channels  : %s" % ", ".join(
        "%s=%s" % (k, v) for k, v in channel_counts.most_common()
    ))
    print("=" * 72)

    flags = 0
    for cat, title in (
        ("strong-copyleft", "STRONG-COPYLEFT (GPL/AGPL/SSPL)"),
        ("unknown", "UNKNOWN LICENSE"),
    ):
        rows_in_cat = sorted(
            (r for r in rows if categorize(r["license"]) == cat),
            key=lambda r: r["name"].lower(),
        )
        if not rows_in_cat:
            continue
        flags += 1 if cat == "strong-copyleft" else 0
        print("\n#### %s [%d]" % (title, len(rows_in_cat)))
        for r in rows_in_cat:
            print("  %-28s %-12s %-45s %s" % (
                r["name"][:28], r["version"][:12],
                (r["license"] or "")[:45], r["channel"][:40],
            ))

    print("\n#### WEAK-COPYLEFT / EXCEPTION (allowed, keep notices)")
    for r in sorted(
        (r for r in rows if categorize(r["license"]) in ("weak-copyleft", "copyleft-exception")),
        key=lambda r: r["name"].lower(),
    ):
        print("  %-28s %-12s %-45s %s" % (
            r["name"][:28], r["version"][:12],
            (r["license"] or "")[:46], r["channel"][:40],
        ))

    print("\nSUMMARY: %d strong-copyleft, %d unknown (of %d)." % (
        by_cat.get("strong-copyleft", 0), by_cat.get("unknown", 0), total,
    ))
    defaults = channel_counts.get("defaults (repo.anaconda.com)", 0)
    if defaults:
        print("NOTE: %d packages come from the repo.anaconda.com 'defaults' channel;"
              % defaults)
        print("      their commercial redistribution is governed by Anaconda's Terms of")
        print("      Service. Building the env from conda-forge avoids this constraint.")
    return 1 if by_cat.get("strong-copyleft", 0) else 0


def emit_notices(env, rows):
    path = os.path.join(env, "THIRD_PARTY_NOTICES.txt")
    lines = [
        "DOE-Toolkit - Third-Party Notices",
        "Generated by tools/license_audit.py. Every package shipped in this",
        "self-contained app plus its license and package origin. Where a wheel",
        "declares License-File entries, the full license text is shipped inside",
        "its .dist-info or conda-meta record.",
        "",
    ]
    for r in sorted(rows, key=lambda x: x["name"].lower()):
        lines.append("%s %s" % (r["name"], r["version"]))
        lines.append("  License : %s" % r["license"])
        lines.append("  Origin  : %s" % r["channel"])
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def scan_source(root):
    ids = set()
    lic = set()
    spdx = re.compile(r"SPDX-License-Identifier:\s*([\w.\-]+)")
    attr = re.compile(r"""__license__\s*=\s*["']([^"']+)["']""")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            ids.update(spdx.findall(text))
            lic.update(attr.findall(text))
    return sorted(ids), sorted(lic)


def main(argv=None):
    parser = argparse.ArgumentParser(description="License audit for DOE-Toolkit builds.")
    parser.add_argument("--env", default="dist/DOE-Toolkit/env",
                        help="path to the packed env (conda-meta + site-packages).")
    parser.add_argument("--emit-notices", action="store_true",
                        help="write THIRD_PARTY_NOTICES.txt into the env.")
    parser.add_argument("--file", default=None,
                        help="scan .py files under this path for SPDX/__license__ markers.")
    args = parser.parse_args(argv)

    env = args.env
    if not os.path.isdir(env):
        print("ERROR: env directory not found: %s" % env, file=sys.stderr)
        return 1

    rows = read_conda(env)
    sp = find_site_packages(env)
    rows.extend(read_pip(sp))
    rc = report(env, rows, sp)

    if args.emit_notices:
        path = emit_notices(env, rows)
        print("\nWrote: %s" % path)

    if args.file:
        ids, lic = scan_source(args.file)
        print("\nSource scan (%s):" % args.file)
        print("  SPDX identifiers : %s" % (", ".join(ids) or "none"))
        print("  __license__      : %s" % (", ".join(lic) or "none"))

    return rc


if __name__ == "__main__":
    sys.exit(main())
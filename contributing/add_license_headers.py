#!/usr/bin/env python3
"""Add the Argonne copyright header to tracked source files.

Usage:
    python contributing/add_license_headers.py          # insert where missing
    python contributing/add_license_headers.py --check  # exit 1 if any missing

Covers tracked *.py, *.c, *.h, and *.cu files. The vendored JZT snapshot
(src/lauelab/analysis/_vendor/) and the MINPACK folder
(src/lauelab/indexing/src/minpack/) are excluded: they contain copies of
third-party code that keep their own headers, and each README documents the
provenance.
Idempotent: files already containing the copyright line are left unchanged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COPYRIGHT = "Copyright \N{COPYRIGHT SIGN} 2026 UChicago Argonne, LLC. All rights reserved."
LICENSE_URL = "Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE"

HEADER_HASH = f"# {COPYRIGHT}\n# {LICENSE_URL}\n".encode()
HEADER_C = f"/* {COPYRIGHT}\n   {LICENSE_URL} */\n".encode()

EXCLUDE_PREFIXES = ("src/lauelab/analysis/_vendor/", "src/lauelab/indexing/src/minpack/")
MARKER = b"UChicago Argonne"


def target_files() -> list[Path]:
    names = subprocess.run(
        ["git", "ls-files", "*.py", "*.c", "*.h", "*.cu"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return [Path(n) for n in names if not n.startswith(EXCLUDE_PREFIXES)]


def apply(path: Path, check_only: bool) -> bool:
    """Return True if the file already had, or now has, the header."""
    data = path.read_bytes()
    if MARKER in data[:400]:
        return True
    if check_only:
        return False
    header = HEADER_HASH if path.suffix == ".py" else HEADER_C
    if data.startswith(b"#!"):
        shebang, _, rest = data.partition(b"\n")
        data = shebang + b"\n" + header + rest
    else:
        data = header + data
    path.write_bytes(data)
    return True


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    missing = [p for p in target_files() if not apply(p, check_only)]
    if check_only:
        for p in missing:
            print(f"missing copyright header: {p}")
        print(f"checked {len(target_files())} files, {len(missing)} missing")
        return 1 if missing else 0
    print(f"processed {len(target_files())} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

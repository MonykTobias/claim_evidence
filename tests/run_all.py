"""Run every standalone test script."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    failed = False
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(path)],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
        status = "OK" if result.returncode == 0 else "FAIL"
        print(f"{status} {path.name} ({time.monotonic() - started:.1f}s)")
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

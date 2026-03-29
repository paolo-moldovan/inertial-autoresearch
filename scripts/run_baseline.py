"""Thin wrapper for `imu_denoise.cli.run_baseline`."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Bootstrap local `src/` imports and delegate to the package CLI."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from imu_denoise.cli.run_baseline import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

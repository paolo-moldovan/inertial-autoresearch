"""CLI entrypoint for syncing mission-control data to external adapters."""

from __future__ import annotations

import argparse
import json

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.observability import sync_observability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync mission-control observability data to MLflow and/or Phoenix."
    )
    add_common_config_arguments(parser)
    parser.add_argument(
        "--target",
        choices=("all", "mlflow", "phoenix"),
        default="all",
        help="Which Phase 2 adapter target to sync.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional single run id to sync.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of runs to replay when syncing all runs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    result = sync_observability(
        config=config,
        target=args.target,
        run_id=args.run_id or None,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

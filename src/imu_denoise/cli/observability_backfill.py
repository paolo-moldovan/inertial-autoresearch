"""CLI entrypoint for observability backfill and Hermes import."""

from __future__ import annotations

import argparse
from pathlib import Path

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.observability import (
    ObservabilityWriter,
    backfill_observability,
    import_hermes_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill mission-control observability data.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--skip-hermes",
        action="store_true",
        help="Skip importing Hermes state even if enabled in config.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    writer = ObservabilityWriter.from_experiment_config(config)

    backfill_counts = backfill_observability(config=config, writer=writer)
    print(f"Backfilled: {backfill_counts}")

    if config.observability.import_hermes_state and not args.skip_hermes:
        hermes_home = Path(config.autoresearch.hermes.home_dir)
        hermes_counts = import_hermes_state(writer=writer, hermes_home=hermes_home)
        print(f"Hermes import: {hermes_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

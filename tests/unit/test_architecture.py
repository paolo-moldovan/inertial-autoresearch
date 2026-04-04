"""Architecture guardrails for the autoresearch-core extraction."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.append(node.module)
    return targets


def test_autoresearch_core_does_not_import_imu_or_torch() -> None:
    banned_prefixes = ("imu_denoise", "torch")
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "autoresearch_core").rglob("*.py")):
        imports = _import_targets(path)
        violations = [
            target
            for target in imports
            if any(
                target == prefix or target.startswith(prefix + ".")
                for prefix in banned_prefixes
            )
        ]
        if violations:
            offenders.append(f"{path.relative_to(ROOT)} -> {violations}")
    assert not offenders, "\n".join(offenders)


def test_ui_modules_do_not_construct_store_or_writer_directly() -> None:
    ui_modules = [
        ROOT / "src" / "imu_denoise" / "observability" / "dashboard_app.py",
        ROOT / "src" / "imu_denoise" / "observability" / "monitor_app.py",
        ROOT / "src" / "imu_denoise" / "observability" / "web_dashboard.py",
    ]
    banned_modules = {
        "imu_denoise.observability.store",
        "imu_denoise.observability.writer",
    }
    offenders: list[str] = []
    for path in ui_modules:
        imports = set(_import_targets(path))
        violations = sorted(imports & banned_modules)
        if violations:
            offenders.append(f"{path.relative_to(ROOT)} -> {violations}")
    assert not offenders, "\n".join(offenders)


def test_ui_modules_flow_through_facade_instead_of_queries_and_controller() -> None:
    ui_modules = [
        ROOT / "src" / "imu_denoise" / "observability" / "dashboard_app.py",
        ROOT / "src" / "imu_denoise" / "observability" / "monitor_app.py",
        ROOT / "src" / "imu_denoise" / "observability" / "web_dashboard.py",
    ]
    offenders: list[str] = []
    for path in ui_modules:
        source = path.read_text(encoding="utf-8")
        if "services.queries" in source or "services.controller" in source:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "\n".join(offenders)


def test_trainer_module_is_not_directly_coupled_to_observability() -> None:
    trainer_path = ROOT / "src" / "imu_denoise" / "training" / "trainer.py"
    imports = _import_targets(trainer_path)
    violations = [
        target
        for target in imports
        if target == "imu_denoise.observability"
        or target.startswith("imu_denoise.observability.")
    ]
    assert not violations


def test_cli_modules_do_not_import_store_layer_directly() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "imu_denoise" / "cli").glob("*.py")):
        imports = _import_targets(path)
        violations = [
            target
            for target in imports
            if target == "imu_denoise.observability.store" or target.startswith("sqlite3")
        ]
        if violations:
            offenders.append(f"{path.relative_to(ROOT)} -> {violations}")
    assert not offenders, "\n".join(offenders)

#!/usr/bin/env python3
"""NTSMR Quality Scorer — compares alternative scoring methods against canonical baseline."""

import argparse
import sys
from pathlib import Path
from statistics import mean

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

BASELINE = "ntsmr_scores"

SCORE_METHODS = [
    "llm_memory_quadrant_scores",
    "minilm_rag_scores",
    "nomic_rag_scores",
    "hybrid_index_scores",
    "sliding_window_scores",
    "semantic_map_reduce_scores",
    "grand_unification_scores",
    "sliding_context_scores",
    "model_tiering_scores",
    "enriched_schema_scores",
]

AXES = [
    "time_linearity",
    "pacing_velocity",
    "threat_scale",
    "protagonist_fate",
    "conflict_style",
    "price_type",
]


def parse_frontmatter(path: Path) -> dict | None:
    """Extract YAML frontmatter from a markdown file."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        return yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None


def compute_errors(baseline: dict, candidate: dict) -> dict[str, float] | None:
    """Compute per-axis absolute errors. Returns None if axes are missing."""
    errors = {}
    for axis in AXES:
        b = baseline.get(axis)
        c = candidate.get(axis)
        if b is None or c is None:
            return None
        errors[axis] = abs(float(b) - float(c))
    return errors


def short_name(method: str) -> str:
    """Shorten method names for display."""
    return (
        method.replace("_quadrant_scores", "")
        .replace("_scores", "")
        .replace("_", " ")
    )


def main():
    parser = argparse.ArgumentParser(description="NTSMR Quality Report")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-axis deltas"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Path to data directory (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(data_dir.glob("*.md"))
    md_files = [f for f in md_files if not f.name.startswith("_")]

    # Collect results: {work_title: {method: {mae, errors, worst_axes}}}
    results: dict[str, dict] = {}
    # Aggregate: {method: [mae_values]}
    aggregate: dict[str, list[float]] = {m: [] for m in SCORE_METHODS}
    # Per-axis aggregate: {method: {axis: [errors]}}
    axis_aggregate: dict[str, dict[str, list[float]]] = {
        m: {a: [] for a in AXES} for m in SCORE_METHODS
    }

    for path in md_files:
        fm = parse_frontmatter(path)
        if not fm or BASELINE not in fm:
            continue

        baseline = fm[BASELINE]
        title = fm.get("title", path.stem)
        work_results = {}

        for method in SCORE_METHODS:
            if method not in fm:
                continue
            candidate = fm[method]
            errors = compute_errors(baseline, candidate)
            if errors is None:
                continue

            mae = mean(errors.values())
            worst = sorted(errors.items(), key=lambda x: -x[1])

            work_results[method] = {
                "mae": mae,
                "errors": errors,
                "worst": worst,
            }
            aggregate[method].append(mae)
            for axis, err in errors.items():
                axis_aggregate[method][axis].append(err)

        if work_results:
            results[title] = work_results

    # --- Report ---
    print("NTSMR Quality Report")
    print("=" * 60)
    print()

    # Per-work section
    print("Per-Work Comparison")
    print("-" * 60)

    for title in sorted(results):
        print(f"\n{title}:")
        for method in SCORE_METHODS:
            if method not in results[title]:
                continue
            r = results[title][method]
            name = short_name(method)
            if r["mae"] < 0.001:
                print(f"  {name:<20s} MAE=0.000  (identical to baseline)")
            else:
                worst_str = "  ".join(
                    f"{a} ({e:.2f})" for a, e in r["worst"] if e > 0.001
                )
                print(f"  {name:<20s} MAE={r['mae']:.3f}  worst: {worst_str}")

            if args.verbose and r["mae"] >= 0.001:
                for axis in AXES:
                    e = r["errors"][axis]
                    marker = " ***" if e > 0.2 else ""
                    print(f"    {axis:<20s} {e:+.2f}{marker}")

    # Aggregate section
    print()
    print()
    print("Aggregate by Method")
    print("-" * 60)
    print(f"{'Method':<28s} {'n':>3s}   {'Mean MAE':>8s}   {'Worst Axis'}")

    for method in SCORE_METHODS:
        vals = aggregate[method]
        if not vals:
            continue
        name = short_name(method)
        mean_mae = mean(vals)
        # Find worst axis across all works
        axis_means = {
            a: mean(axis_aggregate[method][a])
            for a in AXES
            if axis_aggregate[method][a]
        }
        if axis_means:
            worst_axis = max(axis_means, key=lambda a: axis_means[a])
            worst_val = axis_means[worst_axis]
            worst_str = f"{worst_axis} ({worst_val:.3f})" if worst_val > 0.001 else "-"
        else:
            worst_str = "-"
        print(f"  {name:<26s} {len(vals):>3d}   {mean_mae:>8.3f}   {worst_str}")

    print()


if __name__ == "__main__":
    main()

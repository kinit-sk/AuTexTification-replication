"""Run LingRF style ablations and aggregate result tables."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

SCRIPTS_DIR: Path = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import configure_project_root

PROJECT_ROOT: Path = configure_project_root(__file__, remove_shadowing_utils=False)
PRECOMPUTE_SCRIPT = PROJECT_ROOT / "scripts" / "ablation" / "precompute_lstm_probs.py"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "ablation" / "training_ablation.py"
PLOT_SHAP_SCRIPT = PROJECT_ROOT / "scripts" / "plot_shap.py"
RESULTS_DIR = PROJECT_ROOT / "data" / "out" / "results"

from scripts.ablation.constants import (
    ABLATION_VARIANTS,
    STYLE_GROUP_NAMES,
    TASK_LANG_PAIRS,
)


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    excluded_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full ablation study for LingRF style variants."
    )
    parser.add_argument(
        "--subtask", choices=["subtask_1", "subtask_2", "all"], default="all",
    )
    parser.add_argument(
        "--lang", choices=["en", "es", "all"], default="all",
    )
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Use multilingual LSTM probs for predout variants.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ABLATION_VARIANTS,
        default=["lingrf_style"],
        help="Ablation variants to run. Default: lingrf_style.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[10],
        help="RF seeds to run. The data split and cached PredOut probabilities stay fixed.",
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="Enable SHAP in ablation runs (disabled by default for speed).",
    )
    parser.add_argument(
        "--paper-all", action="store_true",
        help="Run LingRF, PredOut baseline, and PredOut multilingual ablations.",
    )
    parser.add_argument(
        "--paper-all-shap", action="store_true",
        help="With --paper-all, also run SHAP for all full feature models and build plots/direction TSV.",
    )
    parser.add_argument(
        "--paper-shap-samples", type=int, default=500,
        help="SHAP sample count used by --paper-all-shap.",
    )
    return parser.parse_args()


def build_precompute_command(subtask: str, lang: str, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(PRECOMPUTE_SCRIPT), "--subtask", subtask, "--lang", lang]
    if args.multilingual:
        cmd.append("--multilingual")
    return cmd


def build_training_command(
    spec: AblationSpec,
    subtask: str,
    lang: str,
    args: argparse.Namespace,
    seed: int,
) -> list[str]:
    cmd = [
        sys.executable, str(TRAINING_SCRIPT),
        "--subtask", subtask,
        "--lang", lang,
        "--variant", spec.variant,
        "--seed", str(seed),
    ]
    if spec.excluded_group:
        cmd.extend(["--exclude-style-groups", spec.excluded_group])
    if args.multilingual:
        cmd.append("--multilingual")
    if not args.shap:
        cmd.append("--no-shap")
    return cmd


def run_command(command: list[str]) -> tuple[int, float]:
    """Run command, stream stdout, return (exit_code, elapsed_seconds)."""
    started_at = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode, time.perf_counter() - started_at


def run_required(command: list[str], label: str) -> None:
    print(f"\n[RUN] {label}")
    print(f"Command: {' '.join(command)}")
    return_code, elapsed = run_command(command)
    print(f"[DONE] {label} exit={return_code} elapsed={elapsed:.1f}s")
    if return_code != 0:
        raise RuntimeError(f"Command failed for {label}: exit={return_code}")


def copy_latest_ablation(filename: str, summary_filename: str) -> None:
    source_path = RESULTS_DIR / "ablation_results_latest.tsv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing expected ablation output: {source_path}")
    target_path = RESULTS_DIR / filename
    shutil.copy2(source_path, target_path)
    print(f"[SAVED] {target_path}")

    summary_source_path = RESULTS_DIR / "ablation_summary_latest.tsv"
    if not summary_source_path.exists():
        raise FileNotFoundError(f"Missing expected ablation summary output: {summary_source_path}")
    summary_target_path = RESULTS_DIR / summary_filename
    shutil.copy2(summary_source_path, summary_target_path)
    print(f"[SAVED] {summary_target_path}")


def run_full_model_shap(variant: str, multilingual: bool, shap_samples: int) -> None:
    for subtask, lang in TASK_LANG_PAIRS:
        command = [
            sys.executable,
            str(TRAINING_SCRIPT),
            "--subtask",
            subtask,
            "--lang",
            lang,
            "--variant",
            variant,
            "--shap-samples",
            str(shap_samples),
        ]
        if multilingual:
            command.append("--multilingual")
        label_suffix = " multilingual" if multilingual else ""
        run_required(command, f"SHAP {variant}{label_suffix} {subtask}/{lang}")


def run_paper_all(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    this_script = Path(__file__).resolve()
    seed_args = ["--seeds"] + [str(seed) for seed in args.seeds]

    stages: tuple[tuple[str, list[str], str, str], ...] = (
        (
            "LingRF ablation",
            [sys.executable, str(this_script), "--variants", "lingrf_style", *seed_args],
            "ablation_results_lingrf.tsv",
            "ablation_summary_lingrf.tsv",
        ),
        (
            "PredOut baseline ablation",
            [sys.executable, str(this_script), "--variants", "lingrf_style_predout", *seed_args],
            "ablation_results_predout_baseline.tsv",
            "ablation_summary_predout_baseline.tsv",
        ),
        (
            "PredOut multilingual ablation",
            [
                sys.executable,
                str(this_script),
                "--variants",
                "lingrf_style_predout",
                "--multilingual",
                *seed_args,
            ],
            "ablation_results_predout_multilingual.tsv",
            "ablation_summary_predout_multilingual.tsv",
        ),
    )

    for label, command, output_filename, summary_filename in stages:
        run_required(command, label)
        copy_latest_ablation(output_filename, summary_filename)

    if args.paper_all_shap:
        run_full_model_shap(
            variant="lingrf_style",
            multilingual=False,
            shap_samples=args.paper_shap_samples,
        )
        run_full_model_shap(
            variant="lingrf_style_predout",
            multilingual=False,
            shap_samples=args.paper_shap_samples,
        )
        run_full_model_shap(
            variant="lingrf_style_predout",
            multilingual=True,
            shap_samples=args.paper_shap_samples,
        )
        run_required(
            [
                sys.executable,
                str(PLOT_SHAP_SCRIPT),
                "--all",
                "--mode",
                "multiclass",
                "--top-n",
                "15",
            ],
            "SHAP plots and direction summary",
        )

    print("\nPaper ablation pipeline complete.")


def read_ablation_result(
    subtask: str, lang: str, spec: AblationSpec, seed: int
) -> dict[str, str] | None:
    key = spec.excluded_group if spec.excluded_group else "baseline"
    filename = f"ablation_single_{subtask}_{lang}_{spec.variant}_{key}_seed{seed}.tsv"
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"[WARNING] Result file not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    return rows[0] if rows else None


def _compute_deltas(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Attach dev/test F1 drops relative to the full-feature baseline."""
    baseline_map: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    for row in rows:
        if row.get("excluded_group") == "baseline":
            key = (
                row.get("variant", ""),
                row.get("subtask", ""),
                row.get("lang", ""),
                row.get("seed", ""),
            )
            try:
                baseline_map[key] = (float(row["dev_f1"]), float(row["test_f1"]))
            except (ValueError, KeyError):
                pass

    out: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        key = (
            row.get("variant", ""),
            row.get("subtask", ""),
            row.get("lang", ""),
            row.get("seed", ""),
        )
        baseline_scores = baseline_map.get(key)
        if row.get("excluded_group") == "baseline":
            row["delta_dev_f1"] = "0.0000"
            row["delta_test_f1"] = "0.0000"
        elif baseline_scores is not None and row.get("dev_f1") and row.get("test_f1"):
            try:
                baseline_dev_f1, baseline_test_f1 = baseline_scores
                row["delta_dev_f1"] = f"{baseline_dev_f1 - float(row['dev_f1']):.4f}"
                row["delta_test_f1"] = f"{baseline_test_f1 - float(row['test_f1']):.4f}"
            except ValueError:
                row["delta_dev_f1"] = ""
                row["delta_test_f1"] = ""
        else:
            row["delta_dev_f1"] = ""
            row["delta_test_f1"] = ""
        out.append(row)
    return out


def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key)]


def _format_mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{mean(values):.4f}"


def _format_std(values: list[float]) -> str:
    if not values:
        return ""
    return f"{pstdev(values):.4f}"


def build_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("excluded_group", ""),
            row.get("variant", ""),
            row.get("subtask", ""),
            row.get("lang", ""),
        )
        grouped_rows.setdefault(key, []).append(row)

    summary_rows: list[dict[str, str]] = []
    for (excluded_group, variant, subtask, lang), group_rows in grouped_rows.items():
        first_row = group_rows[0]
        seeds = sorted({int(row["seed"]) for row in group_rows if row.get("seed")})
        train_values = _float_values(group_rows, "train_f1")
        dev_values = _float_values(group_rows, "dev_f1")
        test_values = _float_values(group_rows, "test_f1")
        delta_dev_values = _float_values(group_rows, "delta_dev_f1")
        delta_test_values = _float_values(group_rows, "delta_test_f1")
        runtime_values = _float_values(group_rows, "run_time_sec")

        summary_rows.append({
            "excluded_group": excluded_group,
            "variant": variant,
            "subtask": subtask,
            "lang": lang,
            "n_seeds": str(len(seeds)),
            "seeds": ",".join(str(seed) for seed in seeds),
            "train_f1_mean": _format_mean(train_values),
            "train_f1_std": _format_std(train_values),
            "dev_f1_mean": _format_mean(dev_values),
            "dev_f1_std": _format_std(dev_values),
            "test_f1_mean": _format_mean(test_values),
            "test_f1_std": _format_std(test_values),
            "delta_dev_f1_mean": _format_mean(delta_dev_values),
            "delta_dev_f1_std": _format_std(delta_dev_values),
            "delta_test_f1_mean": _format_mean(delta_test_values),
            "delta_test_f1_std": _format_std(delta_test_values),
            "n_ling_features": first_row.get("n_ling_features", ""),
            "n_style_features": first_row.get("n_style_features", ""),
            "n_total_features": first_row.get("n_total_features", ""),
            "run_time_sec_mean": _format_mean(runtime_values),
            "run_time_sec_sum": f"{sum(runtime_values):.2f}" if runtime_values else "",
        })
    return summary_rows


def write_combined_results(rows: list[dict[str, str]]) -> tuple[Path, Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = RESULTS_DIR / f"ablation_results_{timestamp}.tsv"
    latest_path = RESULTS_DIR / "ablation_results_latest.tsv"
    summary_timestamped_path = RESULTS_DIR / f"ablation_summary_{timestamp}.tsv"
    summary_latest_path = RESULTS_DIR / "ablation_summary_latest.tsv"

    headers = [
        "excluded_group",
        "variant",
        "subtask",
        "lang",
        "seed",
        "train_f1",
        "dev_f1",
        "test_f1",
        "n_ling_features",
        "n_style_features",
        "n_total_features",
        "delta_dev_f1",
        "delta_test_f1",
        "run_time_sec",
    ]

    for path in (timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=headers, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)

    summary_rows = build_summary_rows(rows)
    summary_headers = [
        "excluded_group",
        "variant",
        "subtask",
        "lang",
        "n_seeds",
        "seeds",
        "train_f1_mean",
        "train_f1_std",
        "dev_f1_mean",
        "dev_f1_std",
        "test_f1_mean",
        "test_f1_std",
        "delta_dev_f1_mean",
        "delta_dev_f1_std",
        "delta_test_f1_mean",
        "delta_test_f1_std",
        "n_ling_features",
        "n_style_features",
        "n_total_features",
        "run_time_sec_mean",
        "run_time_sec_sum",
    ]

    for path in (summary_timestamped_path, summary_latest_path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=summary_headers, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    return timestamped_path, latest_path, summary_timestamped_path, summary_latest_path


def main() -> None:
    args = parse_args()
    if args.paper_all:
        run_paper_all(args)
        return

    subtasks = ["subtask_1", "subtask_2"] if args.subtask == "all" else [args.subtask]
    languages = ["en", "es"] if args.lang == "all" else [args.lang]

    specs = [
        AblationSpec(variant=v, excluded_group=g)
        for v in args.variants
        for g in STYLE_GROUP_NAMES
    ]

    total_runs = len(specs) * len(subtasks) * len(languages) * len(args.seeds)
    precompute_subtasks = subtasks if "lingrf_style_predout" in args.variants else []

    print("=" * 88)
    print("Ablation Study Runner")
    print("=" * 88)
    print(f"Subtasks : {subtasks}")
    print(f"Languages: {languages}")
    print(f"Variants : {args.variants}")
    print(f"Seeds    : {args.seeds}")
    print(f"Groups   : {STYLE_GROUP_NAMES}")
    print(
        f"RF runs  : {total_runs}  "
        f"({len(specs)} configs × {len(subtasks)} subtasks × {len(languages)} langs × {len(args.seeds)} seeds)"
    )
    print(f"SHAP     : {args.shap}")
    print("=" * 88)

    print("\n[PHASE 1] Pre-computing LSTM probs (idempotent — skips if cached)...")
    for subtask in precompute_subtasks:
        for lang in languages:
            cmd = build_precompute_command(subtask, lang, args)
            print(f"\n  subtask={subtask}, lang={lang}")
            print(f"  Command: {' '.join(cmd)}")
            rc, elapsed = run_command(cmd)
            if rc != 0:
                print(f"[ERROR] Pre-compute failed for {subtask}/{lang} (exit {rc}). Aborting.")
                sys.exit(1)
            print(f"  [PHASE 1 OK] elapsed={elapsed:.1f}s")

    print("\n[PHASE 2] Running RF ablation configs...")

    all_rows: list[dict[str, str]] = []
    run_index = 0

    for subtask in subtasks:
        for lang in languages:
            for spec in specs:
                for seed in args.seeds:
                    run_index += 1
                    label = spec.excluded_group if spec.excluded_group else "baseline"
                    cmd = build_training_command(spec, subtask, lang, args, seed)

                    print(f"\n{'─' * 88}")
                    print(
                        f"[{run_index}/{total_runs}] "
                        f"variant={spec.variant} | group={label} | {subtask}/{lang} | seed={seed}"
                    )
                    print(f"Command: {' '.join(cmd)}")
                    print("─" * 88)

                    rc, elapsed = run_command(cmd)
                    print(f"\n[RUN END] exit={rc}  elapsed={elapsed:.1f}s")

                    row = read_ablation_result(subtask, lang, spec, seed)
                    if row is not None:
                        row["run_time_sec"] = f"{elapsed:.2f}"
                        all_rows.append(row)
                    else:
                        all_rows.append({
                            "excluded_group": label,
                            "variant": spec.variant,
                            "subtask": subtask,
                            "lang": lang,
                            "seed": str(seed),
                            "train_f1": "",
                            "dev_f1": "",
                            "test_f1": "",
                            "n_ling_features": "",
                            "n_style_features": "",
                            "n_total_features": "",
                            "delta_dev_f1": "",
                            "delta_test_f1": "",
                            "run_time_sec": f"{elapsed:.2f}",
                        })

    all_rows = _compute_deltas(all_rows)
    (
        timestamped_path,
        latest_path,
        summary_timestamped_path,
        summary_latest_path,
    ) = write_combined_results(all_rows)

    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    print(
        f"\n{'Excluded Group':<28} {'Variant':<22} {'Subtask':<12} "
        f"{'Lang':<6} {'Seed':<6} {'Dev F1':<10} {'Test F1':<10} {'DevDrop':>8} {'TestDrop':>8}"
    )
    print("─" * 100)
    for row in all_rows:
        print(
            f"{row.get('excluded_group', ''):<28} "
            f"{row.get('variant', ''):<22} "
            f"{row.get('subtask', ''):<12} "
            f"{row.get('lang', ''):<6} "
            f"{row.get('seed', ''):<6} "
            f"{row.get('dev_f1', ''):<10} "
            f"{row.get('test_f1', ''):<10} "
            f"{row.get('delta_dev_f1', ''):>8} "
            f"{row.get('delta_test_f1', ''):>8}"
        )

    print(f"\n[INFO] Timestamped results : {timestamped_path}")
    print(f"[INFO] Latest results      : {latest_path}")
    print(f"[INFO] Timestamped summary : {summary_timestamped_path}")
    print(f"[INFO] Latest summary      : {summary_latest_path}")
    print("\nAll ablation runs complete!")


if __name__ == "__main__":
    main()

"""Run a fixed LingRF experiment matrix and aggregate results with runtime."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training_lingrf.py"
RESULTS_DIR = PROJECT_ROOT / "data" / "out" / "results"
LATEST_RESULTS = RESULTS_DIR / "lingrf_style_results_latest.tsv"


@dataclass(frozen=True)
class LingRFRunSpec:
    """Describe one runnable LingRF scenario."""

    scenario: str
    variant: str
    multilingual: bool
    prob_source: str


RUN_SPECS: tuple[LingRFRunSpec, ...] = (
    LingRFRunSpec("lingrf_old", "lingrf", False, "none"),
    LingRFRunSpec("lingrf_new", "lingrf_style", False, "none"),
    LingRFRunSpec("lingrf_predout_old_baseline_prob", "lingrf_predout", False, "baseline"),
    LingRFRunSpec("lingrf_predout_new_baseline_prob", "lingrf_style_predout", False, "baseline"),
    LingRFRunSpec("lingrf_predout_old_multilingual_prob", "lingrf_predout", True, "multilingual"),
    LingRFRunSpec("lingrf_predout_new_multilingual_prob", "lingrf_style_predout", True, "multilingual"),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LingRF matrix execution."""
    parser = argparse.ArgumentParser(description="Run LingRF old/new matrix and aggregate outcomes")
    parser.add_argument(
        "--subtask",
        type=str,
        choices=["subtask_1", "subtask_2", "all"],
        default="all",
        help="Subtask scope passed to training_lingrf.py",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "es", "all"],
        default="all",
        help="Language scope passed to training_lingrf.py",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Enable SHAP in child runs (disabled by default)",
    )
    return parser.parse_args()


def build_command(spec: LingRFRunSpec, args: argparse.Namespace) -> list[str]:
    """Create one `training_lingrf.py` command for a scenario."""
    command = [sys.executable, str(TRAINING_SCRIPT), "--variant", spec.variant]
    if args.subtask != "all":
        command.extend(["--subtask", args.subtask])
    if args.lang != "all":
        command.extend(["--lang", args.lang])
    if spec.multilingual:
        command.append("--multilingual")
    if not args.shap:
        command.append("--no-shap")
    return command


def run_command(command: list[str]) -> tuple[int, float, str]:
    """Run a command, stream logs, and return exit code, elapsed seconds, and output text."""
    started_at = time.perf_counter()
    output_lines: list[str] = []

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
        output_lines.append(line)

    process.wait()
    elapsed_sec = time.perf_counter() - started_at
    return process.returncode, elapsed_sec, "".join(output_lines)


def read_latest_results() -> list[dict[str, str]]:
    """Read rows from the latest LingRF TSV produced by training script."""
    if not LATEST_RESULTS.exists():
        raise FileNotFoundError(f"Expected results file not found: {LATEST_RESULTS}")

    with open(LATEST_RESULTS, encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        return list(reader)


def extract_timestamped_path(output: str) -> str:
    """Extract timestamped TSV path from script output when present."""
    match = re.search(r"\[INFO\] Results saved to (.+lingrf_style_results_\d{8}_\d{6}\.tsv)", output)
    return match.group(1) if match else ""


def write_combined_results(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    """Write timestamped and latest combined TSV outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = RESULTS_DIR / f"lingrf_combined_runs_{timestamp}.tsv"
    latest_path = RESULTS_DIR / "lingrf_combined_runs_latest.tsv"

    headers = [
        "scenario",
        "variant",
        "prob_source",
        "multilingual",
        "subtask",
        "lang",
        "train_f1",
        "dev_f1",
        "test_f1",
        "n_ling",
        "n_style",
        "n_total",
        "run_time_sec",
        "source_results_tsv",
    ]

    for path in (timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=headers, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    return timestamped_path, latest_path


def main() -> None:
    """Execute all LingRF scenarios and persist a combined comparison table."""
    args = parse_args()
    print("=" * 88)
    print("LingRF Matrix Runner")
    print("=" * 88)
    print(f"Subtask scope: {args.subtask}")
    print(f"Language scope: {args.lang}")
    print(f"SHAP enabled: {args.shap}")
    print(f"Total scenarios: {len(RUN_SPECS)}")

    combined_rows: list[dict[str, str]] = []

    for index, spec in enumerate(RUN_SPECS, start=1):
        command = build_command(spec, args)
        print("\n" + "-" * 88)
        print(f"[{index}/{len(RUN_SPECS)}] Running {spec.scenario}")
        print(f"Command: {' '.join(command)}")
        print("-" * 88)

        return_code, elapsed_sec, output = run_command(command)
        print(f"\n[RUN END] scenario={spec.scenario} return_code={return_code} time_sec={elapsed_sec:.1f}")

        if return_code != 0:
            combined_rows.append(
                {
                    "scenario": spec.scenario,
                    "variant": spec.variant,
                    "prob_source": spec.prob_source,
                    "multilingual": str(int(spec.multilingual)),
                    "subtask": "",
                    "lang": "",
                    "train_f1": "",
                    "dev_f1": "",
                    "test_f1": "",
                    "n_ling": "",
                    "n_style": "",
                    "n_total": "",
                    "run_time_sec": f"{elapsed_sec:.2f}",
                    "source_results_tsv": "",
                }
            )
            continue

        source_results_tsv = extract_timestamped_path(output)
        current_rows = read_latest_results()
        for result_row in current_rows:
            combined_rows.append(
                {
                    "scenario": spec.scenario,
                    "variant": spec.variant,
                    "prob_source": spec.prob_source,
                    "multilingual": str(int(spec.multilingual)),
                    "subtask": result_row["subtask"],
                    "lang": result_row["lang"],
                    "train_f1": result_row["train_f1"],
                    "dev_f1": result_row["dev_f1"],
                    "test_f1": result_row["test_f1"],
                    "n_ling": result_row["n_ling"],
                    "n_style": result_row["n_style"],
                    "n_total": result_row["n_total"],
                    "run_time_sec": f"{elapsed_sec:.2f}",
                    "source_results_tsv": source_results_tsv,
                }
            )

    timestamped_out, latest_out = write_combined_results(combined_rows)
    print("\n" + "=" * 88)
    print("Combined LingRF run table saved")
    print(f"- {timestamped_out}")
    print(f"- {latest_out}")
    print("=" * 88)


if __name__ == "__main__":
    main()

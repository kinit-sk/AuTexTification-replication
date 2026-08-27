"""Run `training.py` across a grid and report F1 plus wall-clock duration."""

import subprocess
import sys
import argparse
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training.py"

DEFAULT_SEEDS = [10, 11, 12]


def parse_args():
    parser = argparse.ArgumentParser(description="Run training with multiple seeds")
    parser.add_argument(
        "--subtask",
        type=str,
        default="subtask_1",
        help="subtask_1, subtask_2, all, or comma-separated",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="en, es, all, or comma-separated",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="pred_flm",
        help="pred, flm, pred_flm, pred_flm_add, all, or comma-separated",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="baseline, multilingual, all, or comma-separated",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated seeds, e.g. 10,11,12",
    )
    return parser.parse_args()


def expand_arg(value, all_options):
    """Expand 'all' or comma-separated values to a list."""
    if value == "all":
        return all_options
    return [v.strip() for v in value.split(",")]


def run_single(subtask, lang, variant, config, seed):
    """Run one training job and return parsed metrics with elapsed seconds."""
    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--subtask",
        subtask,
        "--lang",
        lang,
        "--model_variant",
        variant,
        "--config",
        config,
        "--seed",
        str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {subtask} | {lang} | {variant} | {config} | seed={seed}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    started_at = time.perf_counter()
    output_lines = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    process.wait()
    elapsed_sec = time.perf_counter() - started_at
    output = "".join(output_lines)

    match = re.search(r"dev=([0-9.]+)\s*\|\s*test=([0-9.]+)", output)
    if match:
        dev_f1 = float(match.group(1))
        test_f1 = float(match.group(2))
        print(
            f"\n  => RESULT: dev_f1={dev_f1:.4f}, test_f1={test_f1:.4f}, time_sec={elapsed_sec:.1f}"
        )
        return {
            "dev_f1": dev_f1,
            "test_f1": test_f1,
            "time_sec": elapsed_sec,
            "success": True,
        }

    print(f"\n  => FAILED to parse results, time_sec={elapsed_sec:.1f}")
    return {"time_sec": elapsed_sec, "success": False}


def main():
    """Execute the selected experiment grid and write aggregated summaries."""
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    subtasks = expand_arg(args.subtask, ["subtask_1", "subtask_2"])
    langs = expand_arg(args.lang, ["en", "es"])
    variants = expand_arg(args.model_variant, ["pred", "flm", "pred_flm", "pred_flm_add"])
    configs = expand_arg(args.config, ["baseline", "multilingual"])

    print("=" * 70)
    print("MULTI-SEED EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Subtasks: {subtasks}")
    print(f"Languages: {langs}")
    print(f"Variants: {variants}")
    print(f"Configs: {configs}")
    print("=" * 70)

    results = defaultdict(list)
    detailed_rows = []
    sweep_started_at = time.perf_counter()

    for subtask in subtasks:
        for lang in langs:
            for variant in variants:
                for config in configs:
                    key = (subtask, lang, variant, config)
                    for seed in seeds:
                        res = run_single(subtask, lang, variant, config, seed)
                        detailed_rows.append(
                            {
                                "subtask": subtask,
                                "lang": lang,
                                "variant": variant,
                                "config": config,
                                "seed": seed,
                                "success": res["success"],
                                "dev_f1": res.get("dev_f1", np.nan),
                                "test_f1": res.get("test_f1", np.nan),
                                "time_sec": res.get("time_sec", np.nan),
                            }
                        )
                        if res["success"]:
                            results[key].append(res)

    print("\n" + "=" * 90)
    print("AGGREGATED RESULTS (mean ± std)")
    print("=" * 90)
    print(
        f"{'Subtask':<12} {'Lang':<6} {'Variant':<14} {'Config':<14} "
        f"{'Dev F1':<18} {'Test F1':<18} {'Time (s)':<16} {'N'}"
    )
    print("-" * 90)

    for (subtask, lang, variant, config), runs in sorted(results.items()):
        if not runs:
            continue

        dev_scores = [r["dev_f1"] for r in runs]
        test_scores = [r["test_f1"] for r in runs]
        time_scores = [r["time_sec"] for r in runs]

        dev_mean, dev_std = np.mean(dev_scores), np.std(dev_scores)
        test_mean, test_std = np.mean(test_scores), np.std(test_scores)
        time_mean, time_std = np.mean(time_scores), np.std(time_scores)

        print(
            f"{subtask:<12} {lang:<6} {variant:<14} {config:<14} "
            f"{dev_mean:.4f}±{dev_std:.4f}   {test_mean:.4f}±{test_std:.4f}   "
            f"{time_mean:.1f}±{time_std:.1f}   {len(runs)}"
        )

    out_dir = PROJECT_ROOT / "data" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seeds_summary.tsv"
    detailed_out_path = out_dir / "seeds_detailed_runs.tsv"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "subtask\tlang\tvariant\tconfig\tdev_mean\tdev_std\ttest_mean\ttest_std\t"
            "time_mean_sec\ttime_std_sec\tn_runs\n"
        )
        for (subtask, lang, variant, config), runs in sorted(results.items()):
            if not runs:
                continue
            dev_scores = [r["dev_f1"] for r in runs]
            test_scores = [r["test_f1"] for r in runs]
            time_scores = [r["time_sec"] for r in runs]
            dev_mean, dev_std = np.mean(dev_scores), np.std(dev_scores)
            test_mean, test_std = np.mean(test_scores), np.std(test_scores)
            time_mean, time_std = np.mean(time_scores), np.std(time_scores)
            f.write(
                f"{subtask}\t{lang}\t{variant}\t{config}\t"
                f"{dev_mean:.6f}\t{dev_std:.6f}\t{test_mean:.6f}\t{test_std:.6f}\t"
                f"{time_mean:.2f}\t{time_std:.2f}\t{len(runs)}\n"
            )

    with open(detailed_out_path, "w", encoding="utf-8") as f:
        f.write("subtask\tlang\tvariant\tconfig\tseed\tsuccess\tdev_f1\ttest_f1\ttime_sec\n")
        for row in detailed_rows:
            f.write(
                f"{row['subtask']}\t{row['lang']}\t{row['variant']}\t{row['config']}\t{row['seed']}\t"
                f"{int(row['success'])}\t{row['dev_f1']}\t{row['test_f1']}\t{row['time_sec']:.2f}\n"
            )

    total_elapsed_sec = time.perf_counter() - sweep_started_at
    print(f"\nSaved summary to: {out_path}")
    print(f"Saved detailed run log to: {detailed_out_path}")
    print(f"Total sweep wall-clock time: {total_elapsed_sec:.1f}s")


if __name__ == "__main__":
    main()
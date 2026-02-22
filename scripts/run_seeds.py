"""Runs training.py across multiple seeds and aggregates dev/test F1 (mean ± std)."""

import subprocess
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training.py"

DEFAULT_SEEDS = [10, 42, 123]


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
        help="Comma-separated seeds, e.g. 10,42,123",
    )
    return parser.parse_args()


def expand_arg(value, all_options):
    """Expand 'all' or comma-separated values to a list."""
    if value == "all":
        return all_options
    return [v.strip() for v in value.split(",")]


def run_single(subtask, lang, variant, config, seed):
    """Run training.py once and parse dev/test F1 from stdout."""
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
    output = "".join(output_lines)

    match = re.search(r"dev=([0-9.]+)\s*\|\s*test=([0-9.]+)", output)
    if match:
        dev_f1 = float(match.group(1))
        test_f1 = float(match.group(2))
        print(f"\n  => RESULT: dev_f1={dev_f1:.4f}, test_f1={test_f1:.4f}")
        return {"dev_f1": dev_f1, "test_f1": test_f1, "success": True}

    print("\n  => FAILED to parse results")
    return {"success": False}


def main():
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

    for subtask in subtasks:
        for lang in langs:
            for variant in variants:
                for config in configs:
                    key = (subtask, lang, variant, config)
                    for seed in seeds:
                        res = run_single(subtask, lang, variant, config, seed)
                        if res["success"]:
                            results[key].append(res)

    print("\n" + "=" * 90)
    print("AGGREGATED RESULTS (mean ± std)")
    print("=" * 90)
    print(
        f"{'Subtask':<12} {'Lang':<6} {'Variant':<14} {'Config':<14} "
        f"{'Dev F1':<18} {'Test F1':<18} {'N'}"
    )
    print("-" * 90)

    for (subtask, lang, variant, config), runs in sorted(results.items()):
        if not runs:
            continue

        dev_scores = [r["dev_f1"] for r in runs]
        test_scores = [r["test_f1"] for r in runs]

        dev_mean, dev_std = np.mean(dev_scores), np.std(dev_scores)
        test_mean, test_std = np.mean(test_scores), np.std(test_scores)

        print(
            f"{subtask:<12} {lang:<6} {variant:<14} {config:<14} "
            f"{dev_mean:.4f}±{dev_std:.4f}   {test_mean:.4f}±{test_std:.4f}   {len(runs)}"
        )

    out_path = PROJECT_ROOT / "data" / "out" / "seeds_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("subtask\tlang\tvariant\tconfig\tdev_mean\tdev_std\ttest_mean\ttest_std\tn_runs\n")
        for (subtask, lang, variant, config), runs in sorted(results.items()):
            if not runs:
                continue
            dev_scores = [r["dev_f1"] for r in runs]
            test_scores = [r["test_f1"] for r in runs]
            dev_mean, dev_std = np.mean(dev_scores), np.std(dev_scores)
            test_mean, test_std = np.mean(test_scores), np.std(test_scores)
            f.write(
                f"{subtask}\t{lang}\t{variant}\t{config}\t"
                f"{dev_mean:.6f}\t{dev_std:.6f}\t{test_mean:.6f}\t{test_std:.6f}\t{len(runs)}\n"
            )

    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
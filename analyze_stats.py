import os
import json
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt


RUNS_DIR = "runs"
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROTOCOLS = ["CoT", "Socratic", "Congress", "British"]


# ------------------------------------------------------
# Load summary.json for all runs
# ------------------------------------------------------
def load_all_runs():
    run_rows = []

    for run_name in sorted(os.listdir(RUNS_DIR)):
        if not run_name.startswith("run_"):
            continue

        summary_path = os.path.join(RUNS_DIR, run_name, "summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        run_id = summary["run_id"]
        proto_data = summary["protocols"]

        for proto in PROTOCOLS:
            if proto not in proto_data:
                continue

            row = {
                "run_id": run_id,
                "protocol": proto,
                "accuracy": proto_data[proto]["accuracy"],
                "prompt_tokens": proto_data[proto]["prompt_tokens"],
                "completion_tokens": proto_data[proto]["completion_tokens"],
                "total_tokens": proto_data[proto]["total_tokens"],
            }
            run_rows.append(row)

    df = pd.DataFrame(run_rows)
    df = df.sort_values(["protocol", "run_id"])
    return df


# ------------------------------------------------------
# Compute mean, std, stderr per protocol
# ------------------------------------------------------
def compute_protocol_stats(df):
    stats = df.groupby("protocol").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_stderr=("accuracy", lambda x: x.std() / np.sqrt(len(x))),
        tokens_mean=("total_tokens", "mean"),
        tokens_std=("total_tokens", "std"),
    ).reset_index()

    stats_path = os.path.join(OUTPUT_DIR, "protocol_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"Saved protocol stats → {stats_path}")

    return stats


# ------------------------------------------------------
# Paired t-tests between protocols
# ------------------------------------------------------
def paired_test(df, proto_a, proto_b):
    """Returns t-stat, p-value, effect size, CI."""

    df_a = df[df.protocol == proto_a].sort_values("run_id")
    df_b = df[df.protocol == proto_b].sort_values("run_id")

    acc_a = df_a["accuracy"].values
    acc_b = df_b["accuracy"].values

    # Differences
    d = acc_a - acc_b

    # Paired t-test
    t_stat, p_value = st.ttest_rel(acc_a, acc_b)

    # Effect size (Cohen's d)
    effect = d.mean() / d.std(ddof=1)

    # 95% CI of mean difference
    ci_low, ci_high = st.t.interval(
        confidence=0.95,
        df=len(d) - 1,
        loc=d.mean(),
        scale=st.sem(d)
    )

    return {
        "A": proto_a,
        "B": proto_b,
        "mean_diff": d.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "effect_size": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def run_pairwise_tests(df):
    comparisons = [
        ("Congress", "British"),
        ("Congress", "Socratic"),
        ("Congress", "CoT"),
        ("British", "CoT"),
        ("Socratic", "CoT"),
        ("British", "Socratic"),
    ]

    results = []
    for A, B in comparisons:
        print(f"Running paired t-test: {A} vs {B}")
        stats = paired_test(df, A, B)
        results.append(stats)

    df_stats = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "pairwise_tests.csv")
    df_stats.to_csv(out_path, index=False)
    print(f"Saved pairwise statistical tests → {out_path}")

    return df_stats


# ------------------------------------------------------
# Plot accuracy distributions
# ------------------------------------------------------
def plot_accuracy_boxplots(df):
    plt.figure(figsize=(8, 6))
    df.boxplot(column="accuracy", by="protocol", grid=False)
    plt.title("Accuracy distribution across protocols")
    plt.suptitle("")
    plt.ylabel("Accuracy")

    out_path = os.path.join(OUTPUT_DIR, "accuracy_boxplot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved accuracy boxplot → {out_path}")
    plt.close()


# ------------------------------------------------------
# Token usage plot
# ------------------------------------------------------
def plot_token_usage(df):
    tok_stats = df.groupby("protocol")["total_tokens"].mean()

    plt.figure(figsize=(8, 6))
    tok_stats.plot(kind="bar", color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])

    plt.ylabel("Average Token Usage per Run")
    plt.title("Mean Token Usage for Each Protocol")

    out_path = os.path.join(OUTPUT_DIR, "token_usage.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved token usage plot → {out_path}")
    plt.close()


# ------------------------------------------------------
# Mean ± Standard Error accuracy plot
# ------------------------------------------------------
def plot_accuracy_mean_stderr(df):
    stats = df.groupby("protocol")["accuracy"].agg(
        mean="mean",
        stderr=lambda x: x.std(ddof=1) / np.sqrt(len(x))
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        stats.index,
        stats["mean"],
        yerr=stats["stderr"],
        capsize=5,
        color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    )

    plt.ylabel("Accuracy")
    plt.title("Mean Accuracy ± Standard Error by Protocol")
    plt.ylim(0.5, 0.75)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    out_path = os.path.join(OUTPUT_DIR, "accuracy_mean_stderr.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved mean ± stderr accuracy plot → {out_path}")
    plt.close()


# ------------------------------------------------------
# Mean ± Standard Error token usage plot
# ------------------------------------------------------
def plot_token_mean_stderr(df):
    stats = df.groupby("protocol")["total_tokens"].agg(
        mean="mean",
        stderr=lambda x: x.std(ddof=1) / np.sqrt(len(x))
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        stats.index,
        stats["mean"],
        yerr=stats["stderr"],
        capsize=5,
        color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    )

    plt.ylabel("Total Token Usage")
    plt.title("Mean Total Token Usage ± Standard Error by Protocol")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    out_path = os.path.join(OUTPUT_DIR, "token_mean_stderr.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved mean ± stderr token plot → {out_path}")
    plt.close()


# ------------------------------------------------------
# Main
# ------------------------------------------------------
if __name__ == "__main__":
    print("Loading run data...")
    df = load_all_runs()
    print(df)

    print("\nComputing protocol-level stats...")
    protocol_stats = compute_protocol_stats(df)

    print("\nRunning pairwise statistical tests...")
    pairwise = run_pairwise_tests(df)

    # print("\nGenerating accuracy boxplots...")
    # plot_accuracy_boxplots(df)

    print("\nGenerating token usage plot...")
    plot_token_usage(df)

    print("\nGenerating mean ± stderr accuracy plot...")
    plot_accuracy_mean_stderr(df)

    print("\nGenerating mean ± stderr token plot...")
    plot_token_mean_stderr(df)

    print("\nDone! All analysis saved in /analysis/")

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUTPUT_DIR = "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping from long class names → short names
NAME_MAP = {
    "AmericanCongressProtocol": "Congress",
    "BritishParliamentaryProtocol": "British",
    "SingleCoTProtocol": "CoT",
    "SocraticDialogueProtocol": "Socratic",
}

def short_name(protocol):
    for long, short in NAME_MAP.items():
        if protocol.startswith(long):
            return short
    return protocol  # fallback


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def aggregate_results():
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]

    protocol_groups = {}

    # Group files by *short* protocol name
    for filename in files:
        original_prefix = filename.split("_hotpot_qa")[0]
        protocol = short_name(original_prefix)

        protocol_groups.setdefault(protocol, []).append(
            os.path.join(RESULTS_DIR, filename)
        )

    summary_rows = []
    plot_data = []
    token_summary_rows = []

    for protocol, file_list in protocol_groups.items():
        file_list.sort()  # Ensure 0–9, 10–19 order

        total_correct = 0
        total_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        print(f"\n=== Processing {protocol} ({len(file_list)} files) ===")

        for file_path in file_list:
            data = load_json(file_path)
            if data is None:
                continue

            meta = data["metadata"]
            acc = meta["accuracy"]
            correct = meta["correct_count"]
            count = meta["total_count"]

            prompt_tok = meta.get("prompt_tokens", 0)
            comp_tok = meta.get("completion_tokens", 0)
            total_tok = meta.get("total_tokens", 0)

            total_prompt_tokens += prompt_tok
            total_completion_tokens += comp_tok

            total_correct += correct
            total_count += count

            plot_data.append({
                "protocol": protocol,
                "file": os.path.basename(file_path),
                "batch_accuracy": acc
            })

            print(f"{os.path.basename(file_path)} → acc={acc:.3f} ({correct}/{count}), tokens={total_tok}")

        overall_accuracy = total_correct / total_count if total_count > 0 else 0
        total_tokens = total_prompt_tokens + total_completion_tokens
        tokens_per_question = total_tokens / total_count if total_count > 0 else 0

        summary_rows.append({
            "protocol": protocol,
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_count,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_question": tokens_per_question,
        })

        print(f"→ FINAL {protocol}: {overall_accuracy:.3f} ({total_correct}/{total_count}), total_tokens={total_tokens}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    # Save batch data
    plot_df = pd.DataFrame(plot_data)
    batch_path = os.path.join(OUTPUT_DIR, "batch_accuracy.csv")
    plot_df.to_csv(batch_path, index=False)
    print(f"Saved batch accuracy to {batch_path}")

    return summary_df, plot_df


def plot_summary(summary_df):
    plt.figure(figsize=(8, 5))
    plt.bar(
        summary_df["protocol"], 
        summary_df["accuracy"], 
        color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    )

    for i, v in enumerate(summary_df["accuracy"]):
        pct = int(round(v * 100))
        plt.text(
            i,
            v + 0.03,
            f"{pct}%",
            ha="center",
            fontweight="bold",
            fontsize=11
        )

    plt.ylabel("Accuracy")
    plt.title("Accuracy by Protocol (HotpotQA 100 samples)")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    out_path = os.path.join(OUTPUT_DIR, "accuracy_plot.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()


def plot_token_usage(summary_df):
    plt.figure(figsize=(8, 5))

    plt.bar(
        summary_df["protocol"],
        summary_df["total_tokens"],
        color=["#59a14f", "#edc948", "#af7aa1", "#ff9da7"]
    )

    # Annotate tokens above bars
    for i, v in enumerate(summary_df["total_tokens"]):
        plt.text(
            i,
            v * 1.02,
            f"{int(v):,} tokens",
            ha="center",
            fontweight="bold",
            fontsize=10
        )

    plt.ylabel("Total Tokens Used")
    plt.title("Token Usage by Protocol")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    out_path = os.path.join(OUTPUT_DIR, "token_usage_plot.png")
    plt.savefig(out_path)
    print(f"Saved token usage plot to {out_path}")
    plt.close()


def plot_batch_accuracy(plot_df):
    plt.figure(figsize=(10, 6))

    for protocol in plot_df["protocol"].unique():
        df = plot_df[plot_df["protocol"] == protocol]
        plt.plot(df["file"], df["batch_accuracy"], marker="o", label=protocol)

    plt.xticks(rotation=90)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Batches (Every 10 Questions)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    out_path = os.path.join(OUTPUT_DIR, "batch_plot.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    summary_df, plot_df = aggregate_results()
    plot_summary(summary_df)
    plot_token_usage(summary_df)   # ← NEW FIGURE
    plot_batch_accuracy(plot_df)

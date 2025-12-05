import argparse
import os
import json
from datetime import datetime

from src.llm.providers import OpenAIProvider, AnthropicProvider, GeminiProvider
from src.llm.mock import MockProvider
from src.protocols.control import SingleCoTProtocol
from src.protocols.socratic import SocraticDialogueProtocol
from src.protocols.congress import AmericanCongressProtocol
from src.protocols.british import BritishParliamentaryProtocol
from src.experiment.runner import ExperimentRunner

from dotenv import load_dotenv
load_dotenv()


# -------------------------------------------------------------
# Helper: extract accuracy + token counts from a results file
# -------------------------------------------------------------
def load_metadata(path):
    with open(path, "r") as f:
        data = json.load(f)
        meta = data["metadata"]
        return {
            "accuracy": meta["accuracy"],
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "total_tokens": meta.get("total_tokens", 0),
        }


def main():

    parser = argparse.ArgumentParser(description="Multi-Agent Debate Experiment Runner")
    parser.add_argument("--dataset", type=str, default="hotpot_qa",
                        choices=["hotpot_qa", "cais/mmlu"], help="Dataset")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "gemini", "mock"])
    parser.add_argument("--model", type=str, help="Model name (optional)")

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Provider selection
    # -------------------------------------------------------------
    if args.provider == "openai":
        provider = OpenAIProvider(model=args.model or "gpt-4o-mini")
    elif args.provider == "anthropic":
        provider = AnthropicProvider(model=args.model or "claude-3-haiku-20240307")
    elif args.provider == "gemini":
        provider = GeminiProvider(model=args.model or "gemini-2.5-flash-lite")
    elif args.provider == "mock":
        provider = MockProvider()
    else:
        raise ValueError("Unknown provider")

    # -------------------------------------------------------------
    # 28 independent runs
    # -------------------------------------------------------------
    RUNS = 28
    RESULTS_ROOT = "runs"
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    Protocols = {
        "CoT": SingleCoTProtocol,
        "Socratic": SocraticDialogueProtocol,
        "Congress": AmericanCongressProtocol,
        "British": BritishParliamentaryProtocol,
    }

    for run_id in range(1, RUNS + 1):
        run_dir = os.path.join(RESULTS_ROOT, f"run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n==============================")
        print(f"   STARTING RUN {run_id:02d}")
        print(f"==============================")

        run_summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "protocols": {}
        }

        # Reset usage for each run (important!)
        provider.reset_usage()

        # Create a fresh runner
        runner = ExperimentRunner(provider)

        # Run all 4 protocols
        for proto_name, proto_class in Protocols.items():

            print(f"\n--- Running {proto_name} ---")

            out_path = os.path.join(run_dir, f"{proto_name}.json")

            runner.run_experiment(
                proto_class,
                dataset_name="hotpot_qa",
                limit=100,
                start=0,
                end=100,
                output_file=out_path
            )

            # Read back accuracy + token stats
            meta = load_metadata(out_path)
            run_summary["protocols"][proto_name] = meta

        # Save run summary
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2)

        print(f"\nSaved summary for run {run_id:02d} → {summary_path}")


if __name__ == "__main__":
    main()

import argparse
import os
from src.llm.providers import OpenAIProvider, AnthropicProvider, GeminiProvider
from src.llm.mock import MockProvider
from src.protocols.control import SingleCoTProtocol
from src.protocols.socratic import SocraticDialogueProtocol
from src.protocols.congress import AmericanCongressProtocol
from src.protocols.british import BritishParliamentaryProtocol
from src.experiment.runner import ExperimentRunner

import google.generativeai as genai
import os

from dotenv import load_dotenv
load_dotenv()

def main():
    
    parser = argparse.ArgumentParser(description="Multi-Agent Debate Experiment Runner")
    parser.add_argument("--protocol", type=str, required=True, choices=["control", "socratic", "congress", "british"], help="Debate protocol to run")
    parser.add_argument("--dataset", type=str, default="hotpot_qa", choices=["hotpot_qa", "cais/mmlu"], help="Dataset to use")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to run")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "gemini", "mock"], help="LLM Provider")
    parser.add_argument("--model", type=str, help="Model name (optional)")
    
    args = parser.parse_args()
    
    # Initialize Provider
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
        
    # Initialize Runner
    runner = ExperimentRunner(provider)
    
    # Select Protocol Class
    if args.protocol == "control":
        protocol_class = SingleCoTProtocol
    elif args.protocol == "socratic":
        protocol_class = SocraticDialogueProtocol
    elif args.protocol == "congress":
        protocol_class = AmericanCongressProtocol
    elif args.protocol == "british":
        protocol_class = BritishParliamentaryProtocol
        
    # Run
    runner.run_experiment(protocol_class, args.dataset, limit=args.limit)

if __name__ == "__main__":
    main()

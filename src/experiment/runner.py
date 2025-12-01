import json
import os
import string
import time
from datetime import datetime
from typing import List, Type
from ..llm.base import LLMProvider
from ..agents.agent import DebaterAgent
from ..protocols.base import DebateProtocol
from ..data.loader import DataLoader

class ExperimentRunner:
    def __init__(self, provider: LLMProvider, output_dir: str = "results"):
        self.provider = provider
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_experiment(self, 
                       protocol_class: Type[DebateProtocol], 
                       dataset_name: str, 
                       num_agents: int = 2, 
                       limit: int = 3,
                       output_file: str = None,
                       delay: float = 0.0,
                       **protocol_kwargs):
        
        loader = DataLoader(dataset_name, limit=limit)
        data = loader.load()
        
        results = []
        correct_count = 0
        total_count = 0
        processed_ids = set()

        # Determine output filepath
        if output_file:
            filepath = output_file
            # If file exists, load it to resume
            if os.path.exists(filepath):
                print(f"Resuming from {filepath}...")
                try:
                    with open(filepath, "r") as f:
                        existing_data = json.load(f)
                        # Handle both old format (list) and new format (dict with metadata)
                        if isinstance(existing_data, list):
                            results = existing_data
                        elif isinstance(existing_data, dict) and "results" in existing_data:
                            results = existing_data["results"]
                            # Restore counts if available, or recalculate
                            if "metadata" in existing_data:
                                correct_count = existing_data["metadata"].get("correct_count", 0)
                                total_count = existing_data["metadata"].get("total_count", 0)
                        
                        # Populate processed_ids. Assuming 'question' is unique enough for now if ID is missing
                        for r in results:
                            # Try to find a unique identifier. 
                            # HotpotQA has IDs, but our result might not have saved it explicitly if we didn't pass it through.
                            # Let's use question text as dedupe key for now.
                            processed_ids.add(r.get("question"))
                            
                    print(f"Loaded {len(results)} existing results.")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode {filepath}. Starting fresh.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{protocol_class.__name__}_{dataset_name}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
        
        print(f"Starting experiment with {protocol_class.__name__} on {dataset_name}...")
        print(f"Output will be saved to: {filepath}")
        
        for item in data:
            question = item["question"]
            
            if question in processed_ids:
                print(f"Skipping already processed question: {question[:30]}...")
                continue

            print(f"Processing: {question[:50]}...")
            
            # Initialize agents
            agents = []
            roles = ["Agent A", "Agent B", "Agent C"] # Generic roles
            if protocol_class.__name__ == "SingleCoTProtocol":
                agents.append(DebaterAgent("Solver", self.provider, "You are a helpful assistant."))
            elif protocol_class.__name__ == "SocraticDialogueProtocol":
                agents.append(DebaterAgent("Student", self.provider, "You are a student trying to answer questions."))
                agents.append(DebaterAgent("Socrates", self.provider, "You are Socrates. Ask probing questions."))
            elif protocol_class.__name__ == "AmericanCongressProtocol":
                agents.append(DebaterAgent("Affirmative", self.provider, "You are the Affirmative side."))
                agents.append(DebaterAgent("Negative", self.provider, "You are the Negative side."))
            elif protocol_class.__name__ == "BritishParliamentaryProtocol":
                agents.append(DebaterAgent("Government", self.provider, "You are the Government."))
                agents.append(DebaterAgent("Opposition", self.provider, "You are the Opposition."))
            else:
                # Fallback generic agents
                for i in range(num_agents):
                    agents.append(DebaterAgent(roles[i], self.provider, f"You are {roles[i]}."))

            protocol = protocol_class()
            try:
                result = protocol.run(question, agents, **protocol_kwargs)
                result["ground_truth"] = item["answer"]
                
                # Evaluate correctness
                is_correct = self._evaluate_correctness(result.get("final_answer", ""), item["answer"])
                result["is_correct"] = is_correct
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
                results.append({"error": str(e), "question": question})

            # Save incrementally
            self._save_results(filepath, protocol_class.__name__, dataset_name, results, correct_count, total_count)
            
            if delay > 0:
                print(f"Sleeping/waiting for API rate limit for {delay} seconds...")
                time.sleep(delay)


        # Final save
        self._save_results(filepath, protocol_class.__name__, dataset_name, results, correct_count, total_count)
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"Experiment finished. Accuracy: {accuracy:.2f} ({correct_count}/{total_count}). Results saved to {filepath}")

    def _save_results(self, filepath, protocol_name, dataset_name, results, correct_count, total_count):
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        output_data = {
            "metadata": {
                "protocol": protocol_name,
                "dataset": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count
            },
            "results": results
        }
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

    def _evaluate_correctness(self, prediction: str, ground_truth: str) -> bool:
        """
        Evaluates if the prediction is correct based on the ground truth.
        Uses a simple inclusion check after normalization.
        """
        def normalize(s: str) -> str:
            return s.lower().strip(string.punctuation).strip()
            
        pred_norm = normalize(str(prediction))
        gt_norm = normalize(str(ground_truth))
        
        # For MMLU (multiple choice), ground_truth is an index (int) or letter
        # But our loader might return it as is.
        # If ground_truth is very short (like 'A', 'B', '1'), we might want exact match or check if it's the *answer*.
        # For HotpotQA, ground_truth is a string entity.
        
        if not gt_norm:
            return False
            
        return gt_norm in pred_norm

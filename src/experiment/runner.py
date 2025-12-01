import json
import os
import string
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
                       **protocol_kwargs):
        
        loader = DataLoader(dataset_name, limit=limit)
        data = loader.load()
        
        results = []
        correct_count = 0
        total_count = 0
        
        print(f"Starting experiment with {protocol_class.__name__} on {dataset_name}...")
        
        for item in data:
            question = item["question"]
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


        # Calculate metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        output_data = {
            "metadata": {
                "protocol": protocol_class.__name__,
                "dataset": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count
            },
            "results": results
        }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{protocol_class.__name__}_{dataset_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Experiment finished. Accuracy: {accuracy:.2f} ({correct_count}/{total_count}). Results saved to {filepath}")

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

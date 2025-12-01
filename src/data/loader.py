from typing import List, Dict, Any, Optional
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

class DataLoader:
    def __init__(self, dataset_name: str, split: str = "validation", limit: Optional[int] = None):
        if load_dataset is None:
            raise ImportError("datasets library not installed. Please run `pip install datasets`.")
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

    def load(self) -> List[Dict[str, Any]]:
        if self.dataset_name == "hotpot_qa":
            ds = load_dataset("hotpot_qa", "distractor", split=self.split)
            data = []
            for item in ds:
                # ctx_chunks = []
                # for title, sents in item["context"]:
                #     ctx_chunks.append(f"Title: {title}\n" + " ".join(sents))
                # context_text = "\n\n".join(ctx_chunks)

                # data.append({
                #     "id": item["id"],
                #     "question": item["question"],
                #     "answer": item["answer"],
                #     "context": context_text,
                # })

                data.append({
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": item["context"] # List of [title, sentences]
                })
        elif self.dataset_name == "cais/mmlu":
            # MMLU has many subsets, defaulting to 'abstract_algebra' for example, or all?
            # For simplicity, let's pick one or make it configurable. 
            # User might want to pass specific config.
            ds = load_dataset("cais/mmlu", "all", split=self.split)
            data = []
            for item in ds:
                data.append({
                    "id": str(item.get("id", "")), # MMLU might not have ID
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"] # index of correct choice
                })
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

        if self.limit:
            return data[:self.limit]
        return data

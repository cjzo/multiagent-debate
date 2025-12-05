from typing import List, Dict, Any
from .base import DebateProtocol
from ..agents.agent import DebaterAgent

class SingleCoTProtocol(DebateProtocol):
    """Control protocol: Single agent with Chain-of-Thought."""

    def run(self, question: str, agents: List[DebaterAgent], context: str = "", **kwargs) -> Dict[str, Any]:
        if len(agents) != 1:
            raise ValueError("SingleCoTProtocol requires exactly one agent.")
        
        agent = agents[0]
        prompt = (
            "You are answering a question from a QA benchmark.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "1. Think step by step and write a short reasoning (no more than 4 sentences).\n"
            "2. Then, on a new line, give your final answer in EXACTLY this format:\n"
            "Final Answer: <one short answer only (for example: 'yes', 'no', a name, or a short noun phrase)>\n\n"
            "Do NOT include any extra text after that final answer line, and do NOT mention using tools or browsing."
        )
        response = agent.speak(prompt)
        
        return {
            "protocol": "SingleCoT",
            "question": question,
            "transcript": [{"role": agent.name, "content": response}],
            "final_answer": self._extract_final_answer(response)
        }



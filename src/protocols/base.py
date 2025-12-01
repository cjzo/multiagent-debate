from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..agents.agent import DebaterAgent

class DebateProtocol(ABC):
    """Abstract base class for debate protocols."""

    @abstractmethod
    def run(self, question: str, agents: List[DebaterAgent], **kwargs) -> Dict[str, Any]:
        """
        Runs the debate protocol.

        Args:
            question: The question or topic to debate.
            agents: A list of DebaterAgent instances participating in the debate.
            **kwargs: Additional protocol-specific arguments.

        Returns:
            A dictionary containing the results of the debate (e.g., final answer, transcript).
        """
        pass

    def _extract_final_answer(self, text: str) -> str:
        """
        Extracts the final answer from the text.
        Looks for 'Final Answer: <answer>'.
        """
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()
        return text

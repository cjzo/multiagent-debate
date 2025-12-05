from typing import List, Dict, Any
from .base import DebateProtocol
from ..agents.agent import DebaterAgent

class SocraticDialogueProtocol(DebateProtocol):
    """Socratic Dialogue: One agent answers, another questions/corrects."""

    def run(self, question: str, agents: List[DebaterAgent], context: str = "", rounds: int = 1, **kwargs) -> Dict[str, Any]:
        if len(agents) != 2:
            raise ValueError("SocraticDialogueProtocol requires exactly two agents (Student, Socrates).")
        
        student, socrates = agents
        transcript = []
        
        # 1) Initial attempt by Student
        student_prompt = (
            "You are a careful but concise student answering a question in a QA exam.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "1. Think step by step and write a short reasoning (no more than 4 sentences).\n"
            "2. Then on a new line, write: Provisional Answer: <your best short answer in a few words "
            "(e.g. 'yes', 'no', a name, or a short noun phrase)>.\n"
            "Do NOT browse the web; rely only on your knowledge and reasoning."
        )
        student_response = student.speak(student_prompt)
        transcript.append({"role": student.name, "content": student_response})
        
        current_context = f"Question: {question}\nStudent's Answer:\n{student_response}"
        
        # 2) Socratic rounds
        for i in range(rounds):
            # Socrates: ask one targeted factual question
            socrates_prompt = (
                f"{current_context}\n\n"
                "You are Socrates helping the student improve factual accuracy on this QA task.\n"
                "Your job:\n"
                "- Look for ONE concrete possible mistake, missing link, or ambiguity in the "
                "student's reasoning or Provisional Answer, directly related to the question.\n"
                "- Ask exactly ONE short question that targets that issue and could help the student "
                "correct or refine the answer.\n"
                "- Do NOT explain the answer, do NOT introduce long philosophical reflections, and do NOT "
                "discuss why the question might have been asked.\n\n"
                "Output ONLY your question, nothing else."
            )
            socrates_response = socrates.speak(socrates_prompt)
            transcript.append({"role": socrates.name, "content": socrates_response})
            student.listen(socrates_response)
            
            current_context += f"\nSocrates: {socrates_response}"
            
            # Student revises answer
            student_reply_prompt = (
                f"You are revising your answer to a QA exam question.\n\n"
                f"Question: {question}\n"
                f"Socrates asked: {socrates_response}\n\n"
                "1. Briefly update or defend your reasoning in at most 3 sentences.\n"
                "2. If needed, update your Provisional Answer.\n"
                "3. End with: Provisional Answer: <your best updated short answer in a few words>.\n"
                "Keep everything focused strictly on answering the question correctly."
            )
            student_response = student.speak(student_reply_prompt)
            transcript.append({"role": student.name, "content": student_response})
            socrates.listen(student_response)
            
            current_context += f"\nStudent: {student_response}"

        # 3) Final succinct answer
        summary_prompt = (
            "You are now giving the final answer to a QA benchmark question.\n\n"
            f"Question: {question}\n\n"
            "Here is your previous reasoning and dialogue with Socrates:\n"
            f"{current_context}\n\n"
            "Using that, output exactly ONE line in this format:\n"
            "Final Answer: <one short answer only (e.g. 'yes', 'no', a name, or a short noun phrase)>\n\n"
            "Do NOT include any reasoning, explanations, or extra text. Only that one line."
        )
        final_response = student.speak(summary_prompt)
        transcript.append({"role": student.name, "content": final_response})

        return {
            "protocol": "SocraticDialogue",
            "question": question,
            "transcript": transcript,
            "final_answer": self._extract_final_answer(final_response),
        }

from typing import List, Dict, Any
from .base import DebateProtocol
from ..agents.agent import DebaterAgent

class AmericanCongressProtocol(DebateProtocol):
    """American Congress: Long-form speeches, but oriented around factual QA correctness."""

    def run(self, question: str, agents: List[DebaterAgent], context: str = "", rounds: int = 2, **kwargs) -> Dict[str, Any]:
        if len(agents) != 2:
            raise ValueError("AmericanCongressProtocol requires exactly two agents (Affirmative, Negative).")
        
        aff, neg = agents
        transcript = []

        # --- Round 1: Opening Statements ---

        # Affirmative Opening
        aff_prompt = (
            "You are the Affirmative side in a congressional-style debate on a QA benchmark question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Your job:\n"
            "1. Commit to a concrete short answer in the format: Proposed Answer: <short answer>.\n"
            "   - The short answer should be just 'yes', 'no', a name, or a short noun phrase.\n"
            "2. Then justify your Proposed Answer with clear, factual reasoning in 3–6 sentences.\n"
            "3. Focus on accuracy, not rhetoric. Do NOT mention using tools or browsing the web.\n\n"
            "End your response with a line starting with exactly: Proposed Answer: "
        )
        aff_response = aff.speak(aff_prompt)
        transcript.append({"role": aff.name, "content": aff_response})
        neg.listen(aff_response)

        # Negative Opening
        neg_prompt = (
            "You are the Negative side in a congressional-style debate on a QA benchmark question.\n\n"
            f"Question: {question}\n\n"
            f"The Affirmative side said:\n{aff_response}\n\n"
            "Your job:\n"
            "1. Identify any factual errors, gaps, or unjustified assumptions in the Affirmative's reasoning.\n"
            "2. Decide whether you agree or disagree with the Affirmative's Proposed Answer.\n"
            "3. If you disagree, provide your own alternative short answer.\n"
            "4. Give at most 5 sentences of critique and reasoning.\n"
            "5. End with two lines:\n"
            "   Verdict: <AGREE or DISAGREE>\n"
            "   Opposition Proposed Answer: <short answer or SAME as Affirmative>\n"
            "Stay focused on factual correctness, not style or politics."
        )
        neg_response = neg.speak(neg_prompt)
        transcript.append({"role": neg.name, "content": neg_response})
        aff.listen(neg_response)

        # --- Subsequent Rounds: Rebuttals ---
        for i in range(rounds - 1):
            # Affirmative Rebuttal
            aff_rebuttal_prompt = (
                f"Question: {question}\n\n"
                f"The Negative side just said:\n{neg_response}\n\n"
                "You are the Affirmative side responding in a rebuttal.\n"
                "Your job:\n"
                "1. Briefly defend your original reasoning where it is still sound (2–3 sentences).\n"
                "2. Acknowledge any valid corrections from the Negative.\n"
                "3. Decide whether to keep or change your Proposed Answer.\n"
                "4. If you change it, explain why in at most 2 sentences.\n"
                "5. End with a line: Proposed Answer: <your current best short answer>.\n"
                "Keep the total response under 7 sentences."
            )
            aff_response = aff.speak(aff_rebuttal_prompt)
            transcript.append({"role": aff.name, "content": aff_response})
            neg.listen(aff_response)

            # Negative Rebuttal
            neg_rebuttal_prompt = (
                f"Question: {question}\n\n"
                f"The Affirmative side just said:\n{aff_response}\n\n"
                "You are the Negative side responding in a rebuttal.\n"
                "Your job:\n"
                "1. Briefly summarize the Affirmative's current position (1–2 sentences).\n"
                "2. Point out the single most important factual issue or uncertainty that remains (2–3 sentences).\n"
                "3. Decide whether you still disagree with their Proposed Answer.\n"
                "4. End with two lines:\n"
                "   Verdict: <AGREE or DISAGREE>\n"
                "   Opposition Proposed Answer: <your best short answer, or SAME as Affirmative>.\n"
                "Keep everything tightly focused on which short answer is most likely correct."
            )
            neg_response = neg.speak(neg_rebuttal_prompt)
            transcript.append({"role": neg.name, "content": neg_response})
            aff.listen(neg_response)

        # --- Final Answer Step (Affirmative wraps up) ---
        final_prompt = (
            "You are the Affirmative side giving a final, concise answer to a QA benchmark question "
            "after a congressional-style debate.\n\n"
            f"Question: {question}\n\n"
            "Here is the debate transcript (your speeches and the Negative's speeches):\n"
            f"{transcript}\n\n"
            "Based on all arguments and critiques, decide the single best short answer to the question.\n"
            "Output exactly ONE line in this format:\n"
            "Final Answer: <one short answer only (e.g. 'yes', 'no', a name, or a short noun phrase)>\n\n"
            "Do NOT include any reasoning, explanations, or extra text."
        )
        final_response = aff.speak(final_prompt)
        transcript.append({"role": aff.name, "content": final_response})

        return {
            "protocol": "AmericanCongress",
            "question": question,
            "transcript": transcript,
            "final_answer": self._extract_final_answer(final_response),
        }

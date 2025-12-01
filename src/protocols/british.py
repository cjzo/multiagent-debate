from typing import List, Dict, Any
from .base import DebateProtocol
from ..agents.agent import DebaterAgent

class BritishParliamentaryProtocol(DebateProtocol):
    """British Parliamentary: Streaming speech with interruption (POI), tuned for QA."""

    def run(self, question: str, agents: List[DebaterAgent], context: str = "", rounds: int = 2, **kwargs) -> Dict[str, Any]:
        if len(agents) != 2:
            raise ValueError("BritishParliamentaryProtocol requires exactly two agents (Government, Opposition).")
        
        gov, opp = agents
        transcript = []
        
        # --- Government Opening (streamed) ---
        gov_prompt = (
            "You are the Government side in a British Parliamentary style debate on a QA benchmark question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Your job:\n"
            "1. Decide on a concrete short answer to the question.\n"
            "2. Give an opening speech of 4–7 sentences supporting that answer with clear, factual reasoning.\n"
            "3. Somewhere in your speech include a single line of the form:\n"
            "   Proposed Answer: <one short answer (e.g. 'yes', 'no', a name, or a short noun phrase)>\n"
            "4. Avoid repetition and rhetorical fluff; focus on correctness.\n"
            "Do NOT mention using tools or browsing the web."
        )
        
        # Streaming generation
        stream = gov.speak_stream(gov_prompt)
        
        full_speech = ""
        current_chunk = ""
        interruption_count = 0
        
        transcript.append({"role": "System", "content": f"Government ({gov.name}) starts speaking."})
        
        for token in stream:
            full_speech += token
            current_chunk += token
            
            # Check for interruption every ~sentence or when chunk is long enough
            if len(current_chunk) > 120 or (current_chunk.strip().endswith(('.', '?', '!')) and len(current_chunk) > 40):
                # Ask Opposition if they want to interrupt
                opp_check_prompt = (
                    f"Question: {question}\n\n"
                    f"The Government just said (this part of their speech):\n\"{current_chunk.strip()}\"\n\n"
                    "You are the Opposition in a British Parliamentary debate, but this is your PRIVATE thinking, "
                    "not a public speech.\n"
                    "Your job now:\n"
                    "- Decide if there is a clear factual error, missing key fact, or strong counterpoint that would "
                    "significantly change which short answer is most likely correct.\n"
                    "- If YES, respond EXACTLY in this format:\n"
                    "  INTERRUPT: <one short sentence POI pointing out that issue>\n"
                    "- If NO, respond with exactly:\n"
                    "  NO\n\n"
                    "No extra text, no explanations beyond that format."
                )
                decision = opp.speak(opp_check_prompt)
                
                if "INTERRUPT:" in decision:
                    reason = decision.split("INTERRUPT:", 1)[1].strip()
                    transcript.append({"role": opp.name, "content": f"POI: {reason}"})
                    interruption_count += 1
                    
                    # Government has to address it briefly
                    gov_address_prompt = (
                        f"Question: {question}\n\n"
                        f"The Opposition offered a Point of Information (POI): {reason}\n\n"
                        "You are the Government. Briefly address this POI in at most 3 sentences, "
                        "clarifying or correcting your reasoning, and then continue your speech mentally.\n"
                        "Do NOT restate your whole case; just answer the POI.\n"
                        "This is a public response to the POI."
                    )
                    gov_response = gov.speak(gov_address_prompt)
                    transcript.append({"role": gov.name, "content": f"Response to POI: {gov_response}"})
                    
                    # Reset chunk after interruption handling
                    current_chunk = "" 
                else:
                    # No interruption; reset chunk to simulate focusing on the next segment
                    current_chunk = ""

        transcript.append({"role": gov.name, "content": f"Full Speech: {full_speech}"})
        
        # --- Opposition Rebuttal (non-streaming) ---
        opp_rebuttal_prompt = (
            "You are the Opposition side in a British Parliamentary style debate on a QA benchmark question.\n\n"
            f"Question: {question}\n\n"
            f"The Government's full speech was:\n{full_speech}\n\n"
            "Your job:\n"
            "1. Identify the single most important factual disagreement or alternative answer to the question.\n"
            "2. Give a closing rebuttal of 4–7 sentences focusing on why an alternative answer is more likely correct.\n"
            "3. At the end, on a new line, output:\n"
            "   Opposition Proposed Answer: <one short answer (or SAME if you agree with Government)>\n"
            "Keep the focus on factual correctness, not style."
        )
        opp_response = opp.speak(opp_rebuttal_prompt)
        transcript.append({"role": opp.name, "content": opp_response})

        # --- Final Answer Step (Government wraps up) ---
        final_prompt = (
            "You are the Government side giving a final, concise answer after a British Parliamentary style debate.\n\n"
            f"Question: {question}\n\n"
            "You have already given an opening speech and responded to Points of Information. "
            "The Opposition has given a rebuttal:\n"
            f"{opp_response}\n\n"
            "Based on all arguments, decide the single best short answer to the question.\n\n"
            "Output exactly ONE line in this format:\n"
            "Final Answer: <one short answer only (e.g. 'yes', 'no', a name, or a short noun phrase)>\n\n"
            "Do NOT include any reasoning, explanations, or extra text."
        )
        final_response = gov.speak(final_prompt)
        transcript.append({"role": gov.name, "content": final_response})

        return {
            "protocol": "BritishParliamentary",
            "question": question,
            "transcript": transcript,
            "interruptions": interruption_count,
            "final_answer": self._extract_final_answer(final_response),
        }

"""
LLM-judge prompt for SpeechAgentEval.

The judge receives one pipeline run (ground truth transcript, ASR output,
and the agent's response) and classifies it against the failure taxonomy
defined in TAXONOMY.md. It must return strict JSON so results are easy to
aggregate and to compare against manual labels during validation.

Design notes:
- One category per axis (B and C are independent axes; a run can be both
  e.g. B1 + C1). A/D are also independent and detected separately
  (A programmatically from WER, D would need a separate audio-based check
  and is out of scope for the text-based judge below).
- The judge only sees B/C, since A is computed automatically from WER/CER
  and D requires listening to synthesized audio (handled separately, e.g.
  via a second audio-input judge call or manual spot-check).
- Few-shot examples are included because B1 vs B2 vs B3, and C1 vs C3, are
  the distinctions most likely to be inconsistent without anchoring.
"""

SYSTEM_PROMPT = """You are an evaluation judge for a voice-agent pipeline (ASR -> LLM -> TTS).
You will be shown:
1. The ground-truth transcript (what the speaker actually said)
2. The ASR transcript (what the speech recognizer produced, which may contain errors)
3. The agent's response (what the LLM said back)

Classify the interaction using the categories below. Only classify axes B and C —
do not evaluate ASR accuracy directly, that is computed separately.

AXIS B — Propagation (only applies if the ASR transcript differs from ground truth
in a way that changes meaning; if ASR is essentially correct, label B as "not_applicable")
- B1 silent_propagation: The ASR error changed the meaning, and the agent acted on
  the wrong meaning WITHOUT showing any sign of uncertainty or asking for clarification.
- B2 graceful_degradation: The ASR error is present, but the agent's response is still
  correct, harmless, or inconsequential despite the error.
- B3 correct_rejection: The agent noticed the input was garbled/ambiguous and asked
  for clarification instead of guessing.

AXIS C — Agent behavior (always applies, independent of ASR correctness)
- C1 false_confidence: The agent gave a confident, specific answer to input that was
  genuinely ambiguous or unintelligible — it should have hedged or asked instead.
- C2 no_repair: The agent has no clarification/fallback strategy at all — it just
  produces some response regardless of input quality, with no mechanism to flag
  uncertainty.
- C3 overcorrection: The agent asked for clarification even though the input was
  actually clear and answerable. Not dangerous, but a false-positive friction cost.
- C4 appropriate: The agent's behavior was appropriate given the input — confident
  when it should be confident, asks when it should ask. (Use this when none of
  C1-C3 apply.)

IMPORTANT: many inputs in this dataset are a single isolated word with no
surrounding context (e.g. "Paragraph", "Copy", "Alt"). A single word alone is
often GENUINELY ambiguous as a command, even when the ASR transcribed it
perfectly — asking for clarification in that case is C4 (appropriate), NOT C3
(overcorrection). Only label C3 if the input was clearly a complete,
actionable request and the agent still asked for clarification unnecessarily.

Also note: if the agent's response mentions or references the (possibly
wrong) transcribed word while still asking a question and NOT committing to
an action based on it (e.g. "What would you like to know about gold?"), this
is still B3 (correct rejection) — merely repeating the word back inside a
clarifying question is not the same as silently acting on it as fact. Only
label B1 if the agent proceeds as though the wrong transcript were true
(e.g. answering it, taking an action, or treating it as settled).

CRITICAL DISTINCTION for borderline cases: a response counts as B1 (not B3)
if it PRESUPPOSES the mistranscribed content is true and builds a natural
follow-up on that presupposition, even if phrased as a question — e.g. "What
do you want to send?" presupposes sending was the real intent; "I'm glad you
loved it! Anything else?" presupposes genuine positive feedback was given.
These are B1 because the agent never questions whether it heard correctly —
it just moves the conversation forward as if the wrong content were real.
Contrast with true B3, where the response questions the input itself (e.g.
"did you say X?", "I'm not sure I understood that", "there might have been
a transcription error").

This includes a specific and easy-to-miss sub-case: the agent silently
reinterprets a failed/garbled command as ordinary small talk and responds
warmly or socially (e.g. transcribed nonsense becomes "How are you?" and the
agent replies "I'm good, thanks for asking!", or becomes "See you" and the
agent replies "Goodbye!"). This is B1 + C1 — the agent never recognizes a
command failed to register at all; it just silently pivots into a different,
fabricated conversational frame and answers confidently within it.

Return ONLY valid JSON, no preamble, no markdown fences, matching exactly this schema:
{
  "b_label": "B1" | "B2" | "B3" | "not_applicable",
  "c_label": "C1" | "C2" | "C3" | "C4",
  "confidence": "high" | "medium" | "low",
  "justification": "<one sentence, max 25 words>"
}
"""

FEW_SHOT_EXAMPLES = [
    {
        "ground_truth": "call mom",
        "asr_transcript": "call bomb",
        "agent_response": "I've placed a call to your contact 'Bomb'.",
        "label": {
            "b_label": "B1",
            "c_label": "C1",
            "confidence": "high",
            "justification": "ASR error changed meaning entirely and agent acted on it confidently with no check.",
        },
    },
    {
        "ground_truth": "stop",
        "asr_transcript": "stop",
        "agent_response": "Stopping now.",
        "label": {
            "b_label": "not_applicable",
            "c_label": "C4",
            "confidence": "high",
            "justification": "ASR was correct and agent responded appropriately.",
        },
    },
    {
        "ground_truth": "volume up",
        "asr_transcript": "volume of",
        "agent_response": "I'm not sure I caught that — did you want me to change the volume?",
        "label": {
            "b_label": "B3",
            "c_label": "C4",
            "confidence": "high",
            "justification": "ASR error present but agent detected ambiguity and asked for clarification.",
        },
    },
    {
        "ground_truth": "turn on the lights",
        "asr_transcript": "turn on the lice",
        "agent_response": "Turning on the lights now.",
        "label": {
            "b_label": "B2",
            "c_label": "C4",
            "confidence": "medium",
            "justification": "ASR error present but agent's response was still correct despite it.",
        },
    },
    {
        "ground_truth": "play music",
        "asr_transcript": "play music",
        "agent_response": "Could you please clarify what you'd like me to do?",
        "label": {
            "b_label": "not_applicable",
            "c_label": "C3",
            "confidence": "medium",
            "justification": "Input was clear and unambiguous but agent asked for clarification anyway.",
        },
    },
    {
        "ground_truth": "Paragraph",
        "asr_transcript": "Paragraph.",
        "agent_response": "Could you please provide more context or a specific request?",
        "label": {
            "b_label": "not_applicable",
            "c_label": "C4",
            "confidence": "high",
            "justification": "ASR correct, but a single isolated word with no object/verb is genuinely ambiguous as a command — asking is appropriate, not overcorrection.",
        },
    },
    {
        "ground_truth": "Alt",
        "asr_transcript": "Gold.",
        "agent_response": "Could you please provide more context? What would you like to know about gold?",
        "label": {
            "b_label": "B3",
            "c_label": "C4",
            "confidence": "medium",
            "justification": "Response mentions the mistranscribed word but does not act on it or commit to it as fact — it's still asking, not executing, so this counts as correct rejection, not silent propagation.",
        },
    },
    {
        "ground_truth": "Charlie",
        "asr_transcript": "How are you?",
        "agent_response": "I'm good, thanks for asking! How can I help you today?",
        "label": {
            "b_label": "B1",
            "c_label": "C1",
            "confidence": "high",
            "justification": "Agent silently reinterprets a failed command as small talk and answers warmly and confidently, never flagging that no real command was understood.",
        },
    },
]


def build_user_prompt(ground_truth: str, asr_transcript: str, agent_response: str) -> str:
    examples_block = "\n\n".join(
        f"Example:\nGround truth: {ex['ground_truth']}\nASR transcript: {ex['asr_transcript']}\n"
        f"Agent response: {ex['agent_response']}\nLabel: {ex['label']}"
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"""{examples_block}

Now classify this interaction:
Ground truth: {ground_truth}
ASR transcript: {asr_transcript}
Agent response: {agent_response}
Label:"""

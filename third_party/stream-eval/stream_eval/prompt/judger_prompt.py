judger_prompt = '''You are an expert evaluator judging whether a model's answer provides a reasonable and factually plausible explanation that directly addresses the question, based on the reference answer.

**Evaluation Guideline:**
- Focus on whether the model gives a coherent reason that logically explains what the question asks.
- The answer does not need to reproduce all details from the reference — it only needs to offer a factually grounded and relevant cause.
- An answer that captures the essential reason should be considered strong, even if it omits descriptive details.
- Accept simplified, rephrased, or high-level reasoning as long as it is consistent with the reference, plausibly explains the phenomenon in the question, and does not contradict known facts.
- Do not deduct points for omitting secondary or illustrative details when the core causal logic is present, or for using concise or abstract phrasing.
- Only penalize if the explanation is factually wrong, fails to provide a meaningful cause, or is so vague that it does not actually answer the question.

**Scoring (integer 0–5):**
- 5: Fully accurate and complete explanation.
- 4: Correct and logically sufficient explanation; may omit non-essential details but captures the essential reason.
- 3: Partially relevant but weakens or misses part of the core causal link.
- 2: Tangential or speculative without solid grounding.
- 1: Factually incorrect.
- 0: No attempt to answer or completely off-topic.

**Output Format:**
Return a valid JSON object with exactly two keys:
- "explanation": one sentence focusing on whether the answer gives a reasonable and relevant reason for the question
- "score": an integer from 0 to 5

Output only the JSON. No other text, markdown, or commentary.

**Inputs:**
- Question: {question}
- Correct Answer: {answer}
- Predicted Answer: {prediction}
'''

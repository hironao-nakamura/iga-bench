from __future__ import annotations


def build_prompt(premises: list[dict], question: str, prompt_mode: str = 'forced_step') -> str:
    premise_lines = [f"{i}. {p['text']}" for i, p in enumerate(premises, 1)]
    premises_text = "\n".join(premise_lines)

    common = (
        "Treat every invented predicate as distinct and unrelated unless explicitly linked by a premise.\n"
    )

    if prompt_mode == 'free_form':
        return f"""Given the following premises:\n{premises_text}\n\nQuestion: {question}\n\n{common}Reason step by step before answering. End with `Answer: True` or `Answer: False`."""

    if prompt_mode == 'forced_step_atomic_v2':
        return f"""Given the following premises:\n{premises_text}\n\nQuestion: {question}\n\n{common}Solve step by step.
Each step MUST be a single atomic conclusion in one of these forms only:
- X is a Y
- X is not a Y
- All X are Y
- No X is a Y

For each step, write EXACTLY:
Step 1: [atomic conclusion].
Step 2: [atomic conclusion].
...

Do NOT include explanations or rationale in any step.
Do NOT use "because", "therefore", "by premise", or similar justification text.
If no new conclusion can be produced, stop writing steps.

After all steps, write exactly:
Answer: True
or
Answer: False"""

    if prompt_mode == 'proofwriter_forced_step_atomic_v1':
        return f"""Given the following premises:\n{premises_text}\n\nQuestion: {question}\n\n{common}Solve step by step.
Each step MUST be exactly one atomic statement in one of these forms:
- X is a Y
- X is not a Y
- All X are Y
- No X is a Y

Write steps only as:
Step 1: [atomic statement]
Step 2: [atomic statement]
...

No explanations, no rationale text.
Then write exactly one line:
Answer: True
or
Answer: False"""

    if prompt_mode == 'proofwriter_free_form_v1':
        return f"""Given the following premises:\n{premises_text}\n\nQuestion: {question}\n\n{common}Reason briefly and provide the final answer.
End with exactly:
Answer: True
or
Answer: False"""

    return f"""Given the following premises:\n{premises_text}\n\nQuestion: {question}\n\n{common}Solve step by step. For each step, write EXACTLY in this format:\nStep 1: [Your conclusion in one sentence].\nStep 2: [Your conclusion in one sentence].\n...\n\nAfter all steps, write exactly:\nAnswer: True\nor\nAnswer: False"""

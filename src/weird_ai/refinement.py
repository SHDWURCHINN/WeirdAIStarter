"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    # 1. Read score_key from the evaluation dictionary.
    # 2. Convert it to a float.
    # 3. If the key is missing or cannot be converted, return 0.0.
    try:
        val = evaluation.get(score_key, 0.0)
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    # 1. Include the original prompt or topic.
    # 2. Include the current lyrics.
    # 3. Include the current overall score.
    # 4. Include any available evaluation details, such as rhyme, syllables, and structure.
    # 5. Include focus items, such as stronger rhymes or more consistent syllables.
    # 6. Include preserve items, such as topic, tone, and approximate length.
    # 7. Ask the model to return only the revised lyrics.
    score = get_score(evaluation, score_key)
    
    eval_lines = [f"- {score_key}: {score}"]
    for k, v in evaluation.items():
        if k != score_key:
            eval_lines.append(f"- {k}: {v}")
    eval_str = "\n".join(eval_lines)

    focus_str = ", ".join(focus) if focus else "stronger rhymes and more consistent syllables"
    preserve_str = ", ".join(preserve) if preserve else "topic, tone, and approximate length"

    return f"""
Original task: {original_prompt}
Current lyrics:
{current_text}

Current evaluation:
{eval_str}

Focus on: {focus_str}
Preserve: {preserve_str}
Return only the revised lyrics.
""".strip()


def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    # 1. Evaluate current_text.
    prev_eval = evaluation_func(current_text)
    prev_score = get_score(prev_eval, score_key)

    # 2. Build a refinement prompt using build_refinement_prompt.
    ref_prompt = build_refinement_prompt(
        original_prompt, current_text, prev_eval, focus=focus, preserve=preserve, score_key=score_key
    )

    # 3. Generate revised text with generation_func.
    revised_text = generation_func(
        ref_prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, **kwargs
    )

    # 4. Evaluate the revised text.
    revised_eval = evaluation_func(revised_text)
    revised_score = get_score(revised_eval, score_key)

    # 5. Compare the old score and revised score.
    accepted = revised_score > prev_score

    # 6. Return a RefinementStep.
    return RefinementStep(
        iteration=0,
        prompt=ref_prompt,
        previous_text=current_text,
        revised_text=revised_text,
        previous_score=prev_score,
        revised_score=revised_score,
        accepted=accepted,
        evaluation=revised_eval,
    )


def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    # 1. Compare the current score and revised score.
    curr_score = get_score(current_evaluation, score_key)
    rev_score = get_score(revised_evaluation, score_key)

    # 2. If require_improvement is True, accept only strictly higher scores.
    # 3. If require_improvement is False, accept equal scores as well.
    if require_improvement:
        accepted = rev_score > curr_score
    else:
        accepted = rev_score >= curr_score

    # 4. Return the selected text, selected evaluation, and whether the revision was accepted.
    if accepted:
        return revised_text, revised_evaluation, True
    else:
        return current_text, current_evaluation, False


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    # 1. Stop if iteration has reached max_iterations.
    if iteration >= max_iterations:
        return False
    # 2. Stop if target_score is set and current_score is high enough.
    if target_score is not None and current_score >= target_score:
        return False
    # 3. Stop if stop_on_no_improvement is True and last_improved is False.
    if stop_on_no_improvement and not last_improved:
        return False
    # 4. Otherwise continue.
    return True


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")

    # 1. Evaluate initial_text.
    current_text = initial_text
    current_eval = evaluation_func(current_text)
    current_score = get_score(current_eval, score_key)

    history = [
        RefinementStep(
            iteration=0,
            prompt=original_prompt,
            previous_text=initial_text,
            revised_text=initial_text,
            previous_score=current_score,
            revised_score=current_score,
            accepted=True,
            evaluation=current_eval,
        )
    ]

    last_improved = True
    stopped_reason = "max_iterations"
    steps_list = []

    # 2. Repeatedly call refine_once until should_continue says to stop.
    for i in range(1, max_iterations + 1):
        if not should_continue(
            iteration=i - 1,
            max_iterations=max_iterations,
            current_score=current_score,
            target_score=target_score,
            last_improved=last_improved,
            stop_on_no_improvement=stop_on_no_improvement,
        ):
            if target_score is not None and current_score >= target_score:
                stopped_reason = "target_score"
            elif stop_on_no_improvement and not last_improved:
                stopped_reason = "no_improvement"
            else:
                stopped_reason = "max_iterations"
            break

        step = refine_once(
            original_prompt,
            current_text,
            generation_func,
            evaluation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            **kwargs,
        )
        step.iteration = i

        # 3. After each step, keep the better text using choose_better_text.
        current_text, current_eval, accepted = choose_better_text(
            current_text,
            step.revised_text,
            current_eval,
            step.evaluation,
            score_key=score_key,
            require_improvement=require_improvement,
        )

        step.accepted = accepted
        last_improved = accepted
        current_score = get_score(current_eval, score_key)

        history.append(step)
        steps_list.append(step)

    # 5. Return a result dictionary with the initial and final information.
    return {
        "initial_text": initial_text,
        "initial_score": get_score(history[0].evaluation, score_key),
        "final_text": current_text,
        "final_score": current_score,
        "iterations": steps_list,
        "stopped_reason": stopped_reason,
        "history": history,
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""
    init_score = result.get("initial_score", 0.0)
    final_score = result.get("final_score", 0.0)
    iters = result.get("iterations", [])
    num_iters = len(iters) if isinstance(iters, list) else iters
    stopped_reason = result.get("stopped_reason", "unknown")
    
    # Fallback to iterations if history is not explicitly in the dict
    history = result.get("history", iters if isinstance(iters, list) else [])

    lines = [
        f"Refinement Summary:",
        f"Initial score: {init_score:.4f}",
        f"Final score: {final_score:.4f}",
        f"Total iterations: {num_iters}",
        f"Stopped reason: {stopped_reason}",
        f"History Details:"
    ]

    for step in history:
        # Handle if items in history/iterations are RefinementStep objects or dicts
        if hasattr(step, "iteration"):
            if step.iteration == 0:
                continue
            status = "Accepted" if step.accepted else "Rejected"
            score_val = step.revised_score
            lines.append(f"  * Iteration {step.iteration}: {status} (Score: {score_val:.4f})")
        elif isinstance(step, dict):
            iter_num = step.get("iteration", 1)
            accepted = step.get("accepted", True)
            status = "Accepted" if accepted else "Rejected"
            score_val = step.get("revised_score", final_score)
            lines.append(f"  * Iteration {iter_num}: {status} (Score: {score_val:.4f})")

    return "\n".join(lines)
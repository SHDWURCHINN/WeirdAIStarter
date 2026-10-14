"""
Reasoning prompt helpers for Weird AI.

This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.

Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
    """
    Convert a normal user prompt into a general reasoning prompt.

    The returned prompt should encourage the model to think through the task
    before giving its final answer.

    Args:
        user_prompt: The original user request as a string.

    Returns:
        A formatted prompt string that includes the original task and asks for
        step-by-step reasoning.
    """
    return f"""You are Weird AI.
Think step-by-step.
Original Request: {user_prompt}

Please work on this by following the steps given here:
1. Understand the user's request
2. Identify important constraints
3. Plan the response
4. Produce the final answer

Final Answer:"""


def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.

    This prompt should guide Weird AI through a simple creative planning process
    before asking it to write the final parody lyrics.

    Args:
        topic: The parody topic or user request as a string.

    Returns:
        A formatted prompt string with step-by-step parody planning sections.
    """
    return f"""You are Weird AI, a parody songwriter.
Think step-by-step.

Topic: {topic}

Please plan your parody using the following steps:
- Identify the topic.
- Identify funny ideas or exaggerations.
- Identify possible rhymes.
- Write the final parody lyrics.

Final Parody:"""


def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.

    Closed-world reasoning means the model should answer using only the facts
    explicitly provided in the prompt. It should not use outside knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer from those premises.

    Returns:
        A formatted prompt string for closed-world reasoning.
    """
    formatted_premises = "\n".join([f"- {p}" for p in premises])
    return f"""Instructions: Use only the provided premises. Do not use outside knowledge.

Premises:
{formatted_premises}

Question: {question}

Reasoning:
Answer:"""


def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.

    Open-world reasoning allows the model to consider both the provided premises
    and relevant background knowledge. This can reveal contradictions between
    the prompt and the model's background knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer.

    Returns:
        A formatted prompt string for open-world reasoning.
    """
    formatted_premises = "\n".join([f"- {p}" for p in premises])
    return f"""Instructions: You may use relevant background knowledge. Check whether the premises conflict with known facts.

Premises:
{formatted_premises}

Question: {question}

Reasoning:
Answer:"""
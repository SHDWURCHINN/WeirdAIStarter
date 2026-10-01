"""
Instruction data formatting utilities for Weird AI.
"""


def format_input(entry):
    """
    Format one instruction example using an Alpaca-style prompt.
    This function does NOT include the response.
    """
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}"
        if entry.get("input")
        else ""
    )

    return instruction_text + input_text


def format_response(entry):
    """
    Format the expected response section.
    """
    return f"\n\n### Response:\n{entry['output']}"


def format_full_example(entry):
    """
    Format an entire instruction-response example.
    """
    return format_input(entry) + format_response(entry)


def validate_instruction_entry(entry):
    """
    Validate that an instruction dataset entry has instruction, input, and output fields.
    """
    required_keys = ["instruction", "input", "output"]
    for key in required_keys:
        if key not in entry:
            return False

    if not str(entry["instruction"]).strip() or not str(entry["output"]).strip():
        return False

    return True
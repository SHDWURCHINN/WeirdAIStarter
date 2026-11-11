"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_teacher_example(example: dict) -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """
    if not isinstance(example, dict):
        return False
    
    required_fields = ["prompt", "teacher_lyrics", "teacher_notes", "quality_score"]
    for field_name in required_fields:
        if field_name not in example:
            return False

    if not isinstance(example["prompt"], str) or not example["prompt"].strip():
        return False
    if not isinstance(example["teacher_lyrics"], str) or not example["teacher_lyrics"].strip():
        return False
    if not isinstance(example["teacher_notes"], str) or not example["teacher_notes"].strip():
        return False

    try:
        float(example["quality_score"])
    except (TypeError, ValueError):
        return False

    return True


def teacher_example_from_dict(example: dict) -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""
    if not validate_teacher_example(example):
        raise ValueError("Invalid teacher example dictionary.")
    
    data = dict(example)
    prompt = data.pop("prompt")
    teacher_lyrics = data.pop("teacher_lyrics")
    teacher_notes = data.pop("teacher_notes")
    quality_score = float(data.pop("quality_score"))
    
    metadata = data.pop("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(data)

    return TeacherExample(
        prompt=prompt,
        teacher_lyrics=teacher_lyrics,
        teacher_notes=teacher_notes,
        quality_score=quality_score,
        metadata=metadata,
    )


def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    lyrics = teacher_lyrics.strip()
    notes = teacher_notes.strip()

    if not include_thinking:
        return lyrics

    return f"<think>\n{notes}\n</think>\n\n{lyrics}"


def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""
    cleaned_prompt = prompt.strip()
    return f"User: {cleaned_prompt}\nAssistant:"


def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""
    formatted_prompt = format_training_prompt(prompt)
    cleaned_target = target_text.strip()
    return f"{formatted_prompt}\n{cleaned_target}"


def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""
    tokens = tokenizer.encode(text)
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise ValueError("Tokenizer encode must return a list of integers.")
    return tokens


def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """
    labels = list(input_ids)
    if mask_prompt:
        for i in range(min(prompt_length, len(labels))):
            labels[i] = ignore_index
    return labels


def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""
    target_text = format_teacher_response(
        example.teacher_lyrics, example.teacher_notes, include_thinking=include_thinking
    )
    formatted_prompt = format_training_prompt(example.prompt)
    
    prompt_tokens = tokenize_text(tokenizer, formatted_prompt)
    prompt_length = len(prompt_tokens)

    full_text = build_full_text(example.prompt, target_text)
    input_ids = tokenize_text(tokenizer, full_text)

    labels = create_labels(input_ids, prompt_length, mask_prompt=mask_prompt)

    return DistillationRecord(
        prompt=example.prompt,
        target_text=target_text,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=prompt_length,
        metadata=example.metadata,
    )


def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4,
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""
    filtered = []
    for ex in examples:
        if ex.quality_score < min_quality_score:
            continue
        normalized_lyrics = ex.teacher_lyrics.replace("\\n", "\n")
        lyric_lines = [line for line in normalized_lyrics.splitlines() if line.strip()]
        if len(lyric_lines) < min_lyric_lines:
            continue
        filtered.append(ex)
    return filtered


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""
    if not (0.0 <= validation_fraction <= 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * validation_fraction)
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

    return train_records, val_records


def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""
    return asdict(record)


def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            record_dict = record_to_json_dict(record)
            f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""
    path = Path(path)
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                results.append(json.loads(line_str))
    return results


def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""
    if not records:
        return "No records available."

    num_records = len(records)
    avg_input = sum(len(r.input_ids) for r in records) / num_records
    avg_prompt = sum(r.prompt_length for r in records) / num_records
    avg_target = sum(len(r.input_ids) - r.prompt_length for r in records) / num_records

    return (
        f"Total records: {num_records}\n"
        f"Average input length: {avg_input:.1f} tokens\n"
        f"Average prompt length: {avg_prompt:.1f} tokens\n"
        f"Average target length: {avg_target:.1f} tokens"
    )
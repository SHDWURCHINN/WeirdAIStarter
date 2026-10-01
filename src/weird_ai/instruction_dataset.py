"""
Instruction dataset utilities for Weird AI.
"""

from torch.utils.data import Dataset

from weird_ai.instruction_data import format_full_example


class InstructionDataset(Dataset):
    """
    Dataset for supervised instruction fine-tuning.

    Each item is a list of token IDs representing a full formatted instruction-response example.
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []

        for entry in data:
            full_text = format_full_example(entry)
            encoded = tokenizer.encode(full_text)
            self.encoded_texts.append(encoded)

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
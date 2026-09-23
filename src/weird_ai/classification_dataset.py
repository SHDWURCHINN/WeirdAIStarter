"""
Dataset utilities for Weird AI classification tasks.

This module supports Lesson 6: classification fine-tuning.

The goal is to convert labeled text examples into padded token ID tensors
that can be used by a PyTorch DataLoader.
"""

import torch
from torch.utils.data import Dataset


class LyricsClassificationDataset(Dataset):
    """
    Dataset for lyric classification.

    Each item returned by this dataset contains:

        input_ids, label

    where:

        input_ids: Tensor of token IDs with fixed length
        label: Tensor containing the class label

    Args:
        dataframe: A pandas DataFrame containing text and label columns.
        tokenizer: Tokenizer used to encode text.
        text_column: Name of the column containing input text.
        label_column: Name of the column containing labels.
        max_length: Fixed sequence length. If None, uses longest example.
        pad_token_id: Token ID used for padding.
    """

    def __init__(
        self,
        dataframe,
        tokenizer,
        text_column="text",
        label_column="label",
        max_length=None,
        pad_token_id=0
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.pad_token_id = pad_token_id

        self.encoded_texts = [
            tokenizer.encode(text)
            for text in self.dataframe[self.text_column]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        # Truncate every encoded text to self.max_length
        self.encoded_texts = [
            encoded_text[:self.max_length]
            for encoded_text in self.encoded_texts
        ]

        # Pad every encoded text so it has exactly self.max_length tokens
        self.encoded_texts = [
            encoded_text + [self.pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """
        Return one input/label pair.

        Args:
            index: Dataset index.

        Returns:
            input_ids: Tensor of shape (max_length)
            label: Tensor containing the class label
        """

        encoded = self.encoded_texts[index]
        label = self.dataframe.iloc[index][self.label_column]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """
        Return the number of examples in the dataset.
        """

        return len(self.dataframe)

    def _longest_encoded_length(self):
        """
        Find the length of the longest encoded text example.

        Returns:
            Length of the longest encoded example.
        """

        max_length = 0

        for encoded_text in self.encoded_texts:
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)

        return max_length
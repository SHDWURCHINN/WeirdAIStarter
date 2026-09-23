"""
Classification utilities for Weird AI.

This module supports the Lesson 6 classification fine-tuning side quest.
The goal is to show how transformer-style models can be adapted from
next-token generation to text classification.
"""

import torch
import torch.nn as nn


class TinyLyricsClassifier(nn.Module):
    """
    A small lyric classifier used for learning the classification workflow.

    This is not the full Weird AI transformer model. It is intentionally simple:

        token IDs
            ↓
        embedding layer
            ↓
        mean pooling
            ↓
        linear classification head
            ↓
        class logits

    Args:
        vocab_size: Number of tokens in the tokenizer vocabulary.
        emb_dim: Size of each token embedding.
        num_classes: Number of output classes.
    """

    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, input_ids):
        """
        Run a forward pass through the classifier.

        Args:
            input_ids: Tensor of shape (batch_size, num_tokens)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """

        embeddings = self.embedding(input_ids)

        # Average the embeddings across the token dimension (dim=1).
        # embeddings has shape (batch_size, num_tokens, emb_dim)
        # pooled will have shape (batch_size, emb_dim)
        pooled = embeddings.mean(dim=1)

        logits = self.classifier(pooled)

        return logits


def calculate_accuracy(data_loader, model, device):
    """
    Calculate classification accuracy for a model.

    Args:
        data_loader: DataLoader returning input IDs and labels.
        model: Classification model.
        device: CPU or CUDA device.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for input_batch, label_batch in data_loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            logits = model(input_batch)

            # 1. Convert logits to predicted class IDs using argmax along dimension 1.
            predicted_labels = torch.argmax(logits, dim=1)

            # 2. Count how many predictions match label_batch.
            correct += (predicted_labels == label_batch).sum().item()

            # 3. Update total count.
            total += label_batch.size(0)

    return correct / total if total > 0 else 0.0


def classify_text(text, model, tokenizer, max_length, device, pad_token_id=0):
    """
    Classify a single text string.

    Args:
        text: Input text to classify.
        model: Classification model.
        tokenizer: Tokenizer used to encode text.
        max_length: Required input sequence length.
        device: CPU or CUDA device.
        pad_token_id: Token ID used for padding.

    Returns:
        Predicted class ID as an integer.
    """

    model.eval()

    encoded = tokenizer.encode(text)

    # 1. Truncate encoded text to max_length.
    encoded = encoded[:max_length]

    # 2. Pad encoded text to max_length.
    encoded = encoded + [pad_token_id] * (max_length - len(encoded))

    # 3. Convert encoded text to a tensor.
    tensor_input = torch.tensor(encoded, dtype=torch.long)

    # 4. Add a batch dimension -> shape (1, max_length).
    tensor_input = tensor_input.unsqueeze(0)

    # 5. Move tensor to device.
    tensor_input = tensor_input.to(device)

    # 6. Run model forward pass and 7. Use argmax to get predicted label.
    with torch.no_grad():
        logits = model(tensor_input)
        predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label


def label_to_name(label):
    """
    Convert a numeric class label into a readable class name.

    For the starter classification notebook:

        0 = serious lyric
        1 = silly/comedic lyric

    Args:
        label: Integer class label.

    Returns:
        Human-readable label name.
    """

    label_names = {
        0: "serious",
        1: "silly/comedic",
    }

    return label_names.get(label, "unknown")
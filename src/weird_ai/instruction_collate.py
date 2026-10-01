"""
Custom collate function for Weird AI instruction fine-tuning.
"""

import torch


def custom_collate_fn(
    batch,
    pad_token_id=0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Pad variable-length token ID lists, create shifted targets, and mask extra padding.
    """

    batch_max_length = max(len(item) for item in batch) + 1

    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()

        new_item.append(pad_token_id)

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = (targets == pad_token_id)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if len(indices) > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None and len(inputs) > allowed_max_length:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
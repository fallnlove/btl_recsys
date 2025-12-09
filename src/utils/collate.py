from typing import Union
import torch

def collate_fn(dataset_items: list[dict]) -> dict[Union[torch.Tensor, list]]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Union[Tensor, list]]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {
        "history": torch.nn.utils.rnn.pad_sequence(
            [
                item["history"]
                for item in dataset_items
            ],
            batch_first=True,
            padding_value=-1,
        ).long(),
        "user_id": torch.tensor(
            [int(item["user_id"]) for item in dataset_items],
            dtype=torch.long,
        ),
    }

    if "holdout" in dataset_items[0].keys():
        result_batch["holdout"] = torch.tensor(
            [int(item["holdout"]) for item in dataset_items],
            dtype=torch.long,
        )

    return result_batch

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

sns.set_theme()

from .metrics import StatefullMetric


def create_logger(
    name,
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
):
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger = logging.getLogger(name)
    return logger


logger = create_logger(name=__name__)


def move_batch(batch, device):
    for key, value in batch.items():
        batch[key] = value.to(device)


class BasicBatchProcessor:
    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith(".ids"):
                prefix = key.split(".")[0]
                assert "{}.length".format(prefix) in batch[0]

                processed_batch[f"{prefix}.ids"] = []
                processed_batch[f"{prefix}.length"] = []

                for sample in batch:
                    processed_batch[f"{prefix}.ids"].extend(sample[f"{prefix}.ids"])
                    processed_batch[f"{prefix}.length"].append(
                        sample[f"{prefix}.length"]
                    )

        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    return params


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_masked_tensor(data, lengths, device):
    batch_size = lengths.shape[0]
    max_sequence_length = lengths.max().item()

    padded_embeddings = torch.zeros(
        batch_size,
        max_sequence_length,
        data.shape[-1],
        dtype=torch.float,
        device=device,
    )  # (batch_size, max_seq_len, emb_dim)

    mask = (
        torch.arange(end=max_sequence_length, device=device)[None].tile([batch_size, 1])
        < lengths[:, None]
    )  # (batch_size, max_seq_len)

    padded_embeddings[mask] = data

    return padded_embeddings, mask


def get_activation_function(name, **kwargs):
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "elu":
        return torch.nn.ELU(alpha=float(kwargs.get("alpha", 1.0)))
    elif name == "leaky":
        return torch.nn.LeakyReLU(
            negative_slope=float(kwargs.get("negative_slope", 1e-2))
        )
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "softmax":
        return torch.nn.Softmax()
    elif name == "softplus":
        return torch.nn.Softplus(
            beta=int(kwargs.get("beta", 1.0)),
            threshold=int(kwargs.get("threshold", 20)),
        )
    elif name == "softmax_logit":
        return torch.nn.LogSoftmax()
    else:
        raise ValueError("Unknown activation function name `{}`".format(name))


def inference(dataloader, model, metrics, device):
    running_metrics = {}
    for metric_name, metric_function in metrics.items():
        running_metrics[metric_name] = []

    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            for key, value in batch.items():
                batch[key] = value.to(device)
            batch["logits"] = model(batch)

            for key, values in batch.items():
                batch[key] = values.cpu()

            for metric_name, metric_function in metrics.items():
                running_metrics[metric_name].extend(metric_function(inputs=batch))

        for metric_name, metric_function in metrics.items():
            if isinstance(metric_function, StatefullMetric):
                running_metrics[metric_name] = metric_function.reduce(
                    running_metrics[metric_name]
                )

    print("Inference procedure has been finished!")
    print("Metrics are the following:")
    results = {}
    for metric_name, metric_value in running_metrics.items():
        results[metric_name] = np.mean(metric_value)
        print("{}: {}".format(metric_name, np.mean(metric_value)))
    print("Metrics finished!")
    model.train()
    return results


def train(
    dataloader,
    model,
    optimizer,
    loss_function,
    num_epochs,
    early_stopping_rounds,
    device,
    best_metric=None,
    inference_dict_validation=None,
    inference_dict_test=None,
):
    logger.debug("Start training...")
    train_start = time.time()
    best_val_metric = 0.0
    best_epoch = 0
    best_metrics = {}
    all_metrics_list = []
    for epoch_num in range(num_epochs):
        logger.debug(f"Start epoch {epoch_num}")
        for step, batch in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_num}"
        ):
            model.train()
            move_batch(batch, device)
            batch.update(model(batch))
            loss = loss_function(batch)
            optimizer.step(loss)
        print("VAL")
        val_metrics = inference(**inference_dict_validation)
        print("TEST")
        test_metrics = inference(**inference_dict_test)

        all_metrics = {
            f"val/{metric_name}": metric_value
            for metric_name, metric_value in val_metrics.items()
        }
        all_metrics |= {
            f"test/{metric_name}": metric_value
            for metric_name, metric_value in test_metrics.items()
        }

        all_metrics_list.append(all_metrics)

        val_ndcg = val_metrics["ndcg@10"]
        if val_ndcg > best_val_metric:
            best_metrics = all_metrics
            best_val_metric = val_ndcg
            best_epoch = epoch_num
        elif epoch_num - best_epoch > early_stopping_rounds:
            print(f"no more improve in {early_stopping_rounds} epoch")
            break

    train_end = time.time()
    print("Total time:", train_end - train_start)
    return all_metrics_list, best_metrics


def save_metrics(all_metrics: list[dict], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_per_split = defaultdict(lambda: defaultdict(list))

    for epoch_metrics in all_metrics:
        for key, value in epoch_metrics.items():
            split, metric_name = key.split("/", 1)
            metrics_per_split[split][metric_name].append(value)

    metric_names = sorted(
        {name for split in metrics_per_split.values() for name in split}
    )

    def plot_metric(metric_name):
        plt.figure(figsize=(6, 4))
        for split, metrics in metrics_per_split.items():
            if metric_name in metrics:
                plt.plot(
                    range(1, len(metrics[metric_name]) + 1),
                    metrics[metric_name],
                    marker="o",
                    label=split,
                )
        plt.title(metric_name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = output_dir / f"{metric_name.replace('@', '_at_')}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()

    for name in metric_names:
        plot_metric(name)

    print(f"✅ Saved {len(metric_names) + 1} plots to {output_dir.resolve()}")


def compute_item_distance_matrix(df, num_items, num_users):
    """
    df must contain columns: item_id, user_id, rating (0/1 or rating)
    Returns: (num_items × num_items) cosine distance matrix
    """

    mat = torch.zeros((num_items, num_users), dtype=torch.float32)
    mat[df["item_id"], df["user_id"]] = torch.tensor(
        df["rating"].values, dtype=torch.float32
    )

    mat = torch.nn.functional.normalize(mat, p=2, dim=1)  # (I × U)

    sim = mat @ mat.T

    dist = 1 - sim

    return dist

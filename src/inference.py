import copy
import json

import numpy as np
import torch

from .loss import MRGSRecLoss
from .metrics import BaseMetric, StatefullMetric
from .model import MRGSRecModel
from .optimizer import BasicOptimizer
from .utils import create_logger, fix_random_seed, parse_args


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
    for metric_name, metric_value in running_metrics.items():
        print("{}: {}".format(metric_name, np.mean(metric_value)))
        return np.mean(metric_value)
    print("Metrics finished!")
    model.train()

from utils import parse_args, create_logger, DEVICE, fix_random_seed

from dataset import GraphDataset
from dataloader import TorchDataloader
from loss import MRGSRecLoss
from model import MRGSRecModel
from optimizer import BasicOptimizer

from metrics import BaseMetric, StatefullMetric

import copy
import json
import torch
import numpy as np

def inference(dataloader, model, metrics, pred_prefix, labels_prefix):
    running_metrics = {}
    for metric_name, metric_function in metrics.items():
        running_metrics[metric_name] = []
    
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            for key, value in batch.items():
                batch[key] = value.to(DEVICE)
            batch[pred_prefix] = model(batch)

            for key, values in batch.items():
                batch[key] = values.cpu()

            for metric_name, metric_function in metrics.items():
                running_metrics[metric_name].extend(metric_function(
                    inputs=batch,
                    pred_prefix=pred_prefix,
                    labels_prefix=labels_prefix,
                ))
            
        for metric_name, metric_function in metrics.items():
            if isinstance(metric_function, StatefullMetric):
                running_metrics[metric_name] = metric_function.reduce(running_metrics[metric_name])

    print('Inference procedure has been finished!')
    print('Metrics are the following:')
    for metric_name, metric_value in running_metrics.items():
        print('{}: {}'.format(metric_name, np.mean(metric_value)))
        return np.mean(metric_value)
    print('Metrics finished!')
    model.train()
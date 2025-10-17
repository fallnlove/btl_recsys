from time import time
import argparse
import inspect
import json
import logging
import random

import numpy as np
import torch


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


class MetaParent(type):

    def __init__(cls, name, base, params, **kwargs):
        super().__init__(name, base, params)
        is_base_class = cls.mro()[1] is object
        if is_base_class:
            base_class = cls
        else:
            base_class_found = False
            for key in cls.mro():
                if isinstance(key, MetaParent) and key.mro()[1] is object:
                    assert base_class_found is False, "multiple base classes(bug)"
                    base_class = key
                    base_class_found = True
            assert base_class_found is True, f"no base class for {name}"

        if is_base_class:
            cls._subclasses = {}

        @classmethod
        def __init_subclass__(scls, config_name=None):
            super().__init_subclass__()
            if config_name is not None:
                if config_name in base_class._subclasses:
                    raise ValueError(
                        "Class with name `{}` is already registered".format(config_name)
                    )
                scls.config_name = config_name
                base_class._subclasses[config_name] = scls

        cls.__init_subclass__ = __init_subclass__

        @classmethod
        def parent_create_from_config(cls, config, **kwargs):
            if "type" in config:
                return cls._subclasses[config["type"]].create_from_config(
                    config, **kwargs
                )
            else:
                raise ValueError(
                    "There is no `type` provided for the `{}` class".format(name)
                )

        # Take kwargs for the last initialized baseclass
        init_kwargs = {}
        for bcls in cls.mro()[:-1]:  # Look into all base classes except object
            if "__init__" not in bcls.__dict__:
                continue
            init_kwargs = inspect.signature(bcls.__init__).parameters
            break

        @classmethod
        def child_create_from_config(cls, config, **kwargs):
            kwargs = {}
            for key, argspec in init_kwargs.items():
                if key == "self":
                    continue
                value = config.get(key, argspec.default)
                if value is inspect.Parameter.empty:
                    msg = "There is no value for `{}.__init__` required field `{}` in config `{}`"
                    raise ValueError(msg.format(cls, key, config))
                kwargs[key] = value
            return cls(**kwargs)

        if "create_from_config" not in cls.__dict__:
            cls.create_from_config = (
                parent_create_from_config if is_base_class else child_create_from_config
            )


def train(
    dataloader,
    warm_dataloader,
    model,
    optimizer,
    optimizer_fi,
    loss_function,
    num_epochs,
    early_stopping_rounds,
    device,
    best_metric=None,
    inference_dict=None,
):
    logger.debug("Start training...")
    train_start = time.time()
    best_metric = 0.0
    best_epoch = 0
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
        current_metric = inference(**inference_dict)
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch_num
        elif epoch_num - best_epoch > early_stopping_rounds:
            print(f"no more improve in {early_stopping_rounds} epoch")
            break

    train_end = time.time()
    print("Total time:", train_end - train_start)
    return best_metric

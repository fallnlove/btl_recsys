import copy
import json
import time

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from src.dataloader import TorchDataloader
from src.dataset import ScientificDataset, build_graph
from src.inference import inference
from src.loss import MRGSRecLoss
from src.metrics import BaseMetric, StatefullMetric
from src.model import MRGSRecModel
from src.optimizer import BasicOptimizer
from src.utils import DEVICE, create_logger, fix_random_seed, parse_args

logger = create_logger(name=__name__)
seed_val = 42


def train(
    dataloader,
    warm_dataloader,
    model,
    optimizer,
    optimizer_fi,
    loss_function,
    num_epochs,
    early_stopping_rounds,
    best_metric=None,
    inference_dict=None,
):
    logger.debug("Start training...")
    train_start = time.time()
    best_metric = 0.0
    best_epoch = 0
    for epoch_num in trange(num_epochs):
        logger.debug(f"Start epoch {epoch_num}")
        for step, batch in tqdm(enumerate(dataloader)):
            batch_ = copy.deepcopy(batch)

            model.train()

            for key, values in batch_.items():
                batch_[key] = batch_[key].to(DEVICE)

            batch_.update(model(batch_))
            loss = loss_function(batch_)
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

    # print("start folding_in")
    # print(torch.linalg.norm(model._user_embeddings.weight))
    # print(torch.linalg.norm(model._newuser_embeddings.weight))

    # torch.save(model, "ml_model.pt")
    # #model = torch.load("ml_model.pt")
    # num_user = model._num_users
    # model._user_embeddings.requires_grad_(False)
    # model._item_embeddings.requires_grad_(False)
    # mas = []
    # inds = []
    # with open('./data2/MovieLens1M/val_users.txt', 'r') as f:
    #     users = [int(line.strip()) for line in f]
    # for i in tqdm(users[:100]):
    #     it = 0
    #     model._newuser_embeddings.weight = torch.nn.Parameter(model._user_embeddings.weight[i].unsqueeze(0))
    #     for epoch in range(8):
    #         for step, batch in enumerate(dataloader):
    #             it += 1
    #             batch_ = copy.deepcopy(batch)

    #             model.train()

    #             for key, values in batch_.items():
    #                 batch_[key] = batch_[key].to(DEVICE)

    #             batch_.update(model(batch_, ind = i))
    #             loss = loss_function(batch_) + 100 * torch.linalg.norm(model._user_embeddings.weight[i] - model._newuser_embeddings.weight) ** 2

    #             optimizer_fi.step(loss)
    #             step_num += 1

    #             if best_metric is None:
    #                 # Take the last model
    #                 best_checkpoint = copy.deepcopy(model.state_dict())
    #                 best_epoch = epoch_num
    #             elif best_checkpoint is None or best_metric in batch_ and current_metric <= batch_[best_metric]:
    #                 # If it is the first checkpoint, or it is the best checkpoint
    #                 current_metric = batch_[best_metric]
    #                 best_checkpoint = copy.deepcopy(model.state_dict())
    #                 best_epoch = epoch_num

    #             if it >= max_batch:
    #                 break
    #         mas += [model._newuser_embeddings.weight.clone()]
    #         inds += [i]

    # for i in range(len(mas)):
    #     model._user_embeddings.weight[inds[i]] += (model._newuser_embeddings.weight[0] - model._user_embeddings.weight[inds[i]])

    # print(torch.linalg.norm(model._user_embeddings.weight))
    # print(torch.linalg.norm(model._newuser_embeddings.weight))
    # logger.debug('Training procedure has been finished!')
    # return best_checkpoint


@hydra.main(version_base=None, config_path="configs", config_name="ml1m")
def main(cfg):
    run_train(cfg)


def run_train(cfg):
    fix_random_seed(seed_val)
    config = OmegaConf.to_container(cfg, resolve=True)

    logger.info("Training config: \n{}".format(OmegaConf.to_yaml(config)))
    logger.info("Current DEVICE: {}".format(DEVICE))

    dataset = ScientificDataset.create_from_config(config["dataset"])
    graph = build_graph(dataset, config["dataset"]["dataset"]["path_to_data_dir"])

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    train_dataloader = TorchDataloader.create_from_config(
        config["dataloader"]["train"], dataset=train_sampler, **dataset.meta
    )

    warm_dataloader = TorchDataloader.create_from_config(
        config["dataloader"]["warm_val"], dataset=train_sampler, **dataset.meta
    )

    validation_dataloader = TorchDataloader.create_from_config(
        config["dataloader"]["validation"], dataset=validation_sampler, **dataset.meta
    )

    eval_dataloader = TorchDataloader.create_from_config(
        config["dataloader"]["validation"], dataset=test_sampler, **dataset.meta
    )

    model = MRGSRecModel.create_from_config(
        config["model"], graph=graph, **dataset.meta
    ).to(DEVICE)
    loss_function = MRGSRecLoss.create_from_config(config["loss"])
    optimizer = BasicOptimizer.create_from_config(config["optimizer"], model=model)
    optimizer_fi = BasicOptimizer.create_from_config(
        config["optimizer_fi"], model=model
    )

    logger.debug("Everything is ready for training process!")

    metrics = {
        metric_name: BaseMetric.create_from_config(metric_cfg, **dataset.meta)
        for metric_name, metric_cfg in config["metrics"].items()
    }

    inference_dict = dict(
        dataloader=validation_dataloader,
        model=model,
        metrics=metrics,
    )

    # Train process
    best_metric = train(
        dataloader=train_dataloader,
        warm_dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        optimizer_fi=optimizer_fi,
        loss_function=loss_function,
        num_epochs=config.get("num_epochs", 100),
        early_stopping_rounds=config.get("early_stopping_rounds", 10),
        best_metric=config.get("best_metric"),
        inference_dict=inference_dict,
    )
    print(f"ndcg@10 = {best_metric}")
    return best_metric
    _ = inference(**inference_dict)

    logger.debug("Saving model...")
    checkpoint_path = "./checkpoints/{}_final_state.pth".format(
        config["experiment_name"]
    )
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug("Saved model as {}".format(checkpoint_path))


if __name__ == "__main__":
    main()

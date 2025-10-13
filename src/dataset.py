import copy
import logging
import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .utils import DEVICE

logger = logging.getLogger(__name__)


class SequenceSampler:
    def __init__(self, dataset, num_users, num_items, mode):
        assert mode in ["train", "eval"]
        self._mode = mode
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @property
    def dataset(self):
        return self._dataset

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs["dataset"],
            num_users=kwargs["num_users"],
            num_items=kwargs["num_items"],
            mode=kwargs["mode"],
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        if self._mode == "train":
            sample = copy.deepcopy(self._dataset[index])

            item_sequence = sample["item.ids"][:-1]
            next_item_sequence = sample["item.ids"][1:]

            return {
                "user.ids": sample["user.ids"],
                "user.length": sample["user.length"],
                "item.ids": item_sequence,
                "item.length": len(item_sequence),
                "positive.ids": next_item_sequence,
                "positive.length": len(next_item_sequence),
            }
        else:
            sample = copy.deepcopy(self._dataset[index])

            item_sequence = sample["item.ids"][:-1]
            next_item = sample["item.ids"][-1]

            return {
                "user.ids": sample["user.ids"],
                "user.length": sample["user.length"],
                "item.ids": item_sequence,
                "item.length": len(item_sequence),
                "labels.ids": [next_item],
                "labels.length": 1,
            }


def build_graph(dataset, graph_dir_path):
    train_sampler, _, _ = dataset.get_samplers()

    (
        train_interactions,
        train_user_interactions,
        train_item_interactions,
    ) = (
        [],
        [],
        [],
    )

    train_user_2_items = defaultdict(set)
    train_item_2_users = defaultdict(set)
    visited_user_item_pairs = set()

    def _process_sampler(
        sampler,
    ):
        for sample in sampler.dataset:
            user_id = sample["user.ids"][0]
            item_ids = sample["item.ids"]

            for item_id in item_ids:
                if (user_id, item_id) not in visited_user_item_pairs:
                    train_interactions.append((user_id, item_id))
                    train_user_interactions.append(user_id)
                    train_item_interactions.append(item_id)

                    train_user_2_items[user_id].add(item_id)
                    train_item_2_users[item_id].add(user_id)

                    visited_user_item_pairs.add((user_id, item_id))

    _process_sampler(train_sampler)
    _train_interactions = np.array(train_interactions)
    _train_user_interactions = np.array(train_user_interactions)
    _train_item_interactions = np.array(train_item_interactions)

    _graph = _build_general_graph(
        graph_dir_path,
        train_user_interactions,
        train_item_interactions,
        dataset.num_users,
        dataset.num_items,
    )

    return _graph


def _build_general_graph(
    graph_dir_path,
    train_user_interactions,
    train_item_interactions,
    _num_users,
    _num_items,
):
    path_to_graph = os.path.join(graph_dir_path, "general_graph.npz")
    if os.path.exists(path_to_graph):
        logger.info("loading graph from file")
        return sp.load_npz(path_to_graph)
    else:
        logger.info("building new graph")
        # place ones only when co-occurrence happens
        user2item_connections = csr_matrix(
            (
                np.ones(len(train_user_interactions)),
                (train_user_interactions, train_item_interactions),
            ),
            shape=(_num_users + 2, _num_items + 2),
        )  # (num_users + 2, num_items + 2), bipartite graph
        _graph = get_sparse_graph_layer(
            user2item_connections,
            _num_users + 2,
            _num_items + 2,
            biparite=True,
        )
        # sp.save_npz(path_to_graph, self._graph)
        return _convert_sp_mat_to_sp_tensor(_graph).coalesce().to(DEVICE)


def get_sparse_graph_layer(sparse_matrix, fst_dim, snd_dim, biparite=False):
    mat_dim_size = fst_dim + snd_dim if biparite else fst_dim

    adj_mat = sp.dok_matrix((mat_dim_size, mat_dim_size), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    R = sparse_matrix.tolil()  # list of lists (fst_dim, snd_dim)

    if biparite:
        adj_mat[:fst_dim, fst_dim:] = R  # (num_users, num_items)
        adj_mat[fst_dim:, :fst_dim] = R.T  # (num_items, num_users)
    else:
        adj_mat = R

    adj_mat = adj_mat.todok()
    # adj_mat += sp.eye(adj_mat.shape[0])  # remove division by zero issue

    edges_degree = np.array(adj_mat.sum(axis=1))  # D

    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum + 0.00000000001, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)

    d_inv = np.power(edges_degree + 0.00000000001, -0.5).flatten()  # D^(-0.5)
    d_inv[np.isinf(d_inv)] = 0.0  # fix NaNs in case if row with zero connections
    d_mat = sp.diags(d_inv)  # make it square matrix

    # D^(-0.5) @ A @ D^(-0.5)
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)

    return norm_adj.tocsr()


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class ScientificDataset:

    def __init__(
        self,
        max_sequence_length,
        path_to_data_dir,
        dataset_name,
    ):
        self._max_sequence_length = max_sequence_length
        self._path_to_data_dir = path_to_data_dir
        self._dataset_name = dataset_name

        data = self._read_data()
        (
            train_dataset,
            validation_dataset,
            test_dataset,
            self._num_users,
            self._num_items,
        ) = self._create_datasets(data)

        self._log_dataset_stats(
            train_dataset,
            test_dataset,
            self._num_users,
            self._num_items,
        )

        self._train_sampler = SequenceSampler(
            dataset=train_dataset,
            num_users=self._num_users,
            num_items=self._num_items,
            mode="train",
        )

        self._validation_sampler, self._test_sampler = (
            SequenceSampler(
                dataset=sampler_dataset,
                num_users=self._num_users,
                num_items=self._num_items,
                mode="eval",
            )
            for sampler_dataset in [validation_dataset, test_dataset]
        )

    def _log_dataset_stats(
        self,
        train_dataset,
        test_dataset,
        max_user_idx,
        max_item_idx,
    ):
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        logger.info(f"Max user idx: {max_user_idx}")
        logger.info(f"Max item idx: {max_item_idx}")
        logger.info(f"Max sequence length: {self.max_sequence_length}")
        sparsity = (
            (len(train_dataset) + len(test_dataset)) / max_user_idx / max_item_idx
        )
        logger.info(f"{self._dataset_name} dataset sparsity: {sparsity}")

    def _create_datasets(self, data):
        max_user_idx, max_item_idx = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        for sample in data:
            sample = sample.strip("\n").split(" ")
            user_idx = int(sample[0])
            item_ids = [int(item_id) for item_id in sample[1:]]

            max_user_idx = max(max_user_idx, user_idx)
            max_item_idx = max(max_item_idx, max(item_ids))

            assert len(item_ids) >= 5

            train_dataset.append(
                {
                    "user.ids": [user_idx],
                    "user.length": 1,
                    "item.ids": item_ids[:-2][-self._max_sequence_length :],
                    "item.length": len(item_ids[:-2][-self._max_sequence_length :]),
                }
            )
            assert len(item_ids[:-2][-self._max_sequence_length :]) == len(
                set(item_ids[:-2][-self._max_sequence_length :])
            )
            validation_dataset.append(
                {
                    "user.ids": [user_idx],
                    "user.length": 1,
                    "item.ids": item_ids[:-1][-self._max_sequence_length :],
                    "item.length": len(item_ids[:-1][-self._max_sequence_length :]),
                }
            )
            assert len(item_ids[:-1][-self._max_sequence_length :]) == len(
                set(item_ids[:-1][-self._max_sequence_length :])
            )
            test_dataset.append(
                {
                    "user.ids": [user_idx],
                    "user.length": 1,
                    "item.ids": item_ids[-self._max_sequence_length :],
                    "item.length": len(item_ids[-self._max_sequence_length :]),
                }
            )
            assert len(item_ids[-self._max_sequence_length :]) == len(
                set(item_ids[-self._max_sequence_length :])
            )

        print(f"{len(train_dataset)=}")
        print(f"{len(validation_dataset)=}")
        print(f"{len(test_dataset)=}")


        return (
            train_dataset,
            validation_dataset,
            test_dataset,
            max_user_idx,
            max_item_idx,
        )

    def _read_data(self):
        data_dir_path = os.path.join(self._path_to_data_dir, self._dataset_name)
        dataset_path = os.path.join(data_dir_path, "all_data.txt")
        with open(dataset_path, "r") as f:
            data = f.readlines()

        return data

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def meta(self):
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "max_sequence_length": self.max_sequence_length,
        }

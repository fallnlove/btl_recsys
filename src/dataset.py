import copy
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SequenceDataset:
    def __init__(self, data_path, max_sequence_length, mode, all_data=None):
        assert mode in ["train", "val", "test"], "Wrong mode"
        if mode == "val" and all_data is None:
            assert False, "Need all data for val"
        self._mode = mode
        self._index = []
        self._fill_index(data_path, max_sequence_length, mode, all_data)

    def _fill_index(self, data_path, max_sequence_length, mode, all_data):
        df = pd.read_csv(data_path)
        if mode != "val":
            user_items = df.groupby("user_id")["item_id"].apply(list).to_dict()
            max_sequence_length = 10
            for user_idx, item_ids in sorted(
                list(user_items.items()), key=lambda x: x[0]
            ):
                item_sequnce = item_ids[-max_sequence_length:]
                self._index.append(
                    {
                        "user.ids": [user_idx],
                        "user.length": 1,
                        "item.ids": item_sequnce,
                        "item.length": len(item_sequnce),
                    }
                )
        else:
            for _, (val_user_id, val_item_id, val_rating, val_timestamp) in tqdm(
                df.iterrows(), total=len(df), desc="Validation ds creation"
            ):
                user_info = all_data[all_data.user_id == val_user_id]
                previous_user_info = user_info[
                    user_info.timestamp <= val_timestamp
                ].copy()
                # to make out item last
                previous_user_info.loc[
                    previous_user_info.user_id == val_user_id, "timestamp"
                ] += 1
                previous_user_info = previous_user_info.reset_index().sort_values(
                    by=["timestamp", "index"]
                )
                previous_user_items = previous_user_info.item_id.tolist()
                assert previous_user_items[-1] == val_item_id
                # print(previous_user_items)
                item_sequence = previous_user_items[-max_sequence_length:]
                self._index.append(
                    {
                        "user.ids": [val_user_id],
                        "user.length": 1,
                        "item.ids": item_sequence,
                        "item.length": len(item_sequence),
                    }
                )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        sample = self._index[index]
        item_sequence = sample["item.ids"][:-1]
        if self._mode == "train":
            next_item_sequence = sample["item.ids"][1:]
        else:
            next_item_sequence = [sample["item.ids"][-1]]
        return {
            "user.ids": sample["user.ids"],
            "user.length": sample["user.length"],
            "item.ids": item_sequence,
            "item.length": len(item_sequence),
            "labels.ids": next_item_sequence,
            "labels.length": len(next_item_sequence),
        }


def build_graph(train_dataset, graph_dir_path, device, dataset_meta):
    train_sampler = train_dataset

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
        for sample in sampler._index:
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
        dataset_meta["num_users"],
        dataset_meta["num_items"],
        device,
    )

    return _graph


def _build_general_graph(
    graph_dir_path,
    train_user_interactions,
    train_item_interactions,
    _num_users,
    _num_items,
    device,
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
        return _convert_sp_mat_to_sp_tensor(_graph).coalesce().to(device)


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

from collections import defaultdict

from tqdm import tqdm

from utils import DEVICE

import copy

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import os
import logging

logger = logging.getLogger(__name__)


class TrainSampler:

    def __init__(self, dataset, num_users, num_items):
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item_sequence = sample['item.ids'][1:]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'positive.ids': next_item_sequence,
            'positive.length': len(next_item_sequence),
        }


class EvalSampler:

    def __init__(self, dataset, num_users, num_items):
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @property
    def dataset(self):
        return self._dataset

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item = sample['item.ids'][-1]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'labels.ids': [next_item],
            'labels.length': 1
        }


class GraphDataset:

    def __init__(
            self,
            dataset,
            graph_dir_path,
            use_train_data_only=True,
            use_user_graph=False,
            use_item_graph=False
    ):
        self._dataset = dataset
        self._graph_dir_path = graph_dir_path
        self._use_train_data_only = use_train_data_only
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

        train_interactions, train_user_interactions, train_item_interactions = [], [], []

        train_user_2_items = defaultdict(set)
        train_item_2_users = defaultdict(set)
        visited_user_item_pairs = set()

        for sample in train_sampler.dataset:
            user_id = sample['user.ids'][0]
            item_ids = sample['item.ids']

            for item_id in item_ids:
                if (user_id, item_id) not in visited_user_item_pairs:
                    train_interactions.append((user_id, item_id))
                    train_user_interactions.append(user_id)
                    train_item_interactions.append(item_id)

                    train_user_2_items[user_id].add(item_id)
                    train_item_2_users[item_id].add(user_id)

                    visited_user_item_pairs.add((user_id, item_id))

        if not self._use_train_data_only:
            for sample in validation_sampler.dataset:
                user_id = sample['user.ids'][0]
                item_ids = sample['item.ids']

                for item_id in item_ids:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))

            for sample in test_sampler.dataset:
                user_id = sample['user.ids'][0]
                item_ids = sample['item.ids']

                for item_id in item_ids:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

        path_to_graph = os.path.join(graph_dir_path, 'general_graph.npz')
        if os.path.exists(path_to_graph):
            print(1)
            self._graph = sp.load_npz(path_to_graph)
        else:
            print(2)
            # place ones only when co-occurrence happens
            user2item_connections = csr_matrix(
                (np.ones(len(train_user_interactions)), (train_user_interactions, train_item_interactions)),
                shape=(self._num_users + 2, self._num_items + 2)
            )  # (num_users + 2, num_items + 2), bipartite graph
            self._graph = self.get_sparse_graph_layer(
                user2item_connections,
                self._num_users + 2,
                self._num_items + 2,
                biparite=True
            )
            #sp.save_npz(path_to_graph, self._graph)

        self._graph = self._convert_sp_mat_to_sp_tensor(self._graph).coalesce().to(DEVICE)

        if self._use_user_graph:
            path_to_user_graph = os.path.join(graph_dir_path, 'user_graph.npz')
            if os.path.exists(path_to_user_graph):
                self._user_graph = sp.load_npz(path_to_user_graph)
            else:
                user2user_interactions_fst = []
                user2user_interactions_snd = []
                visited_user_item_pairs = set()
                visited_user_user_pairs = set()

                for user_id, item_id in tqdm(zip(self._train_user_interactions, self._train_item_interactions)):
                    if (user_id, item_id) in visited_user_item_pairs:
                        continue  # process (user, item) pair only once
                    visited_user_item_pairs.add((user_id, item_id))

                    for connected_user_id in train_item_2_users[item_id]:
                        if (user_id, connected_user_id) in visited_user_user_pairs or user_id == connected_user_id:
                            continue  # add (user, user) to graph connections pair only once
                        visited_user_user_pairs.add((user_id, connected_user_id))

                        user2user_interactions_fst.append(user_id)
                        user2user_interactions_snd.append(connected_user_id)

                # (user, user) graph
                user2user_connections = csr_matrix(
                    (
                        np.ones(len(user2user_interactions_fst)),
                        (user2user_interactions_fst, user2user_interactions_snd)),
                    shape=(self._num_users + 2, self._num_users + 2)
                )

                self._user_graph = self.get_sparse_graph_layer(
                    user2user_connections,
                    self._num_users + 2,
                    self._num_users + 2,
                    biparite=False
                )
                # sp.save_npz(path_to_user_graph, self._user_graph)

            self._user_graph = self._convert_sp_mat_to_sp_tensor(self._user_graph).coalesce().to(DEVICE)
        else:
            self._user_graph = None

        if self._use_item_graph:
            path_to_item_graph = os.path.join(graph_dir_path, 'item_graph.npz')
            if os.path.exists(path_to_item_graph):
                self._item_graph = sp.load_npz(path_to_item_graph)
            else:
                item2item_interactions_fst = []
                item2item_interactions_snd = []
                visited_user_item_pairs = set()
                visited_item_item_pairs = set()

                for user_id, item_id in tqdm(zip(self._train_user_interactions, self._train_item_interactions)):
                    if (user_id, item_id) in visited_user_item_pairs:
                        continue  # process (user, item) pair only once
                    visited_user_item_pairs.add((user_id, item_id))

                    for connected_item_id in train_user_2_items[user_id]:
                        if (item_id, connected_item_id) in visited_item_item_pairs or item_id == connected_item_id:
                            continue  # add (item, item) to graph connections pair only once
                        visited_item_item_pairs.add((item_id, connected_item_id))

                        item2item_interactions_fst.append(item_id)
                        item2item_interactions_snd.append(connected_item_id)

                # (item, item) graph
                item2item_connections = csr_matrix(
                    (
                        np.ones(len(item2item_interactions_fst)),
                        (item2item_interactions_fst, item2item_interactions_snd)),
                    shape=(self._num_items + 2, self._num_items + 2)
                )
                self._item_graph = self.get_sparse_graph_layer(
                    item2item_connections,
                    self._num_items + 2,
                    self._num_items + 2,
                    biparite=False
                )
                # sp.save_npz(path_to_item_graph, self._item_graph)

            self._item_graph = self._convert_sp_mat_to_sp_tensor(self._item_graph).coalesce().to(DEVICE)
        else:
            self._item_graph = None

    @classmethod
    def create_from_config(cls, config):
        dataset = ScientificDataset.create_from_config(config['dataset'])
        return cls(
            dataset=dataset,
            graph_dir_path=config['graph_dir_path'],
            use_user_graph=config.get('use_user_graph', False),
            use_item_graph=config.get('use_item_graph', False)
        )

    @staticmethod
    def get_sparse_graph_layer(sparse_matrix, fst_dim, snd_dim, biparite=False):
        mat_dim_size = fst_dim + snd_dim if biparite else fst_dim

        adj_mat = sp.dok_matrix(
            (mat_dim_size, mat_dim_size),
            dtype=np.float32
        )
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
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        d_inv = np.power(edges_degree + 0.00000000001, -0.5).flatten()  # D^(-0.5)
        d_inv[np.isinf(d_inv)] = 0.  # fix NaNs in case if row with zero connections
        d_mat = sp.diags(d_inv)  # make it square matrix

        # D^(-0.5) @ A @ D^(-0.5)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)

        return norm_adj.tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    @property
    def num_users(self):
        return self._dataset.num_users

    @property
    def num_items(self):
        return self._dataset.num_items

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        meta = {
            'user_graph': self._user_graph,
            'item_graph': self._item_graph,
            'graph': self._graph,
            **self._dataset.meta
        }
        return meta


class ScientificDataset:

    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(config['path_to_data_dir'], config['name'])
        max_sequence_length = config['max_sequence_length']
        max_user_idx, max_item_idx = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        dataset_path = os.path.join(data_dir_path, '{}.txt'.format('all_data'))
        with open(dataset_path, 'r') as f:
            data = f.readlines()

        for sample in data:
            sample = sample.strip('\n').split(' ')
            user_idx = int(sample[0])
            item_ids = [int(item_id) for item_id in sample[1:]]

            max_user_idx = max(max_user_idx, user_idx)
            max_item_idx = max(max_item_idx, max(item_ids))

            assert len(item_ids) >= 5

            train_dataset.append({
                'user.ids': [user_idx],
                'user.length': 1,
                'item.ids': item_ids[:-2][-max_sequence_length:],
                'item.length': len(item_ids[:-2][-max_sequence_length:])
            })
            assert len(item_ids[:-2][-max_sequence_length:]) == len(set(item_ids[:-2][-max_sequence_length:]))
            validation_dataset.append({
                'user.ids': [user_idx],
                'user.length': 1,
                'item.ids': item_ids[:-1][-max_sequence_length:],
                'item.length': len(item_ids[:-1][-max_sequence_length:])
            })
            assert len(item_ids[:-1][-max_sequence_length:]) == len(set(item_ids[:-1][-max_sequence_length:]))
            test_dataset.append({
                'user.ids': [user_idx],
                'user.length': 1,
                'item.ids': item_ids[-max_sequence_length:],
                'item.length': len(item_ids[-max_sequence_length:])
            })
            assert len(item_ids[-max_sequence_length:]) == len(set(item_ids[-max_sequence_length:]))

        logger.info('Train dataset size: {}'.format(len(train_dataset)))
        logger.info('Test dataset size: {}'.format(len(test_dataset)))
        logger.info('Max user idx: {}'.format(max_user_idx))
        logger.info('Max item idx: {}'.format(max_item_idx))
        logger.info('Max sequence length: {}'.format(max_sequence_length))
        logger.info('{} dataset sparsity: {}'.format(
            config['name'], (len(train_dataset) + len(test_dataset)) / max_user_idx / max_item_idx
        ))

        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            dataset=train_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=validation_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=test_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_idx,
            num_items=max_item_idx,
            max_sequence_length=max_sequence_length
        )

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
            'num_users': self.num_users,
            'num_items': self.num_items,
            'max_sequence_length': self.max_sequence_length
        }

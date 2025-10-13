import torch
import torch.nn.functional as F
from torch import nn


class GraphEncoder(nn.Module):
    def __init__(self, graph, dropout_rate, num_users, num_items):
        super().__init__()
        self._graph = graph
        self._dropout_rate = dropout_rate
        self._num_users = num_users
        self._num_items = num_items

    def forward(self, user_embeddings, item_embeddings, ind=None):
        ego_embeddings = torch.cat(
            (user_embeddings.weight, item_embeddings.weight), dim=0
        )
        if ind is not None:
            ego_embeddings = torch.cat(
                (
                    ego_embeddings[:ind],
                    self._newuser_embeddings.weight,
                    ego_embeddings[ind + 1 :],
                ),
                dim=0,
            )
        all_embeddings = [ego_embeddings]

        if self._dropout_rate > 0:  # drop some edges
            if self.training:  # training_mode
                size = self._graph.size()
                index = self._graph.indices().t()
                values = self._graph.values()
                random_index = torch.rand(len(values)) + (1 - self._dropout_rate)
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / (1 - self._dropout_rate)
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._graph
        else:
            graph_dropped = self._graph

        for i in range(1):
            ego_embeddings = torch.sparse.mm(graph_dropped, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.mean(torch.stack(all_embeddings, dim=-1), dim=-1)
        user_final_embeddings, item_final_embeddings = torch.split(
            all_embeddings, [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

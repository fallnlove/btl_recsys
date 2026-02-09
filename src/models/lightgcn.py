import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LightGCN as PygLightGCN
from torch_geometric.utils import dropout_edge

from src.base import BaseModel
from src.metrics import NDCGMetric


def select_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class LightGCN(BaseModel):
    def __init__(
        self,
        rank,
        n_epochs,
        learning_rate,
        regularization,
        batch_size,
        device,
        seed,
        verbose,
        name,
        n_layers,
        edge_dropout,
        n_valid,
        patience,
        val_top_n,
        n_negatives,
        foldin_epochs,
        foldin_lr,
        foldin_reg,
        need_downvote,
    ):
        super().__init__(name)

        self.rank = int(rank)
        self.n_epochs = int(n_epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.batch_size = int(batch_size)

        self.device = select_device(device)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.n_layers = int(n_layers)
        self.edge_dropout = float(edge_dropout)
        self.n_negatives = int(n_negatives)

        self.n_valid = max(1, int(n_valid))
        self.patience = max(1, int(patience))
        self.val_top_n = int(val_top_n)

        self.foldin_epochs = int(foldin_epochs)
        self.foldin_lr = float(foldin_lr)
        self.foldin_reg = float(foldin_reg)
        self.foldin_batch_size = int(batch_size)
        self.need_downvote = bool(need_downvote)

        self.model = None
        self.n_users = 0
        self.n_items = 0

        self.user_item_edge_index = None
        self.edge_index = None

        self.loss_history = []
        self.val_history = []
        self.trained_epochs = 0

    def _build_graph(self, user_ids, item_ids):
        u = torch.as_tensor(user_ids, device=self.device, dtype=torch.long)
        i = torch.as_tensor(item_ids, device=self.device, dtype=torch.long) + self.n_users

        self.user_item_edge_index = torch.stack([u, i], dim=0)
        self.edge_index = torch.cat([self.user_item_edge_index, self.user_item_edge_index.flip(0)], dim=1)

        data = np.ones(len(user_ids), dtype=np.bool_)
        self.train_seen = sp.csr_matrix((data, (user_ids, item_ids)), shape=(self.n_users, self.n_items))

    def _bpr_loss(self, user_embs, item_embs, users, pos_items):
        batch_size = users.size(0)

        neg_items = torch.randint(0, self.n_items, (batch_size, self.n_negatives), device=self.device)
        for _ in range(5):
            mask_batch = (neg_items == pos_items.unsqueeze(1))
            if mask_batch.any():
                new_samples = torch.randint(0, self.n_items, neg_items.shape, device=self.device)
                neg_items = torch.where(mask_batch, new_samples, neg_items)
            else:
                break

        u = user_embs[users]
        p = item_embs[pos_items]
        n = item_embs[neg_items]

        pos_scores = (u * p).sum(dim=1, keepdim=True)
        neg_scores = (u.unsqueeze(1) * n).sum(dim=-1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        return loss

    def _foldin_optimize_users(self, pos_history_new, item_embs, u_init):
        if self.foldin_epochs <= 0:
            return u_init

        with torch.enable_grad():
            mask_pos = pos_history_new != -1
            if not mask_pos.any():
                return u_init

            pos_users = torch.nonzero(mask_pos, as_tuple=False)[:, 0]
            pos_items = pos_history_new[mask_pos]
            n_pairs = pos_items.numel()

            u = u_init.detach().clone()
            u.requires_grad_(True)
            
            opt = torch.optim.Adam([u], lr=self.foldin_lr)

            for _ in range(self.foldin_epochs):
                perm = torch.randperm(n_pairs, device=self.device)
                pu = pos_users[perm]
                pi = pos_items[perm]

                num_batches = (n_pairs + self.foldin_batch_size - 1) // self.foldin_batch_size
                for b in range(num_batches):
                    s = b * self.foldin_batch_size
                    e = min((b + 1) * self.foldin_batch_size, n_pairs)
                    if s >= e:
                        continue

                    bu = pu[s:e]
                    bp = pi[s:e]

                    loss = self._bpr_loss(u, item_embs, bu, bp)
                    reg = 0.5 * self.foldin_reg * u[bu].norm(dim=1).pow(2).mean()
                    total_loss = loss + reg

                    opt.zero_grad()
                    total_loss.backward()
                    opt.step()

            return u.detach()

    def fit(self, train_dataset, val_dataset=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.n_users = int(train_dataset.n_users)
        self.n_items = int(train_dataset.n_items)

        coo = train_dataset.get_coo_array()
        user_ids = coo.row.astype(np.int64)
        item_ids = coo.col.astype(np.int64)
        n_pos = len(user_ids)

        self._build_graph(user_ids, item_ids)

        self.model = PygLightGCN(
            num_nodes=self.n_users + self.n_items,
            embedding_dim=self.rank,
            num_layers=self.n_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        all_users = self.user_item_edge_index[0]
        all_pos_nodes = self.user_item_edge_index[1]

        use_val = val_dataset is not None
        best_metric = -np.inf
        best_state = None
        patience_cnt = 0
        self.trained_epochs = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0

            perm = torch.randperm(n_pos, device=self.device)
            users = all_users[perm]
            pos_nodes = all_pos_nodes[perm]
            
            num_batches = (n_pos + self.batch_size - 1) // self.batch_size

            for b in range(num_batches):
                start = b * self.batch_size
                end = min((b + 1) * self.batch_size, n_pos)
                if start >= end:
                    continue

                batch_users = users[start:end]
                batch_pos_items = pos_nodes[start:end] - self.n_users

                ui = self.user_item_edge_index
                if self.edge_dropout > 0.0:
                    ui, _ = dropout_edge(ui, p=self.edge_dropout, training=True)
                edge_index = torch.cat([ui, ui.flip(0)], dim=1)

                all_embs = self.model.get_embedding(edge_index)
                user_embs = all_embs[: self.n_users]
                item_embs = all_embs[self.n_users :]

                bpr = self._bpr_loss(user_embs, item_embs, batch_users, batch_pos_items)

                node_ids = torch.unique(torch.cat([batch_users, pos_nodes[start:end]], dim=0))
                e0 = self.model.embedding.weight[node_ids]
                reg = 0.5 * e0.norm(dim=1).pow(2).mean()

                loss = bpr + self.regularization * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())

            epoch_loss /= max(1, num_batches)
            self.loss_history.append(epoch_loss)

            do_val = use_val and ((epoch + 1) % self.n_valid == 0)
            if do_val:
                metric = self._validate(val_dataset)
                self.val_history.append(metric)

                if self.verbose:
                    print(f"Epoch {epoch + 1}/{self.n_epochs} - loss {epoch_loss:.4f} | val NDCG@{self.val_top_n}: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    self.trained_epochs = epoch + 1
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break
            else:
                if self.verbose:
                    print(f"Epoch {epoch + 1}/{self.n_epochs} - loss {epoch_loss:.4f}")

        if use_val and best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            self.trained_epochs = self.n_epochs

    @torch.no_grad()
    def _validate(self, val_dataset):
        self.model.eval()
        predictions = self.predict(val_dataset, top_n=self.val_top_n)
        holdout_users = val_dataset.get_holdout_users()
        metric = NDCGMetric(self.val_top_n)
        val_metric = metric(predictions[holdout_users, :], val_dataset.get_holdout_array()[holdout_users])
        return float(val_metric)

    def predict(self, dataset, top_n=10):
        if self.model is None or self.edge_index is None:
            raise ValueError("Model must be trained")

        self.model.eval()

        with torch.no_grad():
            all_embs = self.model.get_embedding(self.edge_index)
            user_embs_tr = all_embs[: self.n_users].detach()
            item_embs = all_embs[self.n_users :].detach()

        predictions = np.zeros((dataset.n_users, top_n), dtype=np.int64)
        dataloader = dataset.get_dataloader(batch_size=self.batch_size, shuffle=False)

        for batch in dataloader:
            user_raw = batch["user_id"]
            hist_raw = batch["history"]

            user_ids = torch.as_tensor(user_raw, device=self.device, dtype=torch.long)
            history_all = torch.as_tensor(hist_raw, device=self.device, dtype=torch.long)

            u_init = user_embs_tr[user_ids]

            if self.foldin_epochs > 0:
                user_vecs = self._foldin_optimize_users(
                    pos_history_new=history_all,
                    item_embs=item_embs,
                    u_init=u_init,
                )
            else:
                user_vecs = u_init

            scores = user_vecs @ item_embs.t()

            if self.need_downvote:
                mask = history_all != -1
                if mask.any():
                    min_val = scores.min() - 1.0
                    lengths = mask.sum(dim=1)
                    rows = torch.repeat_interleave(torch.arange(history_all.size(0), device=self.device), lengths)
                    cols = history_all[mask]
                    scores[rows, cols] = min_val

            _, top_idx = torch.topk(scores, k=top_n, dim=1)
            predictions[user_raw] = top_idx.detach().cpu().numpy()

        return predictions

    def suggest_additional_params(self):
        return {"n_epochs": int(self.trained_epochs)}

import math
import os
import numpy as np
import torch
from torch_geometric.nn.models import LightGCN as PygLightGCN
from torch_geometric.utils import dropout_edge
from torch_geometric.nn.models.lightgcn import BPRLoss

from src.base import BaseModel
from src.metrics import NDCGMetric


def select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(device)


class LightGCN(BaseModel):
    def __init__(
            self,
            rank: int,
            n_epochs: int,
            learning_rate: float,
            regularization: float,
            batch_size: int,
            device: str = "auto",
            seed: int = 42,
            verbose: bool = False,
            name: str = "LightGCN",
            n_layers: int = 3,
            edge_dropout: float = 0.0,
            lambda_geo: int = 256,
            eta: int = 1,
            resample_every_factor: float = 1.0,
            n_epochs_folding: int = 5,
            min_folding_epochs: int = 2,
            folding_epsilon: float = 1e-3,
            folding_patience: int = 2,
            n_valid: int = 1,
            patience: int = 3,
            val_top_n: int = 10,
    ):
        super().__init__(name)

        self.rank = rank
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size

        self.device = select_device(device)
        self.seed = seed
        self.verbose = verbose

        self.n_layers = n_layers
        self.edge_dropout = edge_dropout

        self.lambda_geo = lambda_geo
        self.eta = eta
        self.resample_every_factor = resample_every_factor

        self.n_epochs_folding = n_epochs_folding
        self.min_folding_epochs = min_folding_epochs
        self.folding_epsilon = folding_epsilon
        self.folding_patience = folding_patience

        self.n_valid = max(1, int(n_valid))
        self.patience = max(1, int(patience))
        self.val_top_n = int(val_top_n)

        self.model: PygLightGCN | None = None
        self.n_users: int = 0
        self.n_items: int = 0
        self.edge_index: torch.Tensor | None = None
        self.user_item_edge_index: torch.Tensor | None = None

        self.loss_history: list[float] = []
        self.val_history: list[float] = []
        self.trained_epochs: int = 0

        self.seen_users: np.ndarray | None = None
        self.seen_items: np.ndarray | None = None

        self.r_inv_asc: torch.Tensor | None = None

    def _build_graph(self, user_ids: np.ndarray, item_ids: np.ndarray) -> None:
        users = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        items = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
        item_nodes = items + self.n_users

        self.user_item_edge_index = torch.stack([users, item_nodes], dim=0)

        self.edge_index = torch.stack(
            [torch.cat([users, item_nodes]), torch.cat([item_nodes, users])],
            dim=0,
        )

    @torch.no_grad()
    def _update_r_inv_asc(self, item_embs: torch.Tensor | None = None) -> None:
        if self.model is None:
            return

        if item_embs is None:
            if self.edge_index is None:
                raise ValueError("edge_index is not set")
            z = self.model.get_embedding(self.edge_index)
            item_embs = z[self.n_users: self.n_users + self.n_items]

        r_inv_asc = torch.argsort(item_embs, dim=0)
        self.r_inv_asc = r_inv_asc.t().contiguous().to(self.device, dtype=torch.long)

    @torch.no_grad()
    def _sample_negative_adaptive_batch(self, user_embs: torch.Tensor) -> torch.Tensor:
        if self.r_inv_asc is None:
            raise RuntimeError("r_inv_asc is not initialized")

        B, K = user_embs.shape
        device = user_embs.device

        abs_user = user_embs.abs()
        sum_abs = abs_user.sum(dim=1, keepdim=True)

        zero_mask = sum_abs <= 0
        sum_abs_safe = sum_abs.clone()
        sum_abs_safe[zero_mask] = 1.0

        probs = abs_user / sum_abs_safe
        if zero_mask.any():
            probs[zero_mask.expand_as(probs)] = 1.0 / K

        f_sel = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)

        rho = math.exp(-1.0 / float(self.lambda_geo))
        p = 1.0 - rho
        geom = torch.distributions.Geometric(probs=torch.tensor(p, device=device))
        r = geom.sample((B,)).to(torch.long)
        r.clamp_(min=1, max=self.n_items)

        gathered = user_embs.gather(1, f_sel.view(-1, 1)).squeeze(1)
        sign_positive = gathered >= 0

        item_rank = torch.empty(B, dtype=torch.long, device=device)
        item_rank[sign_positive] = self.n_items - r[sign_positive]
        item_rank[~sign_positive] = r[~sign_positive] - 1

        neg_item_idx = self.r_inv_asc[f_sel, item_rank]
        neg_item_nodes = neg_item_idx + self.n_users

        return neg_item_nodes

    def fit(self, train_dataset, val_dataset=None) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.n_users = int(train_dataset.n_users)
        self.n_items = int(train_dataset.n_items)

        coo = train_dataset.get_coo_array()
        user_ids = coo.row.astype(np.int64)
        item_ids = coo.col.astype(np.int64)
        n_pos = len(user_ids)

        self._build_graph(user_ids, item_ids)

        self.seen_users = np.zeros(self.n_users, dtype=bool)
        self.seen_users[np.unique(user_ids)] = True
        self.seen_items = np.zeros(self.n_items, dtype=bool)
        self.seen_items[np.unique(item_ids)] = True

        self.model = PygLightGCN(
            num_nodes=self.n_users + self.n_items,
            embedding_dim=self.rank,
            num_layers=self.n_layers,
        ).to(self.device)

        torch.nn.init.xavier_uniform_(self.model.embedding.weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        resample_every = max(
            1, int(self.n_items * np.log(max(self.n_items, 2)) * self.resample_every_factor)
        )

        all_users = self.user_item_edge_index[0]
        all_pos_nodes = self.user_item_edge_index[1]

        use_val = val_dataset is not None
        best_metric = -np.inf
        best_state_dict: dict[str, torch.Tensor] | None = None
        patience_cnt = 0
        self.trained_epochs = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0

            perm = torch.randperm(n_pos, device=self.device)
            users = all_users[perm]
            pos_nodes = all_pos_nodes[perm]

            num_batches = (n_pos + self.batch_size - 1) // self.batch_size
            steps_since_resample = resample_every

            for b in range(num_batches):
                start = b * self.batch_size
                end = min((b + 1) * self.batch_size, n_pos)
                if start >= end:
                    continue

                batch_users = users[start:end]
                batch_pos = pos_nodes[start:end]

                edge_index = self.edge_index
                if self.edge_dropout > 0.0:
                    edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=True)

                z = self.model.get_embedding(edge_index)
                item_embs = z[self.n_users: self.n_users + self.n_items].detach()

                if self.r_inv_asc is None or steps_since_resample >= resample_every:
                    self._update_r_inv_asc(item_embs)
                    steps_since_resample = 0

                user_emb_batch = z[batch_users].detach()
                batch_neg = self._sample_negative_adaptive_batch(user_emb_batch)

                pos_scores = (z[batch_users] * z[batch_pos]).sum(dim=-1)
                neg_scores = (z[batch_users] * z[batch_neg]).sum(dim=-1)

                node_id = torch.unique(torch.cat([batch_users, batch_pos, batch_neg], dim=0))

                loss = self.model.recommendation_loss(
                    pos_scores,
                    neg_scores,
                    node_id=node_id,
                    lambda_reg=self.regularization,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                steps_since_resample += batch_users.size(0)

            epoch_loss /= max(1, num_batches)
            self.loss_history.append(epoch_loss)

            do_val = use_val and ((epoch + 1) % self.n_valid == 0)
            if do_val:
                r_inv_backup = self.r_inv_asc
                try:
                    predictions = self.predict(
                        val_dataset,
                        top_n=self.val_top_n,
                    )
                finally:
                    self.r_inv_asc = r_inv_backup

                holdout_users = val_dataset.get_holdout_users()
                metric = NDCGMetric(self.val_top_n)
                val_metric = metric(
                    predictions[holdout_users, :],
                    val_dataset.get_holdout_array()[holdout_users],
                )
                val_metric_f = float(val_metric)
                self.val_history.append(val_metric_f)

                if self.verbose:
                    print(
                        f"Epoch {epoch + 1}/{self.n_epochs} - loss {epoch_loss:.4f} | "
                        f"val NDCG@{self.val_top_n}: {val_metric_f:.4f}"
                    )

                if val_metric_f < best_metric:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        if self.verbose:
                            print(f"Early stopping on epoch {epoch + 1}")
                        break
                else:
                    patience_cnt = 0
                    best_metric = val_metric_f
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self.trained_epochs = epoch + 1
            else:
                if self.verbose:
                    print(f"Epoch {epoch + 1}/{self.n_epochs} - loss {epoch_loss:.4f}")

        if use_val and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.r_inv_asc = None
        else:
            self.trained_epochs = self.trained_epochs or min(self.n_epochs, len(self.loss_history))

    def _fit_unseen_users_batch(
            self,
            user_ids: torch.Tensor,
            history_items: torch.Tensor,
            item_embs: torch.Tensor,
            mean_user_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model is not trained")

        device = self.device
        user_ids = user_ids.to(device, dtype=torch.long)
        history_items = history_items.to(device, dtype=torch.long)

        U, L = history_items.shape

        pos_mask = history_items != -1
        no_pos = pos_mask.sum(dim=1) == 0

        with torch.no_grad():
            base_weight = self.model.embedding.weight[user_ids]
            user_embs = base_weight.clone()

            zero_mask = user_embs.abs().sum(dim=1) == 0
            if zero_mask.any():
                scale = math.sqrt(1.0 / float(self.rank))
                user_embs[zero_mask] = torch.randn(
                    (int(zero_mask.sum().item()), self.rank),
                    device=device,
                ) * scale

            if no_pos.any():
                user_embs[no_pos] = mean_user_emb

        active_mask = ~no_pos
        if not active_mask.any():
            with torch.no_grad():
                self.model.embedding.weight[user_ids] = user_embs
            return user_embs

        active_history = history_items[active_mask]
        active_pos_mask = pos_mask[active_mask]
        active_embs_init = user_embs[active_mask]
        U_a, Lmax = active_history.shape

        if self.r_inv_asc is None:
            self._update_r_inv_asc(item_embs)

        bpr = BPRLoss(lambda_reg=self.regularization)

        with torch.enable_grad():
            active_embs = torch.nn.Parameter(active_embs_init)
            optimizer = torch.optim.Adam([active_embs], lr=self.learning_rate)

            still_active = torch.ones(U_a, dtype=torch.bool, device=device)
            patience_cnt = torch.zeros(U_a, dtype=torch.long, device=device)

            epoch = 0
            while epoch < self.n_epochs_folding and still_active.any():
                old_embs = active_embs.detach().clone()

                perm_t = torch.randperm(Lmax, device=device)
                for t in perm_t:
                    mask_t = active_pos_mask[:, t] & still_active
                    if not mask_t.any():
                        continue

                    for _ in range(self.eta):
                        optimizer.zero_grad()

                        pos_items_t = active_history[mask_t, t]
                        u_batch = active_embs[mask_t]
                        pos_vecs = item_embs[pos_items_t]

                        with torch.no_grad():
                            neg_nodes = self._sample_negative_adaptive_batch(u_batch.detach())
                        neg_vecs = item_embs[neg_nodes - self.n_users]

                        pos_scores = (u_batch * pos_vecs).sum(dim=1)
                        neg_scores = (u_batch * neg_vecs).sum(dim=1)
                        loss = bpr(pos_scores, neg_scores, u_batch)

                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    diff = torch.norm(active_embs.detach() - old_embs, dim=1)

                    if epoch >= self.min_folding_epochs - 1:
                        inc_mask = (diff < self.folding_epsilon) & still_active
                        reset_mask = (diff >= self.folding_epsilon) & still_active

                        patience_cnt[inc_mask] += 1
                        patience_cnt[reset_mask] = 0

                        finished = (patience_cnt >= self.folding_patience) & still_active
                        still_active[finished] = False

                epoch += 1

            final_active_embs = active_embs.detach()

        with torch.no_grad():
            with torch.no_grad():
                self.model.embedding.weight[user_ids] = user_embs
            user_embs[active_mask] = final_active_embs

        return user_embs

    @torch.no_grad()
    def predict(self, dataset, top_n: int = 10) -> np.ndarray:
        if self.model is None or self.edge_index is None:
            raise ValueError("Model must be trained before calling predict")

        self.model.eval()
        dataloader = dataset.get_dataloader(batch_size=self.batch_size, shuffle=False)

        z = self.model.get_embedding(self.edge_index)
        base_user_embs = z[: self.n_users]
        item_embs = z[self.n_users: self.n_users + self.n_items]
        mean_user_emb = base_user_embs.mean(dim=0)

        self._update_r_inv_asc(item_embs)

        predictions = np.zeros((dataset.n_users, top_n), dtype=np.int64)

        for batch in dataloader:
            user_raw = batch["user_id"]
            hist_raw = batch["history"]

            batch_users = torch.as_tensor(user_raw, dtype=torch.long, device=self.device)
            history = torch.as_tensor(hist_raw, dtype=torch.long, device=self.device)

            B = batch_users.size(0)
            user_emb_batch = torch.empty((B, self.rank), device=self.device)

            if self.seen_users is not None:
                seen_np = self.seen_users[batch_users.cpu().numpy()]
            else:
                seen_np = np.zeros(B, dtype=bool)
            seen_mask = torch.from_numpy(seen_np).to(self.device)
            unseen_mask = ~seen_mask

            if seen_mask.any():
                user_emb_batch[seen_mask] = base_user_embs[batch_users[seen_mask]]

            if unseen_mask.any():
                unseen_users = batch_users[unseen_mask]
                unseen_hist = history[unseen_mask]
                unseen_embs = self._fit_unseen_users_batch(
                    unseen_users,
                    unseen_hist,
                    item_embs,
                    mean_user_emb
                )
                user_emb_batch[unseen_mask] = unseen_embs

            scores = user_emb_batch @ item_embs.t()

            rows = torch.arange(B, device=self.device).unsqueeze(1).expand_as(history)
            cols = history
            valid = cols != -1
            if valid.any():
                rows_flat = rows[valid]
                cols_flat = cols[valid]
                scores[rows_flat, cols_flat] = -float("inf")

            _, top_idx = torch.topk(scores, k=top_n, dim=1)
            predictions[batch_users.cpu().numpy()] = top_idx.cpu().numpy()

        return predictions

    def save_checkpoint(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained; nothing to save.")

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "seen_users": self.seen_users,
            "seen_items": self.seen_items,
            "edge_index": None if self.edge_index is None else self.edge_index.cpu(),
            "config": {
                "rank": self.rank,
                "n_users": self.n_users,
                "n_items": self.n_items,
                "n_layers": self.n_layers,
                "lambda_geo": self.lambda_geo,
                "eta": self.eta,
                "resample_every_factor": self.resample_every_factor,
                "n_epochs_folding": self.n_epochs_folding,
                "min_folding_epochs": self.min_folding_epochs,
                "folding_epsilon": self.folding_epsilon,
                "folding_patience": self.folding_patience,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "batch_size": self.batch_size,
                "n_valid": self.n_valid,
                "patience": self.patience,
                "val_top_n": self.val_top_n,
            },
        }

        torch.save(checkpoint, path)

        if self.verbose:
            print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint.get("config", {})

        self.rank = int(config.get("rank", self.rank))
        self.n_users = int(config.get("n_users", self.n_users))
        self.n_items = int(config.get("n_items", self.n_items))
        self.n_layers = int(config.get("n_layers", self.n_layers))

        self.lambda_geo = int(config.get("lambda_geo", self.lambda_geo))
        self.eta = int(config.get("eta", self.eta))
        self.resample_every_factor = float(
            config.get("resample_every_factor", self.resample_every_factor)
        )
        self.n_epochs_folding = int(config.get("n_epochs_folding", self.n_epochs_folding))
        self.min_folding_epochs = int(config.get("min_folding_epochs", self.min_folding_epochs))
        self.folding_epsilon = float(config.get("folding_epsilon", self.folding_epsilon))
        self.folding_patience = int(config.get("folding_patience", self.folding_patience))

        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.regularization = float(config.get("regularization", self.regularization))
        self.batch_size = int(config.get("batch_size", self.batch_size))

        self.n_valid = max(1, int(config.get("n_valid", self.n_valid)))
        self.patience = max(1, int(config.get("patience", self.patience)))
        self.val_top_n = int(config.get("val_top_n", self.val_top_n))

        self.seen_users = checkpoint.get("seen_users", None)
        self.seen_items = checkpoint.get("seen_items", None)

        self.model = PygLightGCN(
            num_nodes=self.n_users + self.n_items,
            embedding_dim=self.rank,
            num_layers=self.n_layers,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        edge_index = checkpoint.get("edge_index", None)
        self.edge_index = None if edge_index is None else edge_index.to(self.device)

        self.r_inv_asc = None

        if self.verbose:
            print(
                f"Checkpoint loaded. Rank: {self.rank}, Users: {self.n_users}, Items: {self.n_items}"
            )

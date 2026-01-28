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

        foldin_epochs,
        foldin_lr,
        foldin_reg,
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

        self.n_valid = max(1, int(n_valid))
        self.patience = max(1, int(patience))
        self.val_top_n = int(val_top_n)

        self.catalog_chunk_size = None

        self.foldin_epochs = int(foldin_epochs)
        self.foldin_lr = float(foldin_lr)
        self.foldin_reg = float(foldin_reg)
        self.foldin_batch_size = int(batch_size)

        self.model = None
        self.n_users = 0
        self.n_items = 0

        self.user_item_edge_index = None
        self.edge_index = None

        self.train_seen = None

        self.loss_history = []
        self.val_history = []
        self.trained_epochs = 0

    def _auto_chunk_size(self, n_items: int, batch_size: int, rank: int, gpu_gb: float = None) -> int:
        """Auto-select chunk_size based on GPU memory."""
        if gpu_gb is None:
            if self.device.type == "cuda":
                gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            else:
                gpu_gb = 4.0

        target_mem = gpu_gb * 0.7 * 1024**3  # bytes
        
        mem_for_scores = target_mem * 0.7
        
        bytes_per_score = rank * 4
        max_chunk = int(mem_for_scores / (batch_size * bytes_per_score))
        
        chunk = min(max_chunk, n_items // 2)
        chunk = max(chunk, 512)
        
        if self.verbose:
            print(f"Auto chunk_size: {chunk} (GPU: {gpu_gb:.1f}GB, batch: {batch_size}, rank: {rank})")
        
        return chunk

    def _build_graph(self, user_ids, item_ids):
        u = torch.as_tensor(user_ids, device=self.device, dtype=torch.long)
        i = torch.as_tensor(item_ids, device=self.device, dtype=torch.long) + self.n_users

        self.user_item_edge_index = torch.stack([u, i], dim=0)
        self.edge_index = torch.cat([self.user_item_edge_index, self.user_item_edge_index.flip(0)], dim=1)

        data = np.ones(len(user_ids), dtype=np.bool_)
        self.train_seen = sp.csr_matrix((data, (user_ids, item_ids)), shape=(self.n_users, self.n_items))

    @staticmethod
    def _mask_seen_scores_in_chunk(scores, batch_users, seen_csr, start, end, device):
        users_np = batch_users.detach().cpu().numpy()
        indptr = seen_csr.indptr
        indices = seen_csr.indices

        rows_list, cols_list = [], []
        for r, u in enumerate(users_np):
            row_idx = indices[indptr[u] : indptr[u + 1]]
            if row_idx.size == 0:
                continue
            m = (row_idx >= start) & (row_idx < end)
            if np.any(m):
                cols = row_idx[m] - start
                rows = np.full(cols.shape[0], r, dtype=np.int64)
                rows_list.append(rows)
                cols_list.append(cols)

        if rows_list:
            rows = torch.from_numpy(np.concatenate(rows_list)).to(device)
            cols = torch.from_numpy(np.concatenate(cols_list)).to(device)
            scores[rows, cols] = -1e9

    def _bpr_full_catalog_loss(self, batch_users, batch_pos_items, user_embs, item_embs, seen_csr, chunk_size):
        u = user_embs[batch_users]
        p = item_embs[batch_pos_items]
        pos_scores = (u * p).sum(dim=-1, keepdim=True)

        total = u.new_tensor(0.0)
        denom = u.new_tensor(0.0)

        for start in range(0, self.n_items, chunk_size):
            end = min(start + chunk_size, self.n_items)
            scores = u @ item_embs[start:end].t()

            self._mask_seen_scores_in_chunk(scores, batch_users, seen_csr, start, end, scores.device)

            valid = scores > -5e8
            loss_mat = -F.logsigmoid(pos_scores - scores)

            total = total + (loss_mat * valid.float()).sum()
            denom = denom + valid.sum().float()

        return total / denom.clamp(min=1.0)

    def _build_seen_hist_csr(self, history_items, n_items):
        hist = history_items.detach().cpu().numpy()
        mask = hist != -1
        if not mask.any():
            return sp.csr_matrix((hist.shape[0], n_items), dtype=bool)

        rows, cols = np.where(mask)
        items = hist[rows, cols]
        data = np.ones(len(rows), dtype=bool)
        return sp.csr_matrix((data, (rows, items)), shape=(hist.shape[0], n_items))

    def _extract_new_items_only(self, user_ids, history_all):
        users = user_ids.detach().cpu().numpy()
        hist = history_all.detach().cpu().numpy()

        B, L = hist.shape
        out = np.full((B, L), -1, dtype=np.int64)
        any_new = False

        indptr = self.train_seen.indptr
        indices = self.train_seen.indices

        for r, u in enumerate(users):
            items = hist[r]
            valid = items != -1
            if not np.any(valid):
                continue

            cand = items[valid].astype(np.int64)

            seen_u = indices[indptr[u] : indptr[u + 1]]
            if seen_u.size == 0:
                new = cand
            else:
                m = ~np.isin(cand, seen_u, assume_unique=False)
                new = cand[m]

            if new.size > 0:
                any_new = True
                out[r, : new.size] = new

        return torch.as_tensor(out, device=history_all.device, dtype=torch.long), any_new

    def _foldin_optimize_users(
        self,
        user_ids_local,
        pos_history_new,
        seen_hist_csr,
        item_embs,
        u_init,
    ):
        if self.foldin_epochs <= 0:
            return u_init

        with torch.enable_grad():
            device = item_embs.device
            bsz = pos_history_new.size(0)

            mask_pos = pos_history_new != -1
            if not mask_pos.any():
                return u_init

            pos_users = torch.nonzero(mask_pos, as_tuple=False)[:, 0]
            pos_items = pos_history_new[mask_pos]
            n_pairs = pos_items.numel()

            u = u_init.detach().clone()
            u.requires_grad_(True)
            
            opt = torch.optim.Adam([u], lr=self.foldin_lr, betas=(0.9, 0.999), eps=1e-8)

            for _ in range(self.foldin_epochs):
                perm = torch.randperm(n_pairs, device=device)
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

                    u_sub = u[bu]
                    p_emb = item_embs[bp]
                    pos_scores = (u_sub * p_emb).sum(dim=-1, keepdim=True)

                    total = u_sub.new_tensor(0.0)
                    denom = u_sub.new_tensor(0.0)

                    for start in range(0, self.n_items, self.catalog_chunk_size):
                        end = min(start + self.catalog_chunk_size, self.n_items)
                        scores = u_sub @ item_embs[start:end].t()

                        self._mask_seen_scores_in_chunk(scores, bu, seen_hist_csr, start, end, scores.device)

                        valid = scores > -5e8
                        loss_mat = -F.logsigmoid(pos_scores - scores)
                        total = total + (loss_mat * valid.float()).sum()
                        denom = denom + valid.sum().float()

                    bpr = total / denom.clamp(min=1.0)
                    reg = 0.5 * self.foldin_reg * u_sub.norm(dim=1).pow(2).mean()
                    loss = bpr + reg

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            return u.detach()


    def fit(self, train_dataset, val_dataset=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.n_users = int(train_dataset.n_users)
        self.n_items = int(train_dataset.n_items)

        if self.catalog_chunk_size is None:
            self.catalog_chunk_size = self._auto_chunk_size(
                n_items=self.n_items, 
                batch_size=self.batch_size, 
                rank=self.rank
            )

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
                batch_pos_nodes = pos_nodes[start:end]
                batch_pos_items = batch_pos_nodes - self.n_users

                ui = self.user_item_edge_index
                if self.edge_dropout > 0.0:
                    ui, _ = dropout_edge(ui, p=self.edge_dropout, training=True)
                edge_index = torch.cat([ui, ui.flip(0)], dim=1)

                all_embs = self.model.get_embedding(edge_index)
                user_embs = all_embs[: self.n_users]
                item_embs = all_embs[self.n_users :]

                bpr = self._bpr_full_catalog_loss(
                    batch_users=batch_users,
                    batch_pos_items=batch_pos_items,
                    user_embs=user_embs,
                    item_embs=item_embs,
                    seen_csr=self.train_seen,
                    chunk_size=self.catalog_chunk_size,
                )

                node_ids = torch.unique(torch.cat([batch_users, batch_pos_nodes], dim=0))
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
                    print(
                        f"Epoch {epoch + 1}/{self.n_epochs} - loss {epoch_loss:.4f} | "
                        f"val NDCG@{self.val_top_n}: {metric:.4f}"
                    )

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

            pos_history_new, any_new = self._extract_new_items_only(user_ids, history_all)

            seen_hist_csr = self._build_seen_hist_csr(history_all, self.n_items)

            u_init = user_embs_tr[user_ids]

            if any_new and self.foldin_epochs > 0:
                user_vecs = self._foldin_optimize_users(
                    user_ids_local=torch.arange(user_ids.size(0), device=self.device),
                    pos_history_new=pos_history_new,
                    seen_hist_csr=seen_hist_csr,
                    item_embs=item_embs,
                    u_init=u_init,
                )
            else:
                user_vecs = u_init

            scores = user_vecs @ item_embs.t()

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

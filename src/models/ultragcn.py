import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

from src.base import BaseModel
from src.metrics import Summarizer, NDCGMetric, RecallMetric, CoverageMetric
from src.models.utils.repeatable_rec import is_repeatable


class UltraGCN(BaseModel, nn.Module):
    """
    Base class for all models.
    """
    def __init__(
        self,
        name: str = "UltraGCN",
        embedding_dim: int = 64,
        w1: float = 1.0,
        w2: float = 0.1,
        w3: float = 0.1,
        w4: float = 0.1,
        negative_weight: float = 0.1,
        negative_num: int = 10,
        gamma: float = 0.1,
        lambda_reg: float = 1e-4,
        num_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 1024,
        num_neighbors: int = 10,
        early_stopping: bool = True,
        tracked_metric: str = "ndcg@10",
    ):
        BaseModel.__init__(self, name)
        nn.Module.__init__(self)
        self.embedding_dim = embedding_dim
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.negative_weight = negative_weight
        self.negative_num = negative_num
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_neighbors = num_neighbors
        self.early_stopping = early_stopping
        self.tracked_metric = tracked_metric
        self.trained_epochs = None

    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """
        self.n_users = train_dataset.n_users
        self.n_items = train_dataset.n_items

        self.seen_users = np.zeros(self.n_users, dtype=bool)
        self.seen_users[np.unique(
            train_dataset.get_coo_array().row.astype(np.int64)
        )] = True

        self.user_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users,
            embedding_dim=self.embedding_dim,
        ).to(self.device)
        self.item_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=self.embedding_dim,
        ).to(self.device)
        self._init_weights()

        self.user_betas, self.item_betas = self._compute_betas(
            train_dataset.get_coo_array()
        )
        self.constraint_matrix, self.neighbor_matrix = self._compute_item_item_matrix(
            train_dataset.get_coo_array(),
            dataset_name=train_dataset.name,
        )

        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.lambda_reg,
        )

        data = train_dataset.get_coo_array().coords
        dataset = TensorDataset(
            torch.tensor(data[0], dtype=torch.long),
            torch.tensor(data[1], dtype=torch.long),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_metric = -np.inf
        no_improve_epochs = 0
        metrics_module = Summarizer(
            [NDCGMetric(k=k) for k in [5, 10, 20]] +
            [RecallMetric(k=k) for k in [5, 10, 20]] +
            [CoverageMetric(k=k, n_items=self.n_items) for k in [5, 10, 20]],
        )

        for epoch in tqdm(range(self.num_epochs), desc="Training epochs"):
            self.trained_epochs = epoch
            self.train()
            total_loss = 0.0
            for batch in dataloader:
                users = batch[0].to(self.device)
                pos_items = batch[1].to(self.device)
                neg_items = torch.randint(0, self.n_items, (len(users), self.negative_num), device=self.device)

                omega_weight = self.get_omegas(users, pos_items, neg_items)

                loss_L = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
                loss_I = self.cal_loss_I(users, pos_items)

                loss = loss_L + self.gamma * loss_I

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataset)

            if epoch >= 50 and epoch % 5 == 0 and val_dataset is not None:
                holdout_users = val_dataset.get_holdout_users()
                val_predictions = self.predict(val_dataset, top_n=20)[holdout_users]
                val_targets = val_dataset.get_holdout_array()[holdout_users]
                current_metric = metrics_module(val_predictions, val_targets)[self.tracked_metric]

                if current_metric > best_metric:
                    best_metric = current_metric
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if self.early_stopping and no_improve_epochs >= 5:
                    break

    def suggest_additional_params(self) -> dict:
        epochs = self.trained_epochs if self.trained_epochs else self.num_epochs
        return {"num_epochs": int(epochs)}

    def _partial_fit(self, _users, history_items):
        torch.nn.init.xavier_uniform_(self.user_embeddings(_users))
        self.train()
        optimizer = torch.optim.AdamW(
            params=self.user_embeddings.parameters(),
            lr=self.lr,
            weight_decay=self.lambda_reg,
        )
        mask = (history_items != -1)
        count = mask.sum(dim=1)

        dataset = TensorDataset(
            _users.repeat_interleave(count).long(),
            history_items[mask].reshape(-1).long(),
        )
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        for epoch in range(5):
            for batch in dataloader:
                users = batch[0].to(self.device)
                pos_items = batch[1].to(self.device)
                neg_items = torch.randint(0, self.n_items, (len(users), self.negative_num), device=self.device)

                omega_weight = self.get_omegas(users, pos_items, neg_items)

                loss_L = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
                loss_I = self.cal_loss_I(users, pos_items)

                loss = loss_L + self.gamma * loss_I

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, dataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.
        """
        user_embeddings = self.user_embeddings.weight.clone().detach()
        self.eval()
        result = np.zeros((dataset.n_users, top_n), dtype=np.int64)

        for batch in tqdm(dataset.get_dataloader(batch_size=1024, shuffle=False), desc="Predicting"):
            users = batch["user_id"].to(self.device)
            history = batch["history"].to(self.device)
            user_embeds = self.user_embeddings(users)  # batch_size * dim
            all_item_embeds = self.item_embeddings.weight  # n_items * dim

            # seen_users = self.seen_users[users.cpu().numpy()]
            # unseen_mask = ~torch.from_numpy(seen_users).to(self.device)
            # if unseen_mask.any():
            #     unseen_users = users[unseen_mask]
            #     unseen_hist = history[unseen_mask]
                # self._partial_fit(
                #     unseen_users,
                #     unseen_hist,
                # )
                # self.eval()
            self._partial_fit(
                users,
                history,
            )
            self.eval()

            with torch.no_grad():
                scores = torch.matmul(user_embeds, all_item_embeds.T)  # batch_size * n_items

            if not is_repeatable(dataset):
                for i in range(len(users)):
                    scores[i, history[i, history[i] != -1]] = -np.inf

            _, top_items = torch.topk(scores, top_n, dim=-1)  # batch_size * top_n

            result[users.cpu().numpy(), :] = top_items.cpu().numpy()

        self.user_embeddings.weight.data.copy_(user_embeddings)
        self.user_embeddings.weight.requires_grad_(True)

        return result

    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        
        torch.save({
            'user_embeddings': self.user_embeddings.state_dict(),
            'item_embeddings': self.item_embeddings.state_dict(),
            'user_betas': self.user_betas,
            'item_betas': self.item_betas,
            'constraint_matrix': self.constraint_matrix,
            'neighbor_matrix': self.neighbor_matrix,
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        
        checkpoint = torch.load(path, map_location=self.device)
        self.user_embeddings.load_state_dict(checkpoint['user_embeddings'])
        self.item_embeddings.load_state_dict(checkpoint['item_embeddings'])
        self.user_betas = checkpoint['user_betas']
        self.item_betas = checkpoint['item_betas']
        self.constraint_matrix = checkpoint['constraint_matrix']
        self.neighbor_matrix = checkpoint['neighbor_matrix']

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
    
    def _compute_betas(self, coo_matrix):
        """
        Compute the beta coefficients for the UltraGCN model.
        """
        user_degrees = np.array(coo_matrix.sum(axis=1)).flatten()
        item_degrees = np.array(coo_matrix.sum(axis=0)).flatten()

        betas_user = np.zeros_like(user_degrees, dtype=np.float32)
        betas_user[user_degrees > 0] = np.sqrt(user_degrees[user_degrees > 0] + 1) / user_degrees[user_degrees > 0]
        betas_item = 1 / np.sqrt(item_degrees + 1)

        return torch.tensor(betas_user, dtype=torch.float32).to(self.device), \
               torch.tensor(betas_item, dtype=torch.float32).to(self.device)

    def _compute_item_item_matrix(self, coo_matrix, dataset_name: str):
        """
        Compute the item-item co-occurrence matrix.
        """
        save_path = Path(f"ultragcn_cache/{dataset_name}/")
        if save_path.exists():
            constraint_matrix = torch.load(save_path / "constraint_matrix.pt").float().to(self.device)
            neighbor_matrix = torch.load(save_path / "neighbor_matrix.pt").long().to(self.device)
            return constraint_matrix, neighbor_matrix
        import cupy as cp
        import cupyx.scipy.sparse as csp

        A = coo_matrix.T.dot(coo_matrix)
        items_D = np.sum(A, axis = 0).reshape(-1)
        users_D = np.sum(A, axis = 1).reshape(-1)

        beta_uD = np.zeros_like(users_D, dtype=np.float32)
        beta_uD[users_D > 0] = np.sqrt(users_D[users_D > 0] + 1) / users_D[users_D > 0]
        beta_uD = beta_uD
        beta_iD = 1 / np.sqrt(items_D + 1)
        beta_iD, beta_uD = torch.from_numpy(beta_iD), torch.from_numpy(beta_uD)
        A_csr = csp.csr_matrix(
            (cp.asarray(A.data, dtype=cp.float32),
            cp.asarray(A.indices, dtype=cp.int32),
            cp.asarray(A.indptr, dtype=cp.int32)),
            shape=A.shape
        )
        beta_uD_cp = cp.asarray(beta_uD)
        beta_iD_cp = cp.asarray(beta_iD)
        K = self.num_neighbors
        n_items = self.n_items
        res_mat = cp.empty((n_items, K), dtype=cp.int32)
        res_sim_mat = cp.empty((n_items, K), dtype=cp.float32)
        for i in tqdm(range(n_items), desc="Computing item-item matrix"):
            a_row = A_csr.getrow(i).toarray().ravel()
            row = (beta_uD_cp[i] * beta_iD_cp) * a_row
            if K < row.size:
                idx_part = cp.argpartition(row, -K)[-K:]
            else:
                idx_part = cp.arange(row.size, dtype=cp.int32)

            vals_part = row[idx_part]
            order = cp.argsort(vals_part)[::-1]
            topk_idx = idx_part[order][:K]
            topk_val = row[topk_idx]

            res_mat[i] = topk_idx.astype(cp.int32)
            res_sim_mat[i] = topk_val.astype(cp.float32)

        res_mat_torch = torch.utils.dlpack.from_dlpack(res_mat.toDlpack())
        res_sim_mat_torch = torch.utils.dlpack.from_dlpack(res_sim_mat.toDlpack())
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(res_sim_mat_torch.cpu(), save_path / "constraint_matrix.pt")
        torch.save(res_mat_torch.cpu(), save_path / "neighbor_matrix.pt")

        return res_sim_mat_torch.float().to(self.device), res_mat_torch.long().to(self.device)

    def get_omegas(self, users, pos_items, neg_items):
        if self.w2 > 0:
            pos_weight = torch.mul(self.user_betas[users], self.item_betas[pos_items])
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items), device=self.device)

        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.user_betas[users], neg_items.size(1)), self.item_betas[neg_items.flatten()])
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1), device=self.device)

        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        user_embeds = self.user_embeddings(users)
        pos_embeds = self.item_embeddings(pos_items)
        neg_embeds = self.item_embeddings(neg_items)
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size(), device=self.device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction="none").mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size(), device=self.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction="none")

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        neighbor_embeds = self.item_embeddings(self.neighbor_matrix[pos_items])    # len(pos_items) * num_neighbors * dim
        sim_scores = self.constraint_matrix[pos_items]     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeddings(users).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        return loss.sum()

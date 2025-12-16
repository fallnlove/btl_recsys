import numpy as np
from numba import njit, prange
from tqdm import tqdm
import os

from src.base import BaseModel
from src.metrics import NDCGMetric


@njit
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1.0 + z)


@njit
def update_user_vector_adaptive(p_u, Q_pos, Q, r_inv_asc, rank, n_items, lr, reg, lambda_geo, eta):
    n_pos = len(Q_pos)
    indices = np.arange(n_pos)
    np.random.shuffle(indices)

    for pos_idx in indices:
        i_pos_vec = Q_pos[pos_idx]

        rho = np.exp(-1.0 / lambda_geo)

        for _ in range(eta):
            r = 1
            while np.random.rand() < rho:
                r += 1
            r = min(r, n_items)
            abs_pu = np.abs(p_u)
            sum_abs = np.sum(abs_pu)
            if sum_abs == 0:
                f_sel = np.random.randint(rank)
            else:
                probs = abs_pu / sum_abs
                f_sel = sample_index_from_probs(probs)

            if p_u[f_sel] >= 0:
                j_neg_idx = r_inv_asc[f_sel, n_items - r]
            else:
                j_neg_idx = r_inv_asc[f_sel, r - 1]
            q_neg_vec = Q[j_neg_idx]

            # positive
            pred_pos = np.dot(p_u, i_pos_vec)
            err_pos = 1.0 - sigmoid(pred_pos)
            p_u += lr * (err_pos * i_pos_vec - reg * p_u)

            # negative
            pred_neg = np.dot(p_u, q_neg_vec)
            err_neg = 0.0 - sigmoid(pred_neg)
            p_u += lr * (err_neg * q_neg_vec - reg * p_u)

    return p_u


@njit(parallel=True)
def process_batch_with_folding_in(
        user_ids,
        history,
        P,
        Q,
        r_inv_asc,
        rank,
        lr,
        reg,
        max_epochs_folding,
        min_folding_epochs,
        epsilon,
        is_seen_mask,
        lambda_geo,
        eta,
        patience
):
    batch_size = len(user_ids)
    P_batch = np.zeros((batch_size, rank), dtype=np.float32)
    n_items = Q.shape[0]

    for b in prange(batch_size):
        uid = user_ids[b]

        if is_seen_mask[uid]:
            P_batch[b] = P[uid]
        else:
            scale = np.sqrt(1.0 / rank)
            p_u = np.random.normal(0, scale, size=rank).astype(np.float32)

            row = history[b]
            valid_mask = row != -1
            valid_items = row[valid_mask]

            if len(valid_items) > 0:
                Q_pos = Q[valid_items]

                epoch = 0
                patience_cnt = 0
                while epoch < max_epochs_folding:
                    p_u_old = p_u.copy()

                    p_u = update_user_vector_adaptive(
                        p_u, Q_pos, Q, r_inv_asc,
                        rank, n_items, lr, reg, lambda_geo, eta
                    )

                    diff_norm = np.linalg.norm(p_u - p_u_old)

                    if epoch >= min_folding_epochs - 1 and diff_norm < epsilon:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break
                    if diff_norm >= epsilon:
                        patience_cnt = 0
                    epoch += 1

            P_batch[b] = p_u

    return P_batch


@njit  # equivalent to np.random.choice with probs
def sample_index_from_probs(probs):
    total = 0.0
    for i in range(probs.shape[0]):
        total += probs[i]
    if total <= 0.0:
        return np.random.randint(probs.shape[0])
    r = np.random.rand() * total
    cumsum = 0.0
    for i in range(probs.shape[0]):
        cumsum += probs[i]
        if r <= cumsum:
            return i
    return probs.shape[0] - 1


@njit
def adaptive_sgd_epoch(P, Q, pos_users, pos_items, n_pos, n_steps, lambda_geo, eta, resample_every, lr, reg, rank,
                       n_items):
    r_inv_asc = np.empty((rank, n_items), dtype=np.int64)
    loss = 0.0
    step = 0

    while step < n_steps:
        if step % resample_every == 0:
            for f in range(rank):
                r_inv_asc[f] = np.argsort(Q[:, f])

        pos_idx = np.random.randint(n_pos)
        u = pos_users[pos_idx]
        i_pos = pos_items[pos_idx]
        pu = P[u].copy()

        r = 1
        rho = np.exp(-1.0 / lambda_geo)
        while np.random.rand() < rho:
            r += 1
        r = min(r, n_items)

        abs_pu = np.abs(pu)
        sum_abs = np.sum(abs_pu)
        if sum_abs == 0.0:
            f_sel = np.random.randint(rank)
        else:
            probs = abs_pu / sum_abs
            f_sel = sample_index_from_probs(probs)

        if pu[f_sel] >= 0:
            j_neg = r_inv_asc[f_sel, n_items - r]
        else:
            j_neg = r_inv_asc[f_sel, r - 1]

        for _ in range(eta):
            # positive sample
            pred_pos = np.dot(pu, Q[i_pos])
            err_pos = 1.0 - sigmoid(pred_pos)
            delta_p_pos = lr * (err_pos * Q[i_pos] - reg * pu)
            delta_q_pos = lr * (err_pos * pu - reg * Q[i_pos])
            pu += delta_p_pos
            Q[i_pos] += delta_q_pos
            loss += np.logaddexp(0, -pred_pos)

            # negative sample
            pred_neg = np.dot(pu, Q[j_neg])
            err_neg = 0.0 - sigmoid(pred_neg)
            delta_p_neg = lr * (err_neg * Q[j_neg] - reg * pu)
            delta_q_neg = lr * (err_neg * pu - reg * Q[j_neg])
            pu += delta_p_neg
            Q[j_neg] += delta_q_neg
            loss += np.logaddexp(0, pred_neg)

        P[u] = pu
        step += 1

    return loss / (n_steps * eta * 2) if n_steps > 0 else 0.0


class MFSGD(BaseModel):
    def __init__(
            self,
            rank,
            learning_rate,
            regularization,
            n_epochs,
            n_epochs_folding,
            min_folding_epochs,
            folding_epsilon,
            n_valid,
            patience: int = 3,
            seed: int = 42,
            name: str = "MF_SGD_Adaptive",
            verbose: bool = False,
            lambda_geo: int = 256,
            eta: int = 1,
            resample_every_factor: float = 1.0
    ):
        super().__init__(str(name))
        self.rank = int(rank)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.n_epochs = int(n_epochs)
        self.n_epochs_folding = int(n_epochs_folding)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.n_valid = int(n_valid)
        self.patience = int(patience)

        self.lambda_geo = int(lambda_geo)
        self.eta = int(eta)
        self.resample_every_factor = float(resample_every_factor)
        self.min_folding_epochs = int(min_folding_epochs)
        self.folding_epsilon = float(folding_epsilon)

        self.P = None
        self.Q = None
        self.seen_users = None
        self.trained_epochs = 0

    def fit(self, train_dataset, val_dataset):
        np.random.seed(self.seed)
        n_users = train_dataset.n_users
        n_items = train_dataset.n_items

        lambda_geo_val = int(self.lambda_geo)  # numba type problem fix
        eta_val = int(self.eta)
        lr_val = float(self.learning_rate)
        reg_val = float(self.regularization)
        rank_val = int(self.rank)
        n_items_val = int(n_items)
        resample_every_factor_val = float(self.resample_every_factor)

        scale = np.sqrt(1.0 / rank_val)
        self.P = np.random.normal(0, scale, size=(n_users, rank_val)).astype(np.float32)
        self.Q = np.random.normal(0, scale, size=(n_items, rank_val)).astype(np.float32)

        coo = train_dataset.get_coo_array()
        pos_users = coo.row.astype(np.int32)
        pos_items = coo.col.astype(np.int32)
        n_pos = len(pos_users)

        self.seen_users = np.zeros(n_users, dtype=bool)
        self.seen_users[np.unique(pos_users)] = True

        resample_every = max(1, int(n_items * np.log(n_items) * resample_every_factor_val))
        n_steps_per_epoch = n_pos

        if self.verbose:
            print(
                f"Training {self.name}. Users: {n_users}, Items: {n_items}, Steps/epoch: {n_steps_per_epoch}, Resample every: {resample_every} steps.")

        best_metric = -np.inf
        best_P = None
        best_Q = None
        patience_cnt = 0

        for epoch in tqdm(range(self.n_epochs), desc=f"Training {self.name}", disable=not self.verbose):
            avg_loss = adaptive_sgd_epoch(
                self.P, self.Q, pos_users, pos_items, n_pos, n_steps_per_epoch,
                lambda_geo_val, eta_val, resample_every,
                lr_val, reg_val, rank_val, n_items_val
            )

            if (epoch + 1) % self.n_valid == 0:
                predictions = self.predict(val_dataset, 10)
                holdout_users = val_dataset.get_holdout_users()
                metric = NDCGMetric(10)
                val_metric = metric(predictions[holdout_users, :], val_dataset.get_holdout_array()[holdout_users])

                if self.verbose:
                    print(
                        f"Epoch {epoch + 1}/{self.n_epochs} - BCE Loss: {avg_loss:.4f} | val NDCG@10: {val_metric:.4f}")

                if val_metric < best_metric:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        if self.verbose: print(f'Early stopping on epoch {epoch + 1}')
                        break
                else:
                    patience_cnt = 0
                    best_metric = val_metric
                    best_P = self.P.copy()
                    best_Q = self.Q.copy()
                    self.trained_epochs = epoch + 1

        self.P = best_P
        self.Q = best_Q

    def predict(self, dataset, top_n: int) -> np.ndarray:
        predictions = np.zeros((dataset.n_users, top_n), dtype=np.int64)
        dataloader = dataset.get_dataloader(batch_size=1024, shuffle=False)

        P_contiguous = np.ascontiguousarray(self.P)
        Q_contiguous = np.ascontiguousarray(self.Q)

        r_inv_asc = np.empty((self.rank, self.Q.shape[0]), dtype=np.int64)
        for f in range(self.rank):
            r_inv_asc[f] = np.argsort(Q_contiguous[:, f])

        seen_users_contiguous = np.ascontiguousarray(self.seen_users)

        for batch in tqdm(dataloader, desc="Predicting", disable=not self.verbose):
            batch_users = batch['user_id'].numpy().astype(np.int32)
            history = batch['history'].numpy().astype(np.int32)

            P_batch = process_batch_with_folding_in(
                batch_users, history, P_contiguous, Q_contiguous, r_inv_asc,
                self.rank, self.learning_rate, self.regularization,
                self.n_epochs_folding, self.min_folding_epochs, self.folding_epsilon,
                seen_users_contiguous, self.lambda_geo, self.eta, self.patience
            )

            scores = P_batch @ Q_contiguous.T

            rows = np.repeat(np.arange(len(batch_users)), history.shape[1])
            cols = history.ravel()
            valid_mask = cols != -1
            scores[rows[valid_mask], cols[valid_mask]] = -np.inf

            unsorted_top = np.argpartition(-scores, top_n, axis=1)[:, :top_n]
            top_scores = np.take_along_axis(scores, unsorted_top, axis=1)
            sorted_idx = np.argsort(-top_scores, axis=1)
            top_ids = np.take_along_axis(unsorted_top, sorted_idx, axis=1)

            predictions[batch_users] = top_ids

        return predictions

    def save_checkpoint(self, path: str):
        directory = os.path.dirname(path)
        if directory: os.makedirs(directory, exist_ok=True)

        config = {
            'rank': self.rank,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'n_epochs': self.n_epochs,
            'n_epochs_folding': self.n_epochs_folding,
            'lambda_geo': self.lambda_geo,
            'eta': self.eta,
            'resample_every_factor': self.resample_every_factor
        }
        np.savez_compressed(path, P=self.P, Q=self.Q, seen_users=self.seen_users, config=config)
        if self.verbose: print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        path_npz = path if path.endswith('.npz') else path + ".npz"
        if not os.path.exists(path_npz): raise FileNotFoundError(f"Checkpoint not found at {path_npz}")

        with np.load(path_npz, allow_pickle=True) as data:
            self.P = data['P'].astype(np.float32)
            self.Q = data['Q'].astype(np.float32)
            self.seen_users = data['seen_users']

            config = data.get('config', {}).item()
            self.rank = int(config.get('rank', self.rank))
            self.learning_rate = float(config.get('learning_rate', self.learning_rate))
            self.regularization = float(config.get('regularization', self.regularization))
            self.n_epochs = int(config.get('n_epochs', self.n_epochs))
            self.n_epochs_folding = int(config.get('n_epochs_folding', self.n_epochs_folding))
            self.lambda_geo = int(config.get('lambda_geo', 256))
            self.eta = int(config.get('eta', 1))
            self.resample_every_factor = float(config.get('resample_every_factor', 1.0))

        if self.verbose: print(f"Checkpoint loaded. Rank: {self.rank}")

import numpy as np
from numba import njit, prange
from tqdm.auto import tqdm
import os
from src.base import BaseModel
from src.metrics import NDCGMetric


@njit(cache=True, fastmath=True)
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1.0 + z)


@njit(cache=True, fastmath=True)
def sample_geometric_fast(lambda_geo, n_items):
    log_rho = -1.0 / lambda_geo
    u = np.random.rand()
    if u < 1e-10: 
        u = 1e-10
    val = np.log(u) / log_rho
    r = int(val) + 1
    if r > n_items: 
        return n_items
    return r


@njit(cache=True, fastmath=True)
def sample_factor_weighted(abs_weights):
    cdf = np.cumsum(abs_weights)
    total = cdf[-1]
    if total < 1e-15: return np.random.randint(len(abs_weights))
    u = np.random.rand() * total
    idx = np.searchsorted(cdf, u)
    if idx >= len(abs_weights): 
        idx = len(abs_weights) - 1
    return idx


@njit(parallel=True, cache=True)
def update_rankings_parallel(Q, rank, n_items):
    r_inv_asc = np.empty((rank, n_items), dtype=np.int64)
    for f in prange(rank):
        col = Q[:, f].copy()
        r_inv_asc[f] = np.argsort(col)
    return r_inv_asc


@njit(cache=True, fastmath=True)
def update_user_vector_adaptive(p_u, Q_pos, Q, r_inv_asc, rank, n_items, lr, reg, lambda_geo, n_negatives):
    n_pos = len(Q_pos)
    indices = np.arange(n_pos)
    np.random.shuffle(indices)

    for i in range(n_pos):
        pos_idx = indices[i]
        i_pos_vec = Q_pos[pos_idx]
        
        pred_pos = np.dot(p_u, i_pos_vec)
        sig_pos = sigmoid(pred_pos)
        err_pos = 1.0 - sig_pos
        
        decay = 1.0 - 2.0 * lr * reg
        p_u[:] = decay * p_u + lr * (err_pos * i_pos_vec)
        
        for _ in range(n_negatives):
            abs_pu = np.abs(p_u)
            f_sel = sample_factor_weighted(abs_pu)
            r = sample_geometric_fast(lambda_geo, n_items)

            if p_u[f_sel] >= 0:
                j_neg_idx = r_inv_asc[f_sel, n_items - r]
            else:
                j_neg_idx = r_inv_asc[f_sel, r - 1]
                
            q_neg_vec = Q[j_neg_idx]
            
            pred_neg = np.dot(p_u, q_neg_vec)
            sig_neg = sigmoid(pred_neg)
            err_neg = 0.0 - sig_neg
    
            p_u[:] = p_u + lr * (err_neg * q_neg_vec)

    return p_u


@njit(parallel=True, cache=True, fastmath=True)
def process_batch_with_folding_in(
        user_ids, history, P, Q, r_inv_asc, rank, lr, reg,
        max_epochs_folding, min_folding_epochs, epsilon,
        is_seen_mask, lambda_geo, n_negatives, patience
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
                patience_cnt = 0
                p_u_old = np.empty(rank, dtype=np.float32)

                for epoch in range(max_epochs_folding):
                    check_convergence = (epoch >= min_folding_epochs - 1)
                    if check_convergence: p_u_old[:] = p_u[:]

                    p_u = update_user_vector_adaptive(
                        p_u, Q_pos, Q, r_inv_asc,
                        rank, n_items, lr, reg, lambda_geo, n_negatives
                    )

                    if check_convergence:
                        diff = 0.0
                        for k in range(rank):
                            d = p_u[k] - p_u_old[k]
                            diff += d*d
                        if diff < epsilon * epsilon:
                            patience_cnt += 1
                            if patience_cnt >= patience: break
                        else:
                            patience_cnt = 0
            P_batch[b] = p_u
    return P_batch


@njit(cache=True, fastmath=True)
def adaptive_sgd_epoch(P, Q, pos_users, pos_items, n_pos, n_steps, 
                       r_inv_asc, lambda_geo, lr, reg, rank, n_items, n_negatives):
    loss = 0.0
    
    for _ in range(n_steps):
        # positive
        idx = np.random.randint(n_pos)
        u = pos_users[idx]
        i_pos = pos_items[idx]
        pu = P[u]
        
        q_pos_vec = Q[i_pos]
        pred_pos = np.dot(pu, q_pos_vec)
        sig_pos = sigmoid(pred_pos)
        err_pos = 1.0 - sig_pos
        
        decay = 1.0 - 2.0 * lr * reg
        
        grad_P_accum = err_pos * q_pos_vec
        Q[i_pos] = (1.0 - lr * reg) * q_pos_vec + lr * (err_pos * pu)
        
        loss += np.logaddexp(0, -pred_pos)

        # negatives
        for _ in range(n_negatives):
            abs_pu = np.abs(pu)
            f_sel = sample_factor_weighted(abs_pu)
            r = sample_geometric_fast(lambda_geo, n_items)

            if pu[f_sel] >= 0:
                j_neg = r_inv_asc[f_sel, n_items - r]
            else:
                j_neg = r_inv_asc[f_sel, r - 1]
            
            q_neg_vec = Q[j_neg]
            pred_neg = np.dot(pu, q_neg_vec)
            sig_neg = sigmoid(pred_neg)
            err_neg = 0.0 - sig_neg
            
            grad_P_accum += err_neg * q_neg_vec
            Q[j_neg] = (1.0 - lr * reg) * q_neg_vec + lr * (err_neg * pu)
            
            loss += np.logaddexp(0, pred_neg)
            
        pu[:] = decay * pu + lr * grad_P_accum

    return loss / (n_steps * (1 + n_negatives)) if n_steps > 0 else 0.0


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
            n_negatives: int = 1,
            resample_every_factor: float = 1.0,
            gamma: float = 1.0
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
        self.n_negatives = int(n_negatives)
        self.resample_every_factor = float(resample_every_factor)
        self.gamma = float(gamma)
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

        lambda_geo_val = int(self.lambda_geo)
        initial_lr = float(self.learning_rate)
        reg_val = float(self.regularization)
        rank_val = int(self.rank)
        n_negatives_val = int(self.n_negatives)
        
        scale = np.sqrt(1.0 / rank_val)
        self.P = np.random.normal(0, scale, size=(n_users, rank_val)).astype(np.float32)
        self.Q = np.random.normal(0, scale, size=(n_items, rank_val)).astype(np.float32)
        
        r_inv_asc = update_rankings_parallel(self.Q, rank_val, n_items)
        
        coo = train_dataset.get_coo_array()
        pos_users = coo.row.astype(np.int32)
        pos_items = coo.col.astype(np.int32)
        n_pos = len(pos_users)
        
        self.seen_users = np.zeros(n_users, dtype=bool)
        self.seen_users[np.unique(pos_users)] = True
        
        resample_steps = max(1, int(n_items * np.log(n_items + 1) * self.resample_every_factor))
        
        if self.verbose:
            print(f"Training {self.name}. Steps/epoch: {n_pos}, Resample every: {resample_steps} steps.")

        best_metric = -np.inf
        patience_cnt = 0
        total_steps = 0

        for epoch in tqdm(range(self.n_epochs), desc=f"Training {self.name}", disable=not self.verbose):
            lr_val = initial_lr * (self.gamma ** epoch)
            steps_remaining = n_pos
            epoch_loss = 0.0
            
            while steps_remaining > 0:
                steps_to_resample = resample_steps - (total_steps % resample_steps)
                current_chunk = min(steps_remaining, steps_to_resample)
                
                if total_steps > 0 and total_steps % resample_steps == 0:
                     r_inv_asc = update_rankings_parallel(self.Q, rank_val, n_items)

                chunk_loss = adaptive_sgd_epoch(
                    self.P, self.Q, pos_users, pos_items, n_pos, current_chunk,
                    r_inv_asc, lambda_geo_val,
                    lr_val, reg_val, rank_val, n_items, n_negatives_val
                )
                
                epoch_loss += chunk_loss * current_chunk
                steps_remaining -= current_chunk
                total_steps += current_chunk

            avg_loss = epoch_loss / n_pos

            if (epoch + 1) % self.n_valid == 0 and val_dataset is not None:
                predictions = self.predict(val_dataset, 10)
                holdout_users = val_dataset.get_holdout_users()
                metric = NDCGMetric(10)
                val_metric = metric(predictions[holdout_users, :], val_dataset.get_holdout_array()[holdout_users])

                if self.verbose:
                    print(f"Epoch {epoch + 1} - BCE Loss: {avg_loss:.4f} | val NDCG@10: {val_metric:.4f}")

                if val_metric < best_metric:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        if self.verbose: print(f'Early stopping on epoch {epoch + 1}')
                        break
                else:
                    patience_cnt = 0
                    best_metric = val_metric
                    self.P_best = self.P.copy()
                    self.Q_best = self.Q.copy()
                    self.trained_epochs = epoch + 1
        
        if val_dataset is not None:
            self.P = self.P_best
            self.Q = self.Q_best

    def predict(self, dataset, top_n: int) -> np.ndarray:
        predictions = np.zeros((dataset.n_users, top_n), dtype=np.int64)
        dataloader = dataset.get_dataloader(batch_size=1024, shuffle=False)
        
        P_contiguous = np.ascontiguousarray(self.P)
        Q_contiguous = np.ascontiguousarray(self.Q)
        r_inv_asc = update_rankings_parallel(Q_contiguous, self.rank, self.Q.shape[0])
        seen_users_contiguous = np.ascontiguousarray(self.seen_users)
        
        n_negatives_val = int(self.n_negatives)

        for batch in tqdm(dataloader, desc="Predicting", disable=not self.verbose):
            batch_users = batch['user_id'].numpy().astype(np.int32)
            history = batch['history'].numpy().astype(np.int32)
            folding_lr = self.learning_rate * 0.1
            
            P_batch = process_batch_with_folding_in(
                batch_users, history, P_contiguous, Q_contiguous, r_inv_asc,
                self.rank, folding_lr, self.regularization,
                self.n_epochs_folding, self.min_folding_epochs, self.folding_epsilon,
                seen_users_contiguous, self.lambda_geo, n_negatives_val, self.patience
            )

            scores = P_batch @ Q_contiguous.T
            rows = np.repeat(np.arange(len(batch_users)), history.shape[1])
            cols = history.ravel()
            valid_mask = cols != -1
            scores[rows[valid_mask], cols[valid_mask]] = -np.inf

            unsorted_top = np.argpartition(-scores, top_n, axis=1)[:, :top_n]
            top_scores = np.take_along_axis(scores, unsorted_top, axis=1)
            sorted_idx = np.argsort(-top_scores, axis=1)
            predictions[batch_users] = np.take_along_axis(unsorted_top, sorted_idx, axis=1)

        return predictions
    
    def save_checkpoint(self, path: str):
        directory = os.path.dirname(path)
        if directory: 
            os.makedirs(directory, exist_ok=True)


        config = {
        'rank': self.rank,
        'learning_rate': self.learning_rate,
        'regularization': self.regularization,
        'n_epochs': self.n_epochs,
        'n_epochs_folding': self.n_epochs_folding,
        'lambda_geo': self.lambda_geo,
        'n_negatives': self.n_negatives,
        'resample_every_factor': self.resample_every_factor,
        'gamma': self.gamma
        }
        np.savez_compressed(path, P=self.P, Q=self.Q, seen_users=self.seen_users, config=config)
        if self.verbose: 
            print(f"Checkpoint saved to {path}")


    def load_checkpoint(self, path: str):
        path_npz = path if path.endswith('.npz') else path + ".npz"
        if not os.path.exists(path_npz): 
            raise FileNotFoundError(f"Checkpoint not found at {path_npz}")


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
        self.resample_every_factor = float(config.get('resample_every_factor', 1.0))
        self.gamma = float(config.get('gamma', 1.0))
        self.n_negatives = int(config.get('n_negatives', 0))


        if self.verbose: 
            print(f"Checkpoint loaded. Rank: {self.rank}")


    def suggest_additional_params(self) -> dict:
        epochs = self.trained_epochs if self.trained_epochs else self.n_epochs
        return {"n_epochs": int(epochs)}
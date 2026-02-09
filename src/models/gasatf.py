import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange
from numba.typed import List as NumbaList
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.linalg import solve_triangular
from abc import ABC
import time
from sklearn.utils.extmath import randomized_svd
from src.base import BaseModel
from src.metrics import NDCGMetric

@njit(parallel=True, nogil=True)
def _tensordot2_par(idx, val, u, v, mode1, mode2, unqs, inds, res):
    r1 = u.shape[1]
    r2 = v.shape[1]
    n = len(unqs)

    for s in prange(n):
        i0 = unqs[s]
        ul = inds[s]
        for pos in ul:
            i1 = idx[pos, mode1]
            i2 = idx[pos, mode2]
            vp = val[pos]
            for j1 in range(r1):
                tmp = vp * u[i1, j1]
                for j2 in range(r2):
                    res[i0, j1, j2] += tmp * v[i2, j2]

def tensordot2_par(idx, val, shape, u, v, contraction_modes, unqs, inds, dtype=np.float64):
    """
    Tensor dot product along two modes (parallel version).
    """
    mode1, _ = contraction_modes[0]
    mode2, _ = contraction_modes[1]
    
    # Determine target mode (the one not in contraction)
    all_modes = {0, 1, 2}
    contracted = {mode1, mode2}
    target_mode = (all_modes - contracted).pop()
    
    r1 = u.shape[1]
    r2 = v.shape[1]
    dim_target = shape[target_mode]
    
    res = np.zeros((dim_target, r1, r2), dtype=dtype)
    _tensordot2_par(idx, val, u, v, mode1, mode2, unqs, inds, res)
    
    return res

@njit
def fill_missing_sorted(arr, inds, size):
    filler = inds[0][0:0]
    arr_filled = np.empty(size, dtype=arr.dtype)
    inds_filled = NumbaList()
    pos = 0
    for i in range(size):
        if i == arr[pos]:
            inds_filled.append(inds[pos])
            pos += 1
        else:
            inds_filled.append(filler)
        arr_filled[i] = i
    return arr_filled, inds_filled

def arrange_index(array, typed=True, size=None):
    unqs, unq_inv, unq_cnt = np.unique(array, return_inverse=True, return_counts=True)
    inds = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))

    if typed:
        inds_typed = NumbaList()
        for ind in inds:
            inds_typed.append(ind)
        inds = inds_typed

    if (size is not None) and (len(unqs) < size):
        unqs, inds = fill_missing_sorted(unqs, inds, size)
        unqs = np.array(unqs)
    unqs = np.asarray(unqs, dtype=np.int64)
    return unqs, inds

def arrange_indices(idx, mode_mask=None, shape=None):
    n = idx.shape[1]
    res = [[]] * n
    if mode_mask is None:
        mode_mask = [True] * n
    sizes = list(shape) if shape else [None] * n

    n_active_modes = sum(mode_mask)
    if n_active_modes == 0:
        return res

    if n_active_modes == 1:
        mode = mode_mask.index(True)
        res[mode] = arrange_index(idx[:, mode], size=sizes[mode])
        return res

    with ThreadPoolExecutor(max_workers=n_active_modes) as executor:
        arranged_futures = {
            executor.submit(arrange_index, idx[:, mode], size=sizes[mode]): mode
            for mode in range(n) if mode_mask[mode]
        }
        for future in as_completed(arranged_futures):
            mode = arranged_futures[future]
            res[mode] = future.result()
    return res


class TensorBasedModel(ABC):
    
    def __init__(self):
        self.V = None
        self.W = None
        self.A = None
        self.d = None
        self.inv_d = None
        self.attention_vector = None
    
    @staticmethod
    def _shift_vector_last(tmp: np.ndarray) -> np.ndarray:
        m = np.zeros_like(tmp)
        m[1:] = tmp[:-1]
        return m

    def _compute_attention_vector(self) -> np.ndarray:       
        A, W = self.A, self.W
        is_lower = np.allclose(A, np.tril(A))
        lower_flag_for_AT = not is_lower 
        W_hat = solve_triangular(A.T, W, lower=lower_flag_for_AT, check_finite=False)
        w_hat_last = W_hat[-1]
        
        tmp = (A @ W) @ w_hat_last
        self.attention_vector = self._shift_vector_last(tmp)
        return self.attention_vector

class AttentionBuilder:

    @staticmethod
    def lin_decay(decay_factor, n):
        n = np.asarray(n, dtype=np.float64)
        return np.power(n, -float(decay_factor))

    @staticmethod
    def log_decay(decay_factor, n):
        n = np.asarray(n, dtype=np.float64)
        return 1.0 / (1.0 + float(decay_factor) * np.log(n))
    
    def build_weights(self, decay_factor, positions: np.ndarray, decay_mode: str = 'lin') -> np.ndarray:
        if decay_mode == 'log':
            return self.log_decay(decay_factor, positions)
        return self.lin_decay(decay_factor, positions)

    def build_attention_matrix(
        self,
        size: int,
        decay_factor: float,
        span: int = 0,
        triangle: str = 'lower',
        stochastic_axis: int | None = None,
        dtype=np.float64,
        decay_mode: str = 'lin'
    ) -> np.ndarray:
        if size <= 0:
            return None

        if span > 0:
            L = min(int(span), size)
        else:
            L = size

        positions = np.arange(1, L + 1, dtype=np.float64)
        w = self.build_weights(decay_factor, positions, decay_mode=decay_mode)

        i = np.arange(size)[:, None]
        j = np.arange(size)[None, :]
        lag = i - j
        
        A = np.zeros((size, size), dtype=dtype)

        if triangle == 'lower':
            mask = (lag >= 0) & (lag < L)
            A[mask] = w[lag[mask].astype(np.int64)]
        else:
            ulag = -lag
            mask = (ulag >= 0) & (ulag < L)
            A[mask] = w[ulag[mask].astype(np.int64)]

        if stochastic_axis is not None:
            s = A.sum(axis=stochastic_axis, keepdims=True)
            s[s == 0] = 1.0
            A = A / s

        return A

def generate_sequential_tensor(dataset, 
                               max_positions: int = 100, 
                               dtype=np.float64):
    user_idx_all = []
    item_idx_all = []
    position_idx_all = []
    
    dataloader = dataset.get_dataloader(batch_size=2048, shuffle=False)

    for batch in dataloader:
        history = batch["history"].numpy().astype(np.int64)
        batch_users = batch['user_id'].numpy().astype(np.int64)
        n_users_batch, max_len = history.shape

        raw_mask = history != -1
        if not raw_mask.any():
            continue

        lengths = raw_mask.sum(axis=1)

        pos_grid = np.arange(max_len, dtype=np.int64)[None, :]
        dist_from_end = (lengths[:, None] - 1) - pos_grid
        pos_aligned = (max_positions - 1) - dist_from_end

        valid_mask = raw_mask & (pos_aligned >= 0)

        if not valid_mask.any():
            continue

        pos_idx = pos_aligned[valid_mask].ravel()
        item_idx = history[valid_mask].ravel()
        
        user_idx_matrix = np.broadcast_to(batch_users[:, None], history.shape)
        user_idx = user_idx_matrix[valid_mask].ravel()

        user_idx_all.append(user_idx)
        item_idx_all.append(item_idx)
        position_idx_all.append(pos_idx)

    if not user_idx_all:
         raise ValueError("No valid interactions found in dataset.")

    user_idx = np.concatenate(user_idx_all)
    item_idx = np.concatenate(item_idx_all)
    position_idx = np.concatenate(position_idx_all)

    n_users = dataset.n_users
    n_items = dataset.n_items
    n_positions = max_positions

    idx = np.column_stack((user_idx, item_idx, position_idx))
    val = np.ones(len(idx), dtype=dtype) 
    shape = (n_users, n_items, n_positions)
    
    return idx, val, shape


class GASATF(BaseModel, TensorBasedModel, AttentionBuilder):

    def __init__(self,
                 rank_u: int,
                 rank_v: int,
                 rank_w: int,
                 triangle: str,
                 decay_mode: str,
                 decay: float,
                 max_positions: int,
                 growth_tol: float,
                 iters: int,
                 tries: int,
                 val_top_n: int,
                 verbose: bool = False,
                 seed: int | None = None,
                 name: str = "GA-SATF",
                 dtype=np.float32,
                 need_downvote: bool = True,
                 scaling_factor: float = 0.0,
                 rescaled: bool = False,
                 **kwargs
                 ):
        BaseModel.__init__(self, name)
        TensorBasedModel.__init__(self)

        self.rank_u = rank_u
        self.rank_v = rank_v
        self.rank_w = rank_w
        
        self.dtype = np.dtype(dtype)
        self.verbose = verbose
        self.seed = seed
        self.rng = np.random if seed is None else np.random.RandomState(seed)
        
        self.triangle = triangle
        self.decay_mode = decay_mode
        self.decay = decay
        self.max_positions = max_positions
        self.growth_tol = growth_tol
        self.iters = iters
        self.tries = tries
        self.val_top_n = val_top_n
        
        self.scaling_factor = scaling_factor
        self.rescaled = rescaled
        self.downvote_seen_items = need_downvote

        self.idx = None
        self.val = None
        self.shape = None
        self.index_data = None
        self.ttm = None

    def core_norm(self, U, V, W):
        core_norm = np.linalg.norm(self.compute_core(U, V, W))**2
        return core_norm

    def compute_core(self, U, V, W):
        ru, rv, rw = U.shape[1], V.shape[1], W.shape[1]
        
        V_weighted = V * self.d[:, None]
        
        M = self.ttm[2](
            self.idx, self.val, self.shape,
            U, V_weighted,
            [(0, 0), (1, 0)],
            *self.index_data[2],
        )

        M_mat = M.reshape(W.shape[0], ru * rv, order='C')
        AW = self.A @ W
        M_mat = AW.T @ M_mat

        core = M_mat.reshape(rw, ru, rv, order='C').transpose(1, 2, 0)
        print(f'compute core: done')
        return core

    def fit_rank(self, core_shape, growth_tol=1e-6, iters=100, tries=5, val_callback=None):
        ranks = core_shape
        dims = self.shape

        factors = {
            0: self.rng.randn(dims[0], ranks[0]).astype(self.dtype, copy=False),  # U
            1: self.rng.randn(dims[1], ranks[1]).astype(self.dtype, copy=False),  # V
            2: self.rng.randn(dims[2], ranks[2]).astype(self.dtype, copy=False)   # W
        }

        for mode in range(3):
            factors[mode] = self._qr_basis(factors[mode], ranks[mode])

        best_factors = {mode: factors[mode].copy() for mode in range(3)}
        best_metric = -np.inf
        tries_left = tries
        best_sweep = 0

        if self.rescaled and self.d is not None:
            d_vec = self.d[:, None]
        else:
            d_vec = np.ones((dims[1], 1), dtype=self.dtype)

        for sweep in range(1, iters + 1):
            if self.verbose:
                print(f"Sweep {sweep}/{iters}")

            AW = self.A @ factors[2]

            matrix = self.ttm[0](
                self.idx, self.val, self.shape,
                factors[1] * d_vec, AW,
                [(1, 0), (2, 0)],
                *self.index_data[0],
                dtype=self.dtype
            ).reshape(dims[0], ranks[1] * ranks[2], order='C')
            factors[0] = self._svd_basis(matrix, ranks[0])
            
            matrix = self.ttm[1](
                self.idx, self.val, self.shape,
                factors[0], AW, 
                [(0, 0), (2, 0)],
                *self.index_data[1],
                dtype=self.dtype
            ).reshape(dims[1], ranks[0] * ranks[2], order='C')
            factors[1] = self._svd_basis(d_vec * matrix, ranks[1])

            matrix = self.ttm[2](
                self.idx, self.val, self.shape,
                factors[0], factors[1] * d_vec,
                [(0, 0), (1, 0)],
                *self.index_data[2],
                dtype=self.dtype
            ).reshape(dims[2], ranks[0] * ranks[1], order='C')
            
            matrix = self.A.T @ matrix
            factors[2] = self._svd_basis(matrix, ranks[2])

            if val_callback is not None:
                metric = val_callback(factors)
                if self.verbose:
                    print(f'Metric: {metric:.6f}, best: {best_metric:.6f}')
                
                if best_metric != -np.inf:
                    growth = (metric - best_metric) / max(abs(best_metric), 1e-12)
                    if growth < growth_tol:
                        tries_left -= 1
                        if tries_left == 0:
                            if self.verbose:
                                print(f"Converged after {sweep} sweeps")
                            break
                    else:
                        tries_left = tries
                
                if metric > best_metric:
                    best_metric = metric
                    best_sweep = sweep
                    for m in range(3):
                        best_factors[m] = factors[m].copy()
        
        if val_callback is None:
            best_factors = factors
            best_sweep = sweep

        return best_factors, best_sweep
    
    def validate_rank(self):
        if self.rank_u > self.rank_v * self.rank_w:
            return False
        if self.rank_v > self.rank_u * self.rank_w:
            return False
        if self.rank_w > self.rank_u * self.rank_v:
            return False
        return True
    
    def fit(self, train_dataset, val_dataset=None):
        if not self.validate_rank():
            self.V = None
            self.W = None
            return
        
        self.idx, self.val, self.shape = generate_sequential_tensor(
            train_dataset,
            max_positions=self.max_positions,
            dtype=self.dtype
        )   
        
        if self.rescaled:
            item_pops = np.bincount(self.idx[:, 1].astype(np.int64), minlength=self.shape[1]).astype(self.dtype)
            item_pops[item_pops == 0] = 1
            self.d = np.power(item_pops, (self.scaling_factor - 1.0) / 2.0).astype(self.dtype)
            self.inv_d = (1.0 / self.d).astype(self.dtype)
        else:
            self.d = np.ones(self.shape[1], dtype=self.dtype)
            self.inv_d = np.ones(self.shape[1], dtype=self.dtype)

        self.A = self.build_attention_matrix(
            self.max_positions,
            decay_factor=self.decay,
            triangle=self.triangle,
            dtype=self.dtype,
            decay_mode=self.decay_mode
        )
        
        parallel_ttm = [True] * len(self.shape)
        self.index_data = arrange_indices(self.idx, mode_mask=parallel_ttm, shape=self.shape)
        self.ttm = [tensordot2_par for _ in self.shape]
        
        core_shape = (self.rank_u, self.rank_v, self.rank_w)
        
        if val_dataset is not None:

            def val_callback(factors):
                self.V = factors[1]
                self.W = factors[2]
                metric = NDCGMetric(self.val_top_n)
                holdout_users = val_dataset.get_holdout_users()

                predictions = self.predict(val_dataset, self.val_top_n)
                return metric(predictions[holdout_users, :], 
                              val_dataset.get_holdout_array()[holdout_users])
        else:
            val_callback = None
        
        best_factors, sweeps = self.fit_rank(
            core_shape, 
            growth_tol=self.growth_tol, 
            iters=self.iters, 
            tries=self.tries,
            val_callback=val_callback
        )
        self.n_iters = sweeps
        self.U = best_factors[0]
        self.V = best_factors[1]
        self.W = best_factors[2]
        
        return self
    
    def predict(self, dataset, top_n: int) -> np.ndarray:
        if self.V is None or self.W is None:
            return np.tile(np.arange(top_n, dtype=np.int32), (dataset.n_users, 1))
        
        num_users = dataset.n_users
        num_items = dataset.n_items
        
        att_vector = self._compute_attention_vector()
        
        recoms_all = np.zeros((num_users, top_n), dtype=np.int32)
        dataloader = dataset.get_dataloader(batch_size=8192, shuffle=False)
        
        for batch in dataloader:
            history = batch["history"].numpy().astype(np.int64)
            batch_users = batch['user_id'].numpy().astype(np.int64)
            n_users_batch, max_len = history.shape
            
            raw_mask = history != -1
            if not raw_mask.any():
                continue
            
            lengths = raw_mask.sum(axis=1)
            
            pos_grid = np.arange(max_len, dtype=np.int64)[None, :]
            dist_from_end = (lengths[:, None] - 1) - pos_grid
            pos_aligned = (self.max_positions - 1) - dist_from_end
        
            valid_mask = raw_mask & (pos_aligned >= 0)
            
            if not valid_mask.any():
                continue

            positions = pos_aligned[valid_mask].ravel()
            item_idx = history[valid_mask].ravel()
            
            valid_lengths = valid_mask.sum(axis=1)
            row_idx = np.repeat(np.arange(n_users_batch), valid_lengths)
            
            weights = att_vector[positions]
            
            if self.rescaled and self.d is not None:
                weights *= self.d[item_idx]
            
            M_batch = csr_matrix(
                (weights, (row_idx, item_idx)),
                shape=(n_users_batch, num_items)
            )
            
            scores_batch = (M_batch @ self.V) @ self.V.T
            
            if self.rescaled and self.inv_d is not None:
                scores_batch *= self.inv_d[None, :]

            if self.downvote_seen_items:
                min_val = scores_batch.min() - self.dtype.type(1)
                if raw_mask.any():
                    seen_items = history[raw_mask].ravel()
                    seen_rows = np.repeat(np.arange(n_users_batch), lengths)
                    scores_batch[seen_rows, seen_items] = min_val
            
            current_top_n = min(top_n, scores_batch.shape[1])
            top_idx = np.argpartition(-scores_batch, kth=current_top_n - 1, axis=1)[:, :current_top_n]
            
            row_arange = np.arange(top_idx.shape[0])[:, None]
            sorted_indices_in_top = np.argsort(-scores_batch[row_arange, top_idx], axis=1)
            top_sorted = top_idx[row_arange, sorted_indices_in_top]
            
            recoms_all[batch_users, :current_top_n] = top_sorted
        
        return recoms_all

    def save_checkpoint(self, path: str):
        pass
    
    def load_checkpoint(self, path: str):
        pass

    def suggest_additional_params(self) -> dict:
        if self.n_iters is None:
            return {}
        return {"iters": int(self.n_iters)}

    def _svd_basis(self, mat, rank):
        n_iter = 2
        oversample = 12
        h, w = mat.shape
        k = rank + oversample
        
        if k >= min(h, w):
            U, _, _ = np.linalg.svd(mat, full_matrices=False)
            return U[:, :rank].astype(self.dtype, copy=False)
        
        if h * w <= 10**9:
             U, _, _ = randomized_svd(mat, n_components=rank, random_state=self.rng)
             return U.astype(self.dtype, copy=False)

        Omega = self.rng.randn(w, k).astype(self.dtype, copy=False)
        Y = mat @ Omega
        for _ in range(n_iter):
            Y = mat @ (mat.T @ Y)
        Q, _ = np.linalg.qr(Y) 
        B = Q.T @ mat
        U_hat, _, _ = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat[:, :rank]
        return U.astype(self.dtype, copy=False)

    def _qr_basis(self, mat, rank):
        Q, _ = np.linalg.qr(mat, mode='reduced')
        return Q.astype(self.dtype, copy=False)

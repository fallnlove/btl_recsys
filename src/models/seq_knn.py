from src.base import BaseModel
import numpy as np
import json
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity


def generate_sequential_matrix(
    dataset,
    # binary_feedback: bool = False,  at the time of coding, the dataloader does not load ratings, only binary feedback on interactions
    weighting_mode: str = "reciprocal",
    position_decay: float = 1.0,
    session_length_normalization: bool = False
) -> csr_matrix:

    user_idx_all = []
    item_idx_all = []
    weights_all = []
    next_user_idx = 0
    dataloader = dataset.get_dataloader(batch_size=1024, shuffle=False)

    for batch in dataloader:
        history = batch["history"].numpy().astype(np.int64)
        n_users_batch, max_len = history.shape

        user_idx_batch = np.arange(
            next_user_idx, next_user_idx + n_users_batch, dtype=np.int64)
        next_user_idx += n_users_batch

        mask = history != -1
        if not mask.any():
            continue

        lengths = mask.sum(axis=1).astype(np.int64)

        pos_oldest = np.arange(max_len, dtype=np.int64)[None, :]
        pos_from_newest = (lengths[:, None] - 1) - pos_oldest
        pos_from_newest = pos_from_newest[mask]

        item_idx = history[mask].ravel()
        user_idx = np.repeat(user_idx_batch, lengths)

        if weighting_mode == "none":
            w_pos = np.ones_like(pos_from_newest, dtype="float64")
        elif weighting_mode == "reciprocal":
            w_pos = 1.0 / (pos_from_newest + 1)
        elif weighting_mode == "position_exp":
            w_pos = np.exp(-position_decay * pos_from_newest)
        else:
            raise ValueError(
                f"Unknown weighting_mode: {weighting_mode}. "
                "Available modes: 'reciprocal', 'position_exp', 'none'."
            )

        w_fb = np.ones_like(w_pos, dtype="float64")

        weights = w_pos * w_fb

        if session_length_normalization:
            lengths_float = lengths.astype("float64")
            w_len_user = np.divide(1.0, lengths_float, out=np.zeros_like(
                lengths_float), where=lengths_float != 0)
            w_len = np.repeat(w_len_user, lengths)
            weights = weights * w_len

        user_idx_all.append(user_idx)
        item_idx_all.append(item_idx)
        weights_all.append(weights)

    user_idx = np.concatenate(user_idx_all)
    item_idx = np.concatenate(item_idx_all)
    weights = np.concatenate(weights_all).astype("float64")

    n_users = next_user_idx
    n_items = dataset.n_items

    mat = csr_matrix(
        (weights, (user_idx, item_idx)),
        shape=(n_users, n_items),
        dtype="float64",
    )
    return mat


def jaccard_similarity(A, B):
    assert A.shape[1] == B.shape[1]
    A_bin = A.astype('bool').astype('int')
    B_bin = B.astype('bool').astype('int')
    numerator = A_bin @ B_bin.T
    denominator = A_bin.sum(axis=1) + B_bin.sum(axis=1).T - A_bin @ B_bin.T
    similarity = csr_matrix(numerator / denominator)
    return similarity


def compute_similarity(type, m1, m2):
    if type == 'jaccard':
        similarity = jaccard_similarity(m1, m2)
    elif type == 'cosine':
        similarity = cosine_similarity(m1, m2, dense_output=False)
    else:
        raise ValueError(
            f"Unknown similarity type: {type}. "
            "Available types: 'cosine', 'jaccard'."
        )
    return similarity


def truncate_similarity(similarity, k):
    similarity = similarity.tocsr()
    inds = similarity.indices
    ptrs = similarity.indptr
    data = similarity.data
    new_ptrs = [0]
    new_inds = []
    new_data = []
    for i in range(len(ptrs)-1):
        start, stop = ptrs[i], ptrs[i+1]
        if start < stop:
            data_ = data[start:stop]
            topk = min(len(data_), k)
            idx = np.argpartition(data_, -topk)[-topk:]
            new_data.append(data_[idx])
            new_inds.append(inds[idx+start])
            new_ptrs.append(new_ptrs[-1]+len(idx))
        else:
            new_ptrs.append(new_ptrs[-1])
    new_data = np.concatenate(new_data)
    new_inds = np.concatenate(new_inds)
    truncated = csr_matrix(
        (new_data, new_inds, new_ptrs),
        shape=similarity.shape
    )
    return truncated


class SeqKNN(BaseModel):
    def __init__(self,
                 name: str = "SeqKNN",
                 n_neighbors: int = 500,
                 similarity_type: str = "cosine",
                 # at the time of coding, the dataloader does not load ratings, only binary feedback on interactions
                 binary_feedback: bool = True,
                 downvote_seen_items: bool = True,
                 weighting_mode: str = "reciprocal",
                 position_decay: float = 1.0,
                 session_length_normalization: bool = False,
                 ):
        super().__init__(name)
        self.n_neighbors = n_neighbors
        self.similarity_type = similarity_type
        self.binary_feedback = binary_feedback
        self.downvote_seen_items = downvote_seen_items
        self.weighting_mode = weighting_mode
        self.position_decay = position_decay
        self.session_length_normalization = session_length_normalization
        self.train_interactions = None

    def fit(self, train_dataset, val_dataset=None):
        train_interactions = generate_sequential_matrix(
            train_dataset,
            # binary_feedback = self.binary_feedback,  at the time of coding, the dataloader does not load ratings, only binary feedback on interactions
            weighting_mode=self.weighting_mode,
            position_decay=self.position_decay,
            session_length_normalization=self.session_length_normalization
        )
        self.train_interactions = train_interactions
        return self

    def predict(self, dataset, top_n: int) -> np.ndarray:
        test_interactions = generate_sequential_matrix(
            dataset,
            # binary_feedback = self.binary_feedback,  at the time of coding, the dataloader does not load ratings, only binary feedback on interactions
            weighting_mode=self.weighting_mode,
            position_decay=self.position_decay,
            session_length_normalization=self.session_length_normalization
        )

        num_users = dataset.n_users
        recoms_all = np.zeros((num_users, top_n), dtype=np.int32)

        next_user_idx = 0
        dataloader = dataset.get_dataloader(batch_size=1024, shuffle=False)

        for batch in dataloader:
            history = batch["history"].numpy().astype(np.int64)
            batch_users = batch['user_id'].numpy().astype(np.int64)
            n_users_batch = len(batch_users)

            test_batch = test_interactions[next_user_idx: next_user_idx + n_users_batch]
            full_similarity = compute_similarity(
                self.similarity_type, test_batch, self.train_interactions)
            similarity = truncate_similarity(full_similarity, self.n_neighbors)

            scores_batch = similarity.dot(self.train_interactions).toarray()
            min_val = scores_batch.min() - 1
            next_user_idx += n_users_batch

            if self.downvote_seen_items:
                mask = history != -1
                if mask.any():
                    item_idx = history[mask].ravel()
                    lengths = mask.sum(axis=1).astype(np.int64)
                    row_idx = np.repeat(np.arange(n_users_batch), lengths)
                    scores_batch[row_idx, item_idx] = min_val

            current_top_n = min(top_n, scores_batch.shape[1])
            top_idx = np.argpartition(-scores_batch,
                                      kth=current_top_n - 1, axis=1)[:, :current_top_n]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_sorted = top_idx[row_idx,
                                 np.argsort(-scores_batch[row_idx, top_idx], axis=1)]
            recoms_all[batch_users, :current_top_n] = top_sorted

        return recoms_all

    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)

        meta = {
            "n_neighbors": self.n_neighbors,
            "similarity_type": self.similarity_type,
            "binary_feedback": self.binary_feedback,
            "downvote_seen_items": self.downvote_seen_items,
            "weighting_mode": self.weighting_mode,
            "position_decay": self.position_decay,
            "session_length_normalization": self.session_length_normalization,
            "train_interactions": "train_interactions.npz"
        }

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        save_npz(os.path.join(
            path, meta["train_interactions"]), self.train_interactions)

    def load_checkpoint(self, path: str):
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        self.n_neighbors = int(meta["n_neighbors"])
        self.similarity_type = str(meta["similarity_type"])
        self.binary_feedback = bool(meta["binary_feedback"])
        self.downvote_seen_items = bool(meta["downvote_seen_items"])
        self.weighting_mode = str(meta["weighting_mode"])
        self.position_decay = float(meta["position_decay"])
        self.session_length_normalization = bool(
            meta["session_length_normalization"])

        self.train_interactions = load_npz(
            os.path.join(path, meta["train_interactions"]))

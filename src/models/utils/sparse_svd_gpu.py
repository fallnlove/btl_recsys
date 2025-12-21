import numpy as np
import scipy.sparse as sp


def gpu_sparse_svd(
    A_scipy: sp.spmatrix,
    k: int,
    dtype=np.float32,
    return_numpy: bool = True,
):
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import svds

    if not sp.issparse(A_scipy):
        raise TypeError("A_scipy must be a scipy sparse matrix")

    A_gpu = csp.csr_matrix(A_scipy, dtype=dtype)

    U, S, Vt = svds(
        A_gpu,
        k=k,
        which="LM",
        return_singular_vectors=True,
    )

    idx = cp.argsort(-S)
    U = U[:, idx]
    S = S[idx]
    Vt = Vt[idx, :]

    if return_numpy:
        return (
            cp.asnumpy(U),
            cp.asnumpy(S),
            cp.asnumpy(Vt),
        )

    return U, S, Vt

import numpy as np
from scipy.optimize import minimize


def measure_loglikelihood(matrix, probs):
    """
    return loglikelihood
    see equation before (2) in https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    """
    m = matrix * np.log(probs[:, None] / (probs[:, None] + probs[None, :]))
    logprob = np.sum(m)
    return logprob


class ZermeloBradleyTerry:
    """
    Zermelo formula for solving standard Bradley-Terry
    see equation 3 in https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    """

    def __init__(self, n_iters=100):
        self.n_iters = n_iters

    def fit(self, W):
        """
        W : array of shape (n_players, n_players)
            W[i,j] = number of times i beat j
        """
        n = W.shape[0]
        probs = np.ones(n) / n
        logliks = []
        for _ in range(self.n_iters):
            denom = (W + W.T) / (probs[:, None] + probs[None, :])
            probs_new = np.sum(W, axis=1) / np.sum(denom, axis=1)
            probs = probs_new / np.prod(probs_new) ** (1 / n)
            logliks.append(measure_loglikelihood(W, probs))
        ranking = np.argsort(-probs) + 1
        return {'probs': probs, 'loglikelihoods_history': logliks, 'ranking': ranking}


class BayesianBradleyTerry:
    """
    Bayesian Bradley-Terry model for pairwise comparisons.
    https://www.jmlr.org/papers/volume24/22-0907/22-0907.pdf

    Parameters:
    -----------
    n_players : int
        Number of players/items to rank
    n_iter : int
        Number of MCMC iterations
    warmup : int
        Number of warmup iterations to discard
    random_seed : int
        Random seed for reproducibility
    """

    def __init__(self, n_players, n_iter=2000, warmup=1000, random_seed=42):
        self.n_players = n_players
        self.n_iter = n_iter
        self.warmup = warmup
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.log_liks = []

    def win_probability(self, beta_i, beta_j):
        """Probability that player i beats player j"""
        return 1 / (1 + np.exp(-(beta_i - beta_j)))

    def log_likelihood(self, beta, W, N):
        """Log-likelihood of the data given beta parameters"""
        log_lik = 0
        t = len(beta)

        for i in range(t):
            for j in range(i + 1, t):
                if N[i, j] > 0:
                    p_ij = self.win_probability(beta[i], beta[j])
                    # Binomial log-likelihood for W[i,j] wins out of N[i,j] trials
                    log_lik += W[i, j] * np.log(p_ij) + W[j, i] * np.log(1 - p_ij)
                    # Add log binomial coefficient (constant w.r.t. parameters)
                    # log_lik += sp.loggamma(N[i, j] + 1) - sp.loggamma(W[i, j] + 1) - sp.loggamma(W[j, i] + 1)
        self.log_liks.append(log_lik)
        return log_lik

    def log_prior(self, beta, sigma):
        """Log-prior for beta and sigma"""
        # Beta ~ Normal(0, sigma)
        beta_prior = -0.5 * np.sum(beta ** 2) / (sigma ** 2) - 0.5 * self.n_players * np.log(sigma)

        # Sigma ~ LogNormal(0, 0.5)
        sigma_prior = -0.5 * (np.log(sigma) / 0.5) ** 2 - np.log(sigma)  # np.log(0.5 * sigma * np.sqrt(2*np.pi))

        return beta_prior + sigma_prior

    def log_posterior(self, beta, sigma, W, N):
        """Log-posterior (unnormalized)"""
        return self.log_likelihood(beta, W, N) + self.log_prior(beta, sigma)

    def sample_posterior(self, W, N):
        """
        Sample from posterior using Metropolis-Hastings.

        Parameters:
        -----------
        W : array of shape (n_players, n_players)
            W[i,j] = number of times i beat j
        N : array of shape (n_players, n_players)
            N[i,j] = total number of matches between i and j

        Returns:
        --------
        beta_samples : array of shape (n_iter - warmup, n_players)
            MCMC samples of beta parameters
        sigma_samples : array of shape (n_iter - warmup,)
            MCMC samples of sigma parameter
        """
        t = self.n_players

        # Initialize parameters
        beta = np.random.randn(t) * 0.1
        sigma = 1.0

        # Normalize beta to sum to 0 (identifiability constraint)
        beta = beta - np.mean(beta)

        # Store samples
        beta_samples = np.zeros((self.n_iter - self.warmup, t))
        sigma_samples = np.zeros(self.n_iter - self.warmup)

        # Proposal standard deviations
        beta_proposal_sd = 0.1
        sigma_proposal_sd = 0.1

        current_log_post = self.log_posterior(beta, sigma, W, N)

        for it in range(self.n_iter):
            # Sample new beta
            beta_proposed = beta + np.random.randn(t) * beta_proposal_sd
            beta_proposed = beta_proposed - np.mean(beta_proposed)  # enforce sum=0

            # Sample new sigma
            sigma_proposed = np.exp(np.log(sigma) + np.random.randn() * sigma_proposal_sd)

            # Calculate proposed log-posterior
            proposed_log_post = self.log_posterior(beta_proposed, sigma_proposed, W, N)

            # Metropolis-Hastings acceptance ratio
            log_accept_ratio = proposed_log_post - current_log_post

            # For sigma: add Jacobian term for log-transform
            log_accept_ratio += np.log(sigma_proposed) - np.log(sigma)

            # Accept or reject
            if np.log(np.random.rand()) < log_accept_ratio:
                beta = beta_proposed
                sigma = sigma_proposed
                current_log_post = proposed_log_post

            # Store sample after warmup
            if it >= self.warmup:
                idx = it - self.warmup
                beta_samples[idx] = beta
                sigma_samples[idx] = sigma

        return beta_samples, sigma_samples

    def fit(self, W, N):
        """
        Fit the Bayesian Bradley-Terry model.

        Returns:
        --------
        result : dict
            Contains:
            - 'beta_mean': posterior mean of beta
            - 'beta_std': posterior std of beta
            - 'w_mean': posterior mean of w = exp(beta)
            - 'w_std': posterior std of w
            - 'sigma_mean': posterior mean of sigma
            - 'samples': all MCMC samples
        """
        beta_samples, sigma_samples = self.sample_posterior(W, N)

        # Compute posterior means
        beta_mean = np.mean(beta_samples, axis=0)
        beta_std = np.std(beta_samples, axis=0)

        # Convert to w (normalize to sum to 1)
        w_samples = np.exp(beta_samples)
        w_samples = w_samples / np.sum(w_samples, axis=1, keepdims=True)

        w_mean = np.mean(w_samples, axis=0)
        w_std = np.std(w_samples, axis=0)

        sigma_mean = np.mean(sigma_samples)

        # Compute ranking
        ranking = np.argsort(-w_mean)  # descending order

        return {
            'beta_mean': beta_mean,
            'beta_std': beta_std,
            'w_mean': w_mean,
            'w_std': w_std,
            'sigma_mean': sigma_mean,
            'samples': {
                'beta': beta_samples,
                'sigma': sigma_samples,
                'w': w_samples
            },
            'ranking': ranking
        }


class SpokoinyBradleyTerry:
    """
    Solves model from https://arxiv.org/pdf/2503.15045

    v_G = argmax_v (L(v) - ||Gv||^2/2)

    Parameters
    ----------
    S : numpy array of shape (n, n)
        Matrix where S[j,m] = sum of Y_{jm}^{(ℓ)} (number of times j beats m)
        Only upper triangular part of S is used.
    N : numpy array of shape (n, n)
        Matrix where N[j,m] = number of comparisons between j and m
        N should be symmetric with zeros on diagonal
    G : numpy array of shape (n, n) - adjacency matrix of graph
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    v_opt : numpy array of length n
        Optimal v that maximizes penalized log-likelihood
    """

    def __init__(self, S, N, G, tol=1e-8, max_iter=1000):
        self.n = S.shape[0]
        self.S = S
        self.N = N
        self.G_squared = G @ G.T
        self.v = np.ones(self.n) / self.n
        self.tol = tol
        self.max_iter = max_iter

    def neg_log_likelihood_penalized(self, v):
        """
        Compute -[L(v) - 0.5 * ||Gv||^2]
        """
        L = 0.0
        for m in range(self.n):
            for j in range(m):
                if self.N[j, m] > 0:
                    diff = v[j] - v[m]
                    L += diff * self.S[j, m] - self.N[j, m] * np.log(1 + np.exp(diff))
        penalty = 0.5 * v @ self.G_squared @ v
        return -L + penalty

    def grad_neg_log_likelihood_penalized(self, v):
        """
        Compute gradient of -[L(v) - 0.5 * ||Gv||^2]
        """
        grad = np.zeros(self.n)
        for m in range(self.n):
            for j in range(m):
                if j != m and self.N[min(j, m), max(j, m)] > 0:
                    diff = v[j] - v[m]
                    g = self.S[j, m] - self.N[j, m] * np.exp(diff) / (1 + np.exp(diff))
                    grad[j] += g
                    grad[m] -= g
        grad_penalty = self.G_squared @ v
        return -grad + grad_penalty

    def fit(self):
        result = minimize(
            self.neg_log_likelihood_penalized,
            self.v,
            jac=self.grad_neg_log_likelihood_penalized,
            method='L-BFGS-B',
            options={'gtol': self.tol, 'maxiter': self.max_iter, 'disp': False}
        )

        v_opt = result.x
        # Apply identifiability constraint: v_1 = 0
        # This doesn't change the model predictions but makes solution unique
        self.v = v_opt - v_opt[0]
        ranking = np.argsort(-self.v) + 1
        return {'probs': self.v, 'ranking': ranking}

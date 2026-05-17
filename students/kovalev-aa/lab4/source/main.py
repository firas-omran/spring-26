"""
Gaussian Mixture Model implementation from scratch.
Comparison with scikit-learn's implementation.
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        if random_state:
            np.random.seed(random_state)

    def _init_params(self, X):
        n_samples, n_features = X.shape

        idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[idx].copy()

        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights = np.ones(self.n_components) / self.n_components

    def _e_step(self, X):
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            try:
                resp[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], self.covariances[k], allow_singular=True
                )
            except np.linalg.LinAlgError:
                cov_reg = self.covariances[k] + 1e-6 * np.eye(X.shape[1])
                resp[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], cov_reg
                )

        row_sum = resp.sum(axis=1, keepdims=True)
        resp = resp / row_sum
        log_likelihood = np.sum(np.log(row_sum + 1e-10))

        return resp, log_likelihood

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        nk = resp.sum(axis=0)

        self.weights = nk / n_samples
        self.means = (resp.T @ X) / nk[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted = resp[:, k][:, np.newaxis] * diff
            self.covariances[k] = (weighted.T @ diff) / nk[k]
            self.covariances[k] += 1e-6 * np.eye(n_features)

    def fit(self, X):
        self._init_params(X)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            resp, curr_ll = self._e_step(X)

            if abs(curr_ll - prev_ll) < self.tol:
                break

            prev_ll = curr_ll
            self._m_step(X, resp)

        return self

    def score(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            try:
                likelihood += self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], self.covariances[k], allow_singular=True
                )
            except np.linalg.LinAlgError:
                cov_reg = self.covariances[k] + 1e-6 * np.eye(X.shape[1])
                likelihood += self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], cov_reg
                )

        return np.mean(np.log(likelihood + 1e-10))

    def predict(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            try:
                likelihood[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], self.covariances[k], allow_singular=True
                )
            except np.linalg.LinAlgError:
                cov_reg = self.covariances[k] + 1e-6 * np.eye(X.shape[1])
                likelihood[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], cov_reg
                )

        return np.argmax(likelihood, axis=1)


def get_dataset():
    X, _ = make_blobs(n_samples=1000, centers=3, n_features=2,
                      cluster_std=0.8, random_state=42)
    return StandardScaler().fit_transform(X)


def plot_results(X, custom_gmm, sklearn_gmm):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Левая часть: наша реализация
    pred_custom = custom_gmm.predict(grid).reshape(xx.shape)
    axes[0].contourf(xx, yy, pred_custom, alpha=0.3, cmap='Set1')
    axes[0].scatter(X[:, 0], X[:, 1], c=custom_gmm.predict(X),
                    cmap='Set1', s=15, alpha=0.6)
    axes[0].set_title(f'Custom GMM (score: {custom_gmm.score(X):.3f})')

    # Правая часть: sklearn
    pred_sklearn = sklearn_gmm.predict(grid).reshape(xx.shape)
    axes[1].contourf(xx, yy, pred_sklearn, alpha=0.3, cmap='Set1')
    axes[1].scatter(X[:, 0], X[:, 1], c=sklearn_gmm.predict(X),
                    cmap='Set1', s=15, alpha=0.6)
    axes[1].set_title(f'Sklearn GMM (score: {sklearn_gmm.score(X):.3f})')

    plt.tight_layout()
    plt.savefig('gmm_comparison.png', dpi=150)
    plt.show()


def run_experiment():
    X = get_dataset()
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    custom_gmm = GMM(n_components=3, max_iter=200, random_state=42)
    custom_gmm.fit(X_train)

    sklearn_gmm = GaussianMixture(n_components=3, max_iter=200,
                                   random_state=42, covariance_type='full')
    sklearn_gmm.fit(X_train)

    print(f"Custom GMM - Train: {custom_gmm.score(X_train):.4f}, Test: {custom_gmm.score(X_test):.4f}")
    print(f"Sklearn GMM - Train: {sklearn_gmm.score(X_train):.4f}, Test: {sklearn_gmm.score(X_test):.4f}")

    plot_results(X_test, custom_gmm, sklearn_gmm)


if __name__ == "__main__":
    run_experiment()
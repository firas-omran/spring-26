import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingCustom:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.init_pred = None
        self.feature_importances_ = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _log_loss_gradient(self, y, proba):
        # Градиент логистической функции потерь.
        return y - proba

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        pos = np.mean(y)
        self.init_pred = np.log(pos / (1 - pos)) if pos > 0 and pos < 1 else 0.0
        f = np.full(n_samples, self.init_pred)
        self.feature_importances_ = np.zeros(n_features)
        self.trees = []

        for t in range(self.n_estimators):
            proba = self._sigmoid(f)
            residuals = self._log_loss_gradient(y, proba)

            if self.subsample < 1.0:
                idx = rng.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=rng.randint(0, 1e6))
            tree.fit(X_sub, r_sub)
            update = self.learning_rate * tree.predict(X)
            f += update
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_

        self.feature_importances_ /= self.n_estimators
        return self

    def predict_proba(self, X):
        f = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        proba = self._sigmoid(f)
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
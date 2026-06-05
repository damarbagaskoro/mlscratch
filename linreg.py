import numpy as np


class LinearRegression:
    """
    Linear Regression model trained using full-batch gradient descent.

    This implementation intentionally uses explicit iterative summation
    instead of vectorized operations in order to demonstrate the mathematical
    foundations of gradient descent optimization and linear regression training.
    """

    def __init__(
        self, max_iter: int = 1000, tolerance: float = 1e-6, learning_rate: float = 1e-3
    ):
        # Gradient descent hyperparameter
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.learning_rate = learning_rate

        # Parameter validation
        self._validate_params()

        # Initialize class attributes
        self.theta = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self.cost_history_ = []

    def __repr__(self) -> str:
        return (
            f"LinearRegression("
            f"max_iter={self.max_iter}, "
            f"learning_rate={self.learning_rate}, "
            f"tolerance={self.tolerance})"
        )

    def get_params(self) -> dict:

        params = {
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "learning_rate": self.learning_rate,
        }

        return params

    def set_params(self, **params) -> "LinearRegression":
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator "
                    f"{self.__class__.__name__}. "
                    f"Valid parameters are: {list(valid_params.keys())}."
                )
            setattr(self, key, value)

        # Parameter validation
        self._validate_params()

        return self

    # ====================================================
    # LEARNING ALGO
    # ====================================================

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        m_rows, n_X = X.shape
        x_b = np.ones((m_rows, n_X + 1))

        # Join bias term with input array X
        x_b[:, 1:] = X

        return x_b

    def _compute_hypothesis(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        m_rows, n_features = X.shape
        hypothesis_arr = np.zeros((m_rows, 1))

        for i in range(m_rows):
            linear_combination = 0.0
            for j in range(n_features):
                linear_combination += thetas[j, 0] * X[i, j]
            hypothesis_arr[i, 0] = linear_combination

        return hypothesis_arr

    def _compute_cost(self, residuals: np.ndarray) -> float:
        m_rows = residuals.shape[0]
        sum_squared_residual = 0.0

        for i in range(m_rows):
            squared_residual = residuals[i, 0] ** 2
            sum_squared_residual += squared_residual

        cost = sum_squared_residual / 2

        return float(cost)

    def _compute_gradients(self, residuals: np.ndarray, X: np.ndarray) -> np.ndarray:
        m_rows, n_features = X.shape
        gradients = np.zeros((n_features, 1))

        for i in range(m_rows):
            residual = residuals[i, 0]
            for j in range(n_features):
                gradients[j, 0] += residual * X[i, j]

        return gradients

    def _train_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        _, n_features = X.shape
        prev_cost = float("inf")
        iteration = 0

        # Initialize parameter vector theta
        # Shape: (n_features, 1)
        thetas = np.zeros((n_features, 1))

        for iteration in range(self.max_iter):
            # Compute hypothesis array
            # Shape: (m_rows, 1)
            hypothesis = self._compute_hypothesis(X, thetas)

            # Compute residuals array
            # Shape: (m_rows, 1)
            residuals = hypothesis - y

            # Compute cost based on residuals array, returning a float
            curr_cost = self._compute_cost(residuals)
            self.cost_history_.append(curr_cost)

            ## Cost convergence check
            if abs(curr_cost - prev_cost) < self.tolerance:
                break

            prev_cost = curr_cost

            # Compute gradients array
            # Shape: (n_features, 1)
            gradients = self._compute_gradients(residuals, X)

            # Update parameter theta
            for j in range(n_features):
                thetas[j, 0] -= self.learning_rate * gradients[j, 0]

        # Update model attributes
        self.theta = thetas
        self.intercept_ = float(thetas[0, 0])
        self.coef_ = thetas[1:, :]
        ## Convert iteration to a count
        self.n_iter_ = iteration + 1 if self.max_iter > 0 else 0

    # ====================================================
    # INPUT AND MODEL STATE VALIDATION
    # ====================================================

    def _validate_params(self) -> None:

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        if self.tolerance < 0:
            raise ValueError("tolerance must be a non-negative number")

    def _validate_X(self, X: np.ndarray, allow_single_sample: bool = False) -> None:

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray")
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must contain numeric values only")
        min_samples = 1 if allow_single_sample else 2
        if X.shape[0] < min_samples:
            raise ValueError(f"X must contain at least {min_samples} sample(s)")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values")

    def _validate_y(self, y: np.ndarray) -> None:

        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray")
        if y.ndim != 2:
            raise ValueError("y must be a 2-dimensional array")
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError("y must contain numeric values only")
        if y.shape[1] != 1:
            raise ValueError("y must have shape (m_rows, 1)")
        if y.shape[0] < 2:
            raise ValueError("y must contain at least two samples")
        # if y.shape[0] == 0:
        # raise ValueError("y must contain at least one sample")
        # if y.shape[0] == 1:
        # raise ValueError("y must contain at least two samples")
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")
        if np.any(np.isinf(y)):
            raise ValueError("y contains infinite values")

    def _validate_X_y(
        self, X: np.ndarray, y: np.ndarray, allow_single_sample: bool = False
    ) -> None:
        self._validate_X(X, allow_single_sample=allow_single_sample)
        self._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows")

    def _validate_predict_X(self, X: np.ndarray) -> None:
        expected = self.theta.shape[0] - 1

        if X.shape[1] != expected:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but model was trained on {expected} features."
            )

    def _check_is_fitted(self) -> None:

        if self.theta is None:
            raise ValueError(
                "Model has not been fitted yet. "
                "Call fit() with appropriate arguments before using this model."
            )

    # ====================================================
    # ESTIMATOR API
    # ====================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        # Input validation
        self._validate_X_y(X, y, allow_single_sample=False)

        # Reset learning attributes
        self.theta = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self.cost_history_ = []

        # Add bias term to the input array X
        x_b = self._add_bias(X)

        # Train gradient descent
        self._train_gd(x_b, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Model state and input validation
        self._check_is_fitted()
        self._validate_X(X, allow_single_sample=True)
        self._validate_predict_X(X)

        # Add bias term to the input array X
        x_b = self._add_bias(X)

        # Compute prediction using learned parameter theta
        y_pred = self._compute_hypothesis(x_b, self.theta)

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Model state and input validation
        self._check_is_fitted()
        self._validate_X_y(X, y, allow_single_sample=True)
        self._validate_predict_X(X)

        m_rows = y.shape[0]
        y_mean = np.mean(y)
        y_pred = self.predict(X)

        ss_res = 0.0
        ss_total = 0.0

        for i in range(m_rows):
            ss_res += (y[i, 0] - y_pred[i, 0]) ** 2
            ss_total += (y[i, 0] - y_mean) ** 2

        # Avoid zero-division in r_square calculation
        if ss_total == 0.0:
            raise ValueError(
                "R² is undefined when y has zero variance (all target values are identical)"
            )

        r_square = 1 - (ss_res / ss_total)

        return float(r_square)

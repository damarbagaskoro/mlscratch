# mlscratch

A from-scratch implementation of machine learning algorithms in Python — built to understand the math, not just use the tools.

This library intentionally avoids high-level ML frameworks like scikit-learn for its core implementations. Every algorithm is written using only **NumPy**, with explicit loops and manual math to expose the underlying mechanics of each method.

> ⚠️ **This is a learning-oriented project, not a production library.** If you need performance-optimized ML, use [scikit-learn](https://scikit-learn.org/).

---

## Algorithms

| Algorithm | Module | Status |
|---|---|---|
| Linear Regression | `linreg.py` | ✅ Available |

More algorithms coming soon.

---

## Installation

```bash
git clone https://github.com/damarbagaskoro/mlscratch.git
cd mlscratch
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from linreg import LinearRegression

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([[2], [4], [6], [8], [10]], dtype=float)

model = LinearRegression(max_iter=1000, learning_rate=1e-3)
model.fit(X, y)

print(model.coef_)       # Learned weights
print(model.intercept_)  # Learned bias
print(model.score(X, y)) # R² score
```

---

## Linear Regression (`linreg.py`)

Trained using **full-batch gradient descent**. The implementation deliberately uses explicit iterative summation over vectorized operations to make the math readable and traceable.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_iter` | `int` | `1000` | Maximum number of gradient descent iterations |
| `learning_rate` | `float` | `1e-3` | Step size for parameter updates |
| `tolerance` | `float` | `1e-6` | Stops early if cost improvement falls below this threshold |

### Attributes (after fitting)

| Attribute | Description |
|---|---|
| `coef_` | Learned weight vector, shape `(n_features, 1)` |
| `intercept_` | Learned bias term |
| `n_iter_` | Number of iterations the training actually ran |
| `cost_history_` | List of MSE cost values recorded at each iteration |

### Methods

- **`fit(X, y)`** — Train the model on input `X` and target `y`
- **`predict(X)`** — Return predicted values for input `X`
- **`score(X, y)`** — Return the R² coefficient of determination
- **`get_params()`** — Return current hyperparameters as a dict
- **`set_params(**params)`** — Update hyperparameters by name

### Input Format

Both `X` and `y` must be **2D NumPy arrays* and expected to be scaled first*:
- `X` shape: `(n_samples, n_features)`
- `y` shape: `(n_samples, 1)`

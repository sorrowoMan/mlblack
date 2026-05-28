from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _positive_transform(values: np.ndarray, transform: str) -> np.ndarray:
    key = str(transform or "softplus").strip().lower()
    arr = np.asarray(values, dtype=float)
    if key == "softplus":
        return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)
    if key == "abs":
        return np.abs(arr)
    if key == "exp":
        return np.exp(np.clip(arr, -50.0, 50.0))
    if key in {"identity", "none"}:
        return np.maximum(arr, 0.0)
    raise ValueError(f"unsupported positive transform: {transform}")


@dataclass(frozen=True)
class NormalDistributionModel:
    mu_model: Any
    sigma_model: Any
    sigma_transform: str = "softplus"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.mu_model.predict(X), dtype=float).reshape(-1)

    def predict_params(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self.predict(X)
        raw_sigma = np.asarray(self.sigma_model.predict(X), dtype=float).reshape(-1)
        sigma = _positive_transform(raw_sigma, self.sigma_transform)
        return mu, sigma

    def log_prob(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu, sigma = self.predict_params(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        z = (y_arr - mu) / np.maximum(sigma, 1e-8)
        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * z * z


@dataclass(frozen=True)
class PoissonDistributionModel:
    rate_model: Any
    rate_transform: str = "softplus"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.rate_model.predict(X), dtype=float).reshape(-1)
        return _positive_transform(raw, self.rate_transform)

    def predict_params(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def log_prob(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        rate = self.predict_params(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        r = np.maximum(rate, 1e-8)
        # log P(y|rate) = y*log(rate) - rate - log(y!)
        log_factorial = np.where(y_arr <= 0, 0.0, y_arr * np.log(y_arr) - y_arr + 0.5 * np.log(2 * np.pi * y_arr + 1.0 / 3.0))
        return y_arr * np.log(r) - r - log_factorial


@dataclass(frozen=True)
class NegativeBinomialDistributionModel:
    mu_model: Any
    alpha_model: Any
    alpha_transform: str = "softplus"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.mu_model.predict(X), dtype=float).reshape(-1)

    def predict_params(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self.predict(X)
        raw_alpha = np.asarray(self.alpha_model.predict(X), dtype=float).reshape(-1)
        alpha = _positive_transform(raw_alpha, self.alpha_transform)
        return np.maximum(mu, 1e-8), np.maximum(alpha, 1e-8)

    def log_prob(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu, alpha = self.predict_params(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        # NB(y | mu, alpha): Gamma(y + 1/alpha) / (Gamma(y+1) * Gamma(1/alpha))
        #   * (alpha*mu / (1+alpha*mu))^y * (1/(1+alpha*mu))^(1/alpha)
        # In log space, using Stirling's approximation for gamma when large
        inv_alpha = 1.0 / alpha
        log_gamma_y1a = _log_gamma_approx(y_arr + inv_alpha)
        log_gamma_y1 = _log_gamma_approx(y_arr + 1.0)
        log_gamma_1a = _log_gamma_approx(inv_alpha)
        p = 1.0 / (1.0 + alpha * mu)
        return (log_gamma_y1a - log_gamma_y1 - log_gamma_1a
                + y_arr * np.log(1.0 - p) + inv_alpha * np.log(p))


def _log_gamma_approx(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(np.clip(np.asarray(x, dtype=float), 1e-8, None), dtype=float)
    # Stirling: log Gamma(x) ≈ (x-0.5)*log(x) - x + 0.5*log(2*pi) + 1/(12*x)
    return (x_arr - 0.5) * np.log(x_arr) - x_arr + 0.5 * np.log(2 * np.pi) + 1.0 / (12.0 * x_arr)

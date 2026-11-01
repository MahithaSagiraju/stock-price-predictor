"""Lightweight NumPy inference for the trained stock prediction LSTM.

The original model is a small Keras LSTM stack saved as
``stock_prediction_model.keras``. Loading Keras/TensorFlow requires a ~600 MB
install and has no wheels for the newest Python versions, so the weights are
exported to ``stock_prediction_model_weights.npz`` and evaluated here with
NumPy only. ``NumpyStockModel.predict`` mirrors the Keras ``predict`` API used
by the app.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHTS_PATH = Path(__file__).resolve().parent / "stock_prediction_model_weights.npz"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class NumpyStockModel:
    """NumPy reimplementation of the trained 4-layer LSTM + Dense model."""

    def __init__(self, path: Path | str = WEIGHTS_PATH) -> None:
        self.path = Path(path)
        with np.load(self.path) as weights:
            layers = []
            index = 0
            while f"lstm_{index}_kernel" in weights:
                layers.append(
                    (
                        weights[f"lstm_{index}_kernel"].astype(np.float32),
                        weights[f"lstm_{index}_recurrent"].astype(np.float32),
                        weights[f"lstm_{index}_bias"].astype(np.float32),
                        bool(int(weights[f"lstm_{index}_return_sequences"][0])),
                    )
                )
                index += 1
            self._layers = layers
            self._dense_kernel = weights["dense_kernel"].astype(np.float32)
            self._dense_bias = weights["dense_bias"].astype(np.float32)

    def _lstm(
        self,
        x: np.ndarray,
        kernel: np.ndarray,
        recurrent: np.ndarray,
        bias: np.ndarray,
        return_sequences: bool,
    ) -> np.ndarray:
        batch, timesteps, _ = x.shape
        units = kernel.shape[1] // 4
        h = np.zeros((batch, units), dtype=np.float32)
        c = np.zeros((batch, units), dtype=np.float32)
        outputs = np.empty((batch, timesteps, units), dtype=np.float32)

        for step in range(timesteps):
            z = x[:, step, :] @ kernel + h @ recurrent + bias
            i = _sigmoid(z[:, :units])
            f = _sigmoid(z[:, units : 2 * units])
            g = _relu(z[:, 2 * units : 3 * units])
            o = _sigmoid(z[:, 3 * units :])
            c = f * c + i * g
            # The model was built with activation="relu", which Keras applies
            # to both the candidate cell state and the output cell state.
            h = o * _relu(c)
            outputs[:, step, :] = h

        return outputs if return_sequences else h

    def predict(self, x, batch_size: int | None = None, verbose: int = 0) -> np.ndarray:
        """Return predictions for ``x`` shaped ``(samples, 100, 1)``."""
        data = np.asarray(x, dtype=np.float32)
        if data.ndim == 2:
            data = data[..., np.newaxis]
        if data.ndim != 3:
            raise ValueError("Expected input shaped (samples, timesteps, features).")

        out = data
        for kernel, recurrent, bias, return_sequences in self._layers:
            out = self._lstm(out, kernel, recurrent, bias, return_sequences)

        return out @ self._dense_kernel + self._dense_bias

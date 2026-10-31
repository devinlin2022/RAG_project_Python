"""Exact inner-product index over L2-normalised vectors.

Below ~100k chunks a NumPy matmul already returns the exact top-k, so the
interface mirrors FAISS IndexFlatIP without the dependency: add, search, remove.
"""

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def normalize_one(vector: list[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype="float32").reshape(1, -1)
    return normalize(arr)[0]


class FlatIndex:
    """Row order is the caller's record order; no id map needed."""

    def __init__(self, vectors: np.ndarray | None = None) -> None:
        self.matrix = normalize(vectors) if vectors is not None and len(vectors) else None

    def __len__(self) -> int:
        return 0 if self.matrix is None else int(self.matrix.shape[0])

    @property
    def dim(self) -> int:
        return 0 if self.matrix is None else int(self.matrix.shape[1])

    def append(self, vectors: np.ndarray) -> None:
        if vectors is None or len(vectors) == 0:
            return
        block = normalize(vectors)
        self.matrix = block if self.matrix is None else np.vstack([self.matrix, block])

    def search(
        self,
        vector: list[float] | np.ndarray,
        k: int,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Top-k rows as (row, cosine). `allowed` is a boolean mask over rows."""
        if self.matrix is None or k <= 0:
            return []
        query = np.asarray(vector, dtype="float32").reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"向量维度不一致：索引 {self.dim}，查询 {query.shape[0]}")

        scores = self.matrix @ query
        if allowed is not None:
            if allowed.shape[0] != scores.shape[0]:
                raise ValueError("过滤掩码长度与索引行数不一致")
            scores = np.where(allowed, scores, -np.inf)

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(row), float(scores[row])) for row in top if np.isfinite(scores[row])]

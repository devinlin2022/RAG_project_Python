"""Reciprocal rank fusion.

Rank-based fusion needs no score normalisation, so a cosine in [0, 1] and a
BM25 score in [0, 40] can be combined without inventing a weight. k=60 is the
value from the original TREC paper and is kept configurable.
"""


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    """Fuse ranked id lists into {id: score}. Higher is better."""
    fused: dict[int, float] = {}
    for position, ranking in enumerate(rankings):
        weight = weights[position] if weights else 1.0
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused

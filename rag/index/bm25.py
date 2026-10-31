"""BM25 over chunk tokens, with no external dependency."""

import math
from collections import Counter


class BM25:
    """Okapi BM25 with the usual k1/b defaults."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.size = len(corpus)
        self.lengths = [len(doc) for doc in corpus]
        self.avg_len = (sum(self.lengths) / self.size) if self.size else 0.0

        self.term_freqs: list[Counter] = [Counter(doc) for doc in corpus]
        doc_freq: Counter = Counter()
        for freqs in self.term_freqs:
            doc_freq.update(freqs.keys())
        self.doc_freq = doc_freq
        self.idf = {
            term: math.log(1 + (self.size - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        """Score every document. Unknown query terms contribute nothing."""
        result = [0.0] * self.size
        if not self.size or not query_tokens:
            return result

        for token in dict.fromkeys(query_tokens):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for row, freqs in enumerate(self.term_freqs):
                tf = freqs.get(token)
                if not tf:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[row] / (self.avg_len or 1))
                result[row] += idf * tf * (self.k1 + 1) / (tf + norm)
        return result

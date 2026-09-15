import math
import re
from collections import Counter
from collections.abc import Sequence


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def search(
        self,
        query: str,
        documents: Sequence[dict],
        top_k: int,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_terms = _tokens(query)
        if not query_terms or not documents:
            return []

        tokenized = [_tokens(document["document"]) for document in documents]
        document_frequency = Counter(
            token
            for tokens in tokenized
            for token in set(tokens)
        )
        average_length = sum(map(len, tokenized)) / len(tokenized)

        scored = []
        for document, tokens in zip(documents, tokenized):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in frequencies:
                    continue
                idf = math.log(
                    1 + (len(documents) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequencies[term] + 1.5 * (
                    1 - 0.75 + 0.75 * len(tokens) / average_length
                )
                score += idf * frequencies[term] * 2.5 / denominator
            if score:
                scored.append((score, document))

        return [
            document
            for _, document in sorted(
                scored,
                key=lambda item: (-item[0], item[1]["chunk_id"]),
            )[:top_k]
        ]

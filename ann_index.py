from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Literal
import json
import math

import faiss
import numpy as np
import torch


ANNBackend = Literal["flat", "hnsw", "lsh"]
Tier = Literal["gold", "silver"]


# ============================================================
# Utilities
# ============================================================

def _to_faiss(x: torch.Tensor) -> np.ndarray:
    """
    Convert normalized torch tensor -> float32 numpy.
    """
    return (
        x.detach()
        .cpu()
        .numpy()
        .astype("float32")
    )


def _assert_normalized(x: torch.Tensor, eps: float = 1e-3):
    norms = torch.linalg.norm(x, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=eps):
        raise ValueError("Embeddings must be L2-normalized")


# ============================================================
# LSH (Random Hyperplane) Index
# ============================================================

class RandomHyperplaneLSH:
    """
    Simple, deterministic random-hyperplane LSH.

    This is NOT FAISS-backed.
    It exists as a diagnostic / recall-baseline comparator.
    """

    def __init__(self, dim: int, num_planes: int = 24):
        self.dim = dim
        self.num_planes = num_planes
        self.planes: np.ndarray | None = None
        self.buckets: Dict[str, List[int]] = {}
        self.vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray):
        rng = np.random.default_rng(seed=42)
        self.planes = rng.standard_normal((self.num_planes, self.dim)).astype(
            "float32"
        )

        self.vectors = vectors
        self.buckets.clear()

        projections = np.dot(vectors, self.planes.T)
        hashes = projections > 0

        for i, h in enumerate(hashes):
            key = "".join("1" if b else "0" for b in h)
            self.buckets.setdefault(key, []).append(i)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        proj = np.dot(query, self.planes.T)[0]
        key = "".join("1" if b else "0" for b in proj > 0)

        candidates = self.buckets.get(key, [])
        if not candidates:
            return []

        vecs = self.vectors[candidates]
        sims = np.dot(vecs, query.T).reshape(-1)

        order = np.argsort(-sims)[:top_k]
        return [(candidates[i], float(sims[i])) for i in order]


# ============================================================
# Intent ANN Index
# ============================================================

class IntentANNIndex:
    """
    Per-intent, per-tier ANN index with multiple backends.

    Supports:
    - FAISS Flat (exact cosine)
    - FAISS HNSW
    - Random Hyperplane LSH (baseline)
    """

    def __init__(
        self,
        dim: int,
        backend: ANNBackend = "flat",
        hnsw_m: int = 32,
        lsh_planes: int = 24,
    ):
        self.dim = dim
        self.backend = backend
        self.hnsw_m = hnsw_m
        self.lsh_planes = lsh_planes

        # indices[intent][tier] -> index
        self.indices: Dict[str, Dict[Tier, object]] = {}

        # texts[intent][tier] -> list[str]
        self.texts: Dict[str, Dict[Tier, List[str]]] = {}
    def ensure_intent(self, intent: str):
        if intent not in self.indices:
            self.indices[intent] = {}
            self.texts[intent] = {}


    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    def _build_faiss_flat(self, vectors: np.ndarray):
        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors)
        return index

    def _build_faiss_hnsw(self, vectors: np.ndarray):
        index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
        index.hnsw.efConstruction = 200
        index.add(vectors)
        return index

    def _build_lsh(self, vectors: np.ndarray):
        lsh = RandomHyperplaneLSH(self.dim, self.lsh_planes)
        lsh.build(vectors)
        return lsh

    def build_intent_tier(
        self,
        intent: str,
        tier: Tier,
        embeddings: torch.Tensor,
        texts: List[str],
    ):
        """
        Build ANN for (intent, tier).
        """

        if embeddings.ndim != 2 or embeddings.size(1) != self.dim:
            raise ValueError("Invalid embedding shape")

        if len(texts) != embeddings.size(0):
            raise ValueError("Text / embedding mismatch")

        _assert_normalized(embeddings)
        vectors = _to_faiss(embeddings)

        if intent not in self.indices:
            self.indices[intent] = {}
            self.texts[intent] = {}

        if self.backend == "flat":
            index = self._build_faiss_flat(vectors)
        elif self.backend == "hnsw":
            index = self._build_faiss_hnsw(vectors)
        elif self.backend == "lsh":
            index = self._build_lsh(vectors)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.indices[intent][tier] = index
        self.texts[intent][tier] = texts

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def _search_faiss(
        self,
        index: faiss.Index,
        vectors: List[str],
        query: np.ndarray,
        top_k: int,
    ):
        scores, idx = index.search(query, top_k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i >= 0:
                out.append((vectors[i], float(s)))
        return out

    def _search_lsh(
        self,
        index: RandomHyperplaneLSH,
        vectors: List[str],
        query: np.ndarray,
        top_k: int,
    ):
        hits = index.search(query, top_k)
        return [(vectors[i], s) for i, s in hits]

    def search(
        self,
        intent: str,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[str, float, Tier]]:
        """
        Query order:
        1. GOLD
        2. SILVER (only if needed)

        Returns:
            (text, score, tier)
        """

        if intent not in self.indices:
            return []


        q = _to_faiss(query_embedding).reshape(1, -1)

        results: List[Tuple[str, float, Tier]] = []

        for tier in ("gold", "silver"):
            if tier not in self.indices[intent]:
                continue

            index = self.indices[intent][tier]
            texts = self.texts[intent][tier]

            if self.backend in ("flat", "hnsw"):
                hits = self._search_faiss(index, texts, q, top_k)
            else:
                hits = self._search_lsh(index, texts, q, top_k)

            for t, s in hits:
                results.append((t, s, tier))

            if len(results) >= top_k:
                break

        # deterministic ordering
        results.sort(key=lambda x: (-x[1], x[2]))
        return results[:top_k]

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "dim": self.dim,
            "backend": self.backend,
            "hnsw_m": self.hnsw_m,
            "lsh_planes": self.lsh_planes,
            "intents": list(self.indices.keys()),
        }

        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        for intent, tiers in self.indices.items():
            for tier, index in tiers.items():
                prefix = f"{intent}.{tier}"

                if isinstance(index, faiss.Index):
                    faiss.write_index(index, str(path / f"{prefix}.index"))
                else:
                    with open(path / f"{prefix}.lsh.json", "w") as f:
                        json.dump(
                            {
                                "planes": index.planes.tolist(),
                                "buckets": index.buckets,
                                "vectors": index.vectors.tolist(),
                            },
                            f,
                        )

                with open(path / f"{prefix}.texts.json", "w", encoding="utf-8") as f:
                    json.dump(self.texts[intent][tier], f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "IntentANNIndex":
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        ann = cls(
            dim=meta["dim"],
            backend=meta["backend"],
            hnsw_m=meta["hnsw_m"],
            lsh_planes=meta["lsh_planes"],
        )

        for intent in meta["intents"]:
            ann.indices[intent] = {}
            ann.texts[intent] = {}

            for tier in ("gold", "silver"):
                prefix = f"{intent}.{tier}"

                faiss_path = path / f"{prefix}.index"
                lsh_path = path / f"{prefix}.lsh.json"

                if faiss_path.exists():
                    ann.indices[intent][tier] = faiss.read_index(str(faiss_path))
                elif lsh_path.exists():
                    with open(lsh_path, "r") as f:
                        payload = json.load(f)

                    lsh = RandomHyperplaneLSH(ann.dim, ann.lsh_planes)
                    lsh.planes = np.array(payload["planes"], dtype="float32")
                    lsh.buckets = {k: v for k, v in payload["buckets"].items()}
                    lsh.vectors = np.array(payload["vectors"], dtype="float32")
                    ann.indices[intent][tier] = lsh
                else:
                    continue

                with open(path / f"{prefix}.texts.json", "r", encoding="utf-8") as f:
                    ann.texts[intent][tier] = json.load(f)

        return ann

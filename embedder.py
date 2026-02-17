from pathlib import Path
from typing import List, Dict, Optional, NamedTuple
import json
import pickle

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from ann_index import IntentANNIndex, Tier


# ============================================================
# Prediction Contract
# ============================================================

class IntentPrediction(NamedTuple):
    intent: str
    confidence: float
    method: str
    margin: float


# ============================================================
# Embedder (Gold/Silver Aware)
# ============================================================

class Embedder:
    """
    Multilingual intent embedder.

    Gold examples:
      - define centroids
      - define authority

    Silver examples:
      - improve ANN recall
      - never dominate confidence
    """

    def __init__(
        self,
        model_name: str = "sft_mnrl_model",
        device: Optional[str] = None,
        ann_backend: str = "flat",
        hnsw_m: int = 32,
        lsh_planes: int = 24,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        self.gold_examples: Dict[str, List[str]] = {}
        self.silver_examples: Dict[str, List[str]] = {}
        self.intent_centroids: Dict[str, torch.Tensor] = {}

        self.ann_index: Optional[IntentANNIndex] = None
        self.ann_backend = ann_backend
        self.hnsw_m = hnsw_m
        self.lsh_planes = lsh_planes

    # ----------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------
    def embed(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            return torch.empty(
                (0, self.model.get_sentence_embedding_dimension()),
                device=self.device,
            )
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    # ----------------------------------------------------------
    # Intent Registration (Gold + Silver)
    # ----------------------------------------------------------
    def register_intent(
        self,
        intent: str,
        gold_examples: List[str],
        silver_examples: Optional[List[str]] = None,
        batch_size: int = 64,
    ):
        if not gold_examples:
            raise ValueError(f"Intent '{intent}' must have gold examples")

        silver_examples = silver_examples or []

        self.gold_examples[intent] = list(dict.fromkeys(gold_examples))
        self.silver_examples[intent] = list(dict.fromkeys(silver_examples))

        gold_emb = self.embed(self.gold_examples[intent], batch_size)
        centroid = F.normalize(gold_emb.mean(dim=0), dim=0)
        self.intent_centroids[intent] = centroid

    # ----------------------------------------------------------
    # ANN Construction (Gold + Silver)
    # ----------------------------------------------------------
    def build_ann_index(self, batch_size: int = 64):
        dim = self.model.get_sentence_embedding_dimension()
        self.ann_index = IntentANNIndex(
            dim=dim,
            backend=self.ann_backend,
            hnsw_m=self.hnsw_m,
            lsh_planes=self.lsh_planes,
        )

        for intent in self.gold_examples:
            gold_texts = self.gold_examples[intent]
            silver_texts = self.silver_examples.get(intent, [])

            # Embed gold and silver separately
            if gold_texts:
                gold_embeddings = self.embed(gold_texts, batch_size)
                self.ann_index.build_intent_tier(
                    intent, "gold", gold_embeddings, gold_texts
                )

            if silver_texts:
                silver_embeddings = self.embed(silver_texts, batch_size)
                self.ann_index.build_intent_tier(
                    intent, "silver", silver_embeddings, silver_texts
                )


    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------
    def predict_intent(
        self,
        text: str,
        top_k: int = 5,
        min_margin: float = 0.08,
        min_confidence: float = 0.65,
    ) -> IntentPrediction:

        query = self.embed([text])[0]

        # ---------- centroid scoring (gold only)
        # ---------- structural gates (pre-pass)
        allowed_intents, score_bias = apply_structural_gates(
            text,
            set(self.intent_centroids.keys()),
        )

        # ---------- centroid scoring (gold only)
        centroid_scores = [
            (
                intent,
                torch.dot(query, c).item() + score_bias.get(intent, 0.0),
            )
            for intent, c in self.intent_centroids.items()
            if intent in allowed_intents
        ]

        centroid_scores.sort(key=lambda x: x[1], reverse=True)
        if not centroid_scores:
            return IntentPrediction(
                "UNKNOWN", 0.0, "structural_reject", 0.0
            )

        best_intent, best_score = centroid_scores[0]
        second_score = centroid_scores[1][1] if len(centroid_scores) > 1 else 0.0
        margin = best_score - second_score

        # ---------- ANN refinement
        ann_hits = (
            self.ann_index.search(best_intent, query, top_k)
            if self.ann_index
            else []
        )

        best_ann_score = 0.0
        best_ann_tier = None

        if ann_hits:
            _, best_ann_score, best_ann_tier = ann_hits[0]

        # ---------- decision logic
        if best_score >= min_confidence and margin >= min_margin:
            return IntentPrediction(
                best_intent, best_score, "gold_centroid", margin
            )

        if best_ann_score >= min_confidence:
            method = (
                "ann_gold"
                if best_ann_tier == Tier("gold")
                else "ann_silver"
            )
            return IntentPrediction(
                best_intent, best_ann_score, method, margin
            )

        return IntentPrediction(
            "UNKNOWN",
            max(best_score, best_ann_score),
            "rejected",
            margin,
        )

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------
    def save_model(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path / "sbert_model")

        with open(path / "intent_store.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gold": self.gold_examples,
                    "silver": self.silver_examples,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if self.ann_index:
            self.ann_index.save(path / "ann_index")

    @classmethod
    def load_model(
        cls, path: str, device: Optional[str] = None
    ) -> "Embedder":
        path = Path(path)
        embedder = cls(
            model_name=str(path / "sbert_model"),
            device=device,
        )

        with open(path / "intent_store.json", "r", encoding="utf-8") as f:
            store = json.load(f)

        for intent in store["gold"]:
            embedder.register_intent(
                intent,
                store["gold"][intent],
                store["silver"].get(intent, []),
            )

        ann_path = path / "ann_index"
        if ann_path.exists():
            embedder.ann_index = IntentANNIndex.load(ann_path)

        return embedder

    def save_pickle(self, pkl_path: str):
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, pkl_path: str) -> "Embedder":
        with open(pkl_path, "rb") as f:
            return pickle.load(f)


# ============================
# Regexes (language-agnostic)
# ============================
from typing import Set, Dict
from collections import defaultdict
import re
_URL_RE = re.compile(r"(https?://|www\.)", re.I)
_EMAIL_RE = re.compile(r"\S+@\S+")
_LONG_DIGIT_RE = re.compile(r"\d{8,}")
_UNIT_RE = re.compile(r"\b(mg|g|kg|ml|l|cl|dl|kcal|kj|%)\b", re.I)


# ============================
# Gate evaluation
# ============================

def apply_structural_gates(
    text: str,
    all_intents: Set[str],
) -> tuple[Set[str], Dict[str, float]]:
    """
    Returns:
      - allowed_intents: reduced intent search space
      - score_bias: soft additive bias per intent

    This function MUST:
      - never force a single intent (except URLs)
      - never remove UNKNOWN handling
      - never rely on language-specific semantics
    """

    allowed_intents = set(all_intents)
    score_bias = defaultdict(float)

    t = text.strip().lower()
    if not t:
        return allowed_intents, score_bias

    # --------------------------------------------------
    # HARD STRUCTURAL GATES (safe)
    # --------------------------------------------------

    # URL / website (hard gate)
    if _URL_RE.search(t):
        return {"website"}, score_bias

    # Email / phone-like contact info
    if _EMAIL_RE.search(t) or _LONG_DIGIT_RE.search(t):
        allowed_intents -= {
            "ingredients",
            "usage_instructions",
            "storage_instructions",
        }

    # --------------------------------------------------
    # SOFT STRUCTURAL BIASES (safe)
    # --------------------------------------------------

    # Net quantity (digits + unit)
    if _UNIT_RE.search(t) and any(c.isdigit() for c in t):
        score_bias["net_quantity"] += 0.10

    return allowed_intents, score_bias
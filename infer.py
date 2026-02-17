from __future__ import annotations

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("intent_infer")

# ============================================================
# Structural Feature Helpers
# ============================================================
_DIGIT_RE = re.compile(r"\d")
_LETTER_RE = re.compile(r"[A-Za-z]")
_EMAIL_RE = re.compile(r"\S+@\S+")
_URL_RE = re.compile(r"https?://|www\.") 
_UNIT_LIKE_RE = re.compile(r"\d+\s?(mg|g|kg|kcal|kj|ml|l|%)", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+")
# Regex triggers
_ALLERGEN_REGEX = re.compile(r"(may contain|traces|contains|चेतावनी|एलर्जेन)", re.I)
_USAGE_REGEX = re.compile(r"(dilute|chauffer|microwave)", re.I)

def compute_structural_features(text: str) -> Dict[str, float]:
    if not text:
        return {}
    tokens = text.split()
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return {
        "num_tokens": len(tokens),
        "num_sentences": max(1, len(sentences)),
        "num_digits": len(_DIGIT_RE.findall(text)),
        "num_letters": len(_LETTER_RE.findall(text)),
        "num_emails": len(_EMAIL_RE.findall(text)),
        "num_urls": len(_URL_RE.findall(text)),
        "num_units": len(_UNIT_LIKE_RE.findall(text)),
        "avg_token_len": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
    }

# ============================================================
# Import existing ANN implementation
# ============================================================
from ann_index import IntentANNIndex

# ============================================================
# Intent Inferencer
# ============================================================
class IntentInferencer:
    """
    Multilingual intent inference with embeddings + ANN + structural gating + tier/intent weighting.
    """

    def __init__(
        self,
        model_dir: str,
        data_path: Optional[str] = None,
        ann_backend: str = "flat",
        device: str = "cpu",
        temperature: float = 1.0,
        gold_bonus: float = 0.08,
        silver_bonus: float = 0.0,
        rebuild_ann: bool = False,
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.gold_bonus = gold_bonus
        self.silver_bonus = silver_bonus

        self.model_dir = Path(model_dir)
        self.encoder_path = self.model_dir / "encoder"
        self.centroids_path = self.model_dir / "centroids.pt"
        self.ann_path = self.model_dir / "ann_index"

        # Load Encoder
        logger.info(f"Loading encoder from {self.encoder_path}")
        self.model = SentenceTransformer(str(self.encoder_path), device=self.device)
        self.model.eval()

        # Load Centroids
        logger.info(f"Loading centroids from {self.centroids_path}")
        if not self.centroids_path.exists():
            raise FileNotFoundError(f"Centroids not found at {self.centroids_path}")
        centroid_data = torch.load(self.centroids_path, map_location=self.device)
        self.intent_list = centroid_data["intent_list"]
        self.intent_centroids = centroid_data["centroids"].to(self.device)

        # Load or build ANN
        if self.ann_path.exists() and not rebuild_ann:
            logger.info(f"Loading ANN from {self.ann_path}")
            self.ann = IntentANNIndex.load(self.ann_path)
        else:
            if data_path is None:
                raise ValueError("data_path is required to build ANN index for the first time.")
            logger.info("Building ANN index from scratch...")
            self.ann = IntentANNIndex(dim=self.intent_centroids.size(1), backend=ann_backend)
            self._build_ann(data_path)
            self.ann.save(self.ann_path)
            logger.info(f"ANN index saved to {self.ann_path}")

        # Intent prior weights (regulatory priority)
        self.intent_prior = {
            "allergen_warning": 1.5,
            "usage_instructions": 1.2,
            "net_quantity": 1.1,
            "ingredients": 1.0,
            "product_descriptor": 0.9,
            "nutrition_claim": 0.9,
            "quality_claim": 0.9,
            "contact_information": 0.8,
        }

    def _build_ann(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info("Encoding training data for ANN...")
        for intent in self.intent_list:
            if intent not in raw_data:
                continue
            tiers = raw_data[intent]
            for tier_name in ["gold", "silver"]:
                texts = list(set(tiers.get(tier_name, [])))
                if texts:
                    embs = self.model.encode(
                        texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=128, show_progress_bar=False
                    )
                    self.ann.build_intent_tier(intent, tier_name, embs, texts)

    def _structural_gate(self, text: str) -> Optional[str]:
        """Apply regex/structural hard-gates for high-priority intents before ANN."""
        if _ALLERGEN_REGEX.search(text):
            return "allergen_warning"
        if _USAGE_REGEX.search(text):
            return "usage_instructions"
        if _URL_RE.search(text):
            return "contact_information"
        if _UNIT_LIKE_RE.search(text):
            return "net_quantity"
        return None

    def predict_batch(self, texts: List[str], field_name: Optional[str] = None) -> List[Dict]:
        if not texts:
            return []

        results = []
        with torch.no_grad():
            embeddings = self.model.encode(
                texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=64
            ).to(self.device)

            for idx, text in enumerate(texts):
                try:
                    lang = detect(text)
                except Exception:
                    lang = "unknown"

                structural = compute_structural_features(text)
                gated_intent = self._structural_gate(text)
                candidates = []

                # Stage 1: centroid similarity
                sims = F.cosine_similarity(
                    embeddings[idx].unsqueeze(0), self.intent_centroids, dim=1
                ) / self.temperature

                for i, intent in enumerate(self.intent_list):
                    if field_name and intent != field_name:
                        continue
                    base_score = float(sims[i].item())

                    # Structural gating shortcut
                    if gated_intent == intent:
                        candidates.append({
                            "intent": intent,
                            "tier": "structural_gate",
                            "raw_score": base_score,
                            "tier_bonus": self.gold_bonus,
                            "final_score": base_score + self.gold_bonus + self.intent_prior.get(intent, 1.0),
                            "nearest_example": None
                        })
                        continue

                    # Stage 2: ANN retrieval
                    ann_hits = self.ann.search(intent, embeddings[idx], top_k=1)
                    if not ann_hits:
                        candidates.append({
                            "intent": intent,
                            "tier": "centroid_only",
                            "raw_score": base_score,
                            "tier_bonus": 0.0,
                            "final_score": base_score * self.intent_prior.get(intent, 1.0),
                            "nearest_example": None
                        })
                        continue

                    hit_text, hit_score, tier = ann_hits[0]
                    tier_bonus = self.gold_bonus if tier == "gold" else self.silver_bonus

                    # Penalize structural mismatch
                    hit_struct = compute_structural_features(hit_text)
                    mismatch_penalty = 0.0
                    for key in ["num_digits", "num_units", "num_urls"]:
                        if structural.get(key, 0) != hit_struct.get(key, 0):
                            mismatch_penalty -= 0.05

                    final_score = base_score + tier_bonus + mismatch_penalty
                    final_score *= self.intent_prior.get(intent, 1.0)

                    candidates.append({
                        "intent": intent,
                        "tier": tier,
                        "raw_score": base_score,
                        "tier_bonus": tier_bonus,
                        "final_score": final_score,
                        "nearest_example": {
                            "text": hit_text,
                            "score": hit_score,
                            "tier": tier,
                        },
                    })

                # Pick the highest final score
                best = max(candidates, key=lambda x: x["final_score"])
                best["structural"] = structural
                best["detected_lang"] = lang
                results.append(best)

        return results


def save_predictions(texts: List[str], predictions: List[Dict], output_dir: Path, filename: str = "predictions.json"):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_samples": len(texts),
        "results": [{"input_text": t, "prediction": p} for t, p in zip(texts, predictions)],
    }
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Predictions saved to: {output_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # PATHS CONFIGURATION
    MODEL_DIR = r"D:\GDS\Adithya\copy_classification\food_intent_embedder_v2_1"
    TRAINING_DATA_PATH = r"D:\GDS\Adithya\copy_classification\Datasets\training_data_gold_silver_FINAL.json"
    OUTPUT_DIR = Path("./output")

    # Initialize inferencer
    inferencer = IntentInferencer(
        model_dir=MODEL_DIR,
        data_path=TRAINING_DATA_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.85,
        gold_bonus=0.08,
        silver_bonus=0.0,
        rebuild_ann=False
    )

    # Test Data
    sample_texts = [
        "सामग्री: चीनी, पानी, साइट्रिक एसिड, आम का गूदा", 
        "Nach dem Öffnen gekühlt aufbewahren.", 
        "Poids net: 500g e", 
        "Dilute one part syrup with six parts water.", 
        "Azeite de Oliva Extra Virgem", 
        "उपयोग करने से पहले अच्छी तरह हिलाएं।", 
        "http://www.pastafresca.it/ricette", 
        "Kann Spuren von Erdnüssen und Schalenfrüchten enthalten.", 
        "Ingrédients : Farine de blé, huile de palme, sel, levure.", 
        "น้ำหนักสุทธิ 150 กรัม", 
        "सीधे धूप से दूर, ठंडी और सूखी जगह पर रखें।", 
        "Traces: May contain soy and milk derivatives.", 
        "Chauffer au micro-ondes pendant 2 minutes.", 
        "Bio-Vollmilchschokolade", 
        "Conservare in luogo fresco e asciutto.", 
        "Ingredientes: Leche desnatada, azúcar, fermentos lácticos.", 
        "मसाला चाय पाउडर", 
        "Conteúdo líquido 750 ml", 
        "https://www.organic-farming.org/safety-data", 
        "चेतावनी: इसमें ग्लूटेन और मेवे हो सकते हैं."
    ]

    preds = inferencer.predict_batch(sample_texts)

    # Console preview
    for text, p in zip(sample_texts, preds):
        print(f"\nText: {text[:40]}...")
        print(f" >> Intent: {p['intent']} (Score: {p['final_score']:.4f} | Tier: {p['tier']})")
        if p['nearest_example']:
            print(f"    Nearest ({p['nearest_example']['tier']}): {p['nearest_example']['text'][:60]}...")

    # Save
    save_predictions(texts=sample_texts, predictions=preds, output_dir=OUTPUT_DIR)

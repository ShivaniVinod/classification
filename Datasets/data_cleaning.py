import json
import random
import re
from collections import defaultdict, Counter
from pathlib import Path
import pyarrow.dataset as ds

# ============================================================
# CONFIG & ARCHITECTURAL CONSTANTS
# ============================================================
PARQUET_PATH = Path("food.parquet")
OUTPUT_JSON = Path("training_data_gold_silver_FINAL.json")
DIAGNOSTICS_JSON = Path("dataset_diagnostics.json")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_PER_FIELD = 7500
MIN_PER_LANG = 100 
GOLD_PROMOTION_THRESHOLD = 500 

LANGUAGES = [
    "en","fr","de","es","it","nl","pl","pt","sv","bg","ro","fi","ru","nb","cs",
    "th","da","hr","hu","ar","el","ja","ca","sr","sl","sk","tr","lt","zh","et",
    "lv","uk","id","he","vi","is","la","in","ko","sq","ka","ms","bs","fa","bn",
    "gl","kk","mk","nn","hi","aa","uz","so","af","eu"
]

# ============================================================
# REFINED MULTILINGUAL HEURISTICS
# ============================================================
SENTENCE_END_RE = re.compile(r"[.!?。؟]")
DIGIT_RE = re.compile(r"\d")
MEASURE_RE = re.compile(r"\d+(\.\d+)?\s?(g|kg|ml|l|oz|lb|cl|dl|pcs|servings)\b", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
EMAIL_RE = re.compile(r"\S+@\S+")

def is_sentence_like(text: str) -> bool:
    return len(text.split()) >= 4 and SENTENCE_END_RE.search(text) is not None

def is_list_like(text: str) -> bool:
    return len(text) >= 15 and ("," in text or ":" in text or "[" in text)

def is_address_like(text: str) -> bool:
    return 5 < len(text) < 100 and bool(DIGIT_RE.search(text))

def contains_contact(text: str) -> bool:
    return bool(URL_RE.search(text) or EMAIL_RE.search(text) or "tel:" in text.lower())

# ============================================================
# SEMANTIC CONTRACTS: MAPPED TO YOUR SCHEMA
# ============================================================
SEMANTIC_CONTRACTS = {
    "product_name": {
        "column": "product_name",
        "multilang": True,
        "tier_policy": "gold_silver",
        "gold_rule": lambda t: 1 <= len(t.split()) <= 10,
        "silver_rule": lambda t: True,
    },
    "net_quantity": {
        "column": "quantity",
        "multilang": False,
        "tier_policy": "gold_silver",
        "gold_rule": lambda t: bool(MEASURE_RE.search(t)),
        "silver_rule": lambda t: bool(DIGIT_RE.search(t)),
    },
    "ingredients": {
        "column": "ingredients_text",
        "multilang": True,
        "tier_policy": "gold_silver",
        "gold_rule": is_list_like,
        "silver_rule": lambda t: len(t.split()) >= 3,
    },
    "allergen_statement": {
        "column": "allergens_tags", # Found in your schema
        "multilang": False,
        "tier_policy": "gold_silver",
        "gold_rule": lambda t: len(t) > 5,
        "silver_rule": lambda t: True,
    },
    "address": {
        "column": "manufacturing_places",
        "multilang": False,
        "tier_policy": "gold_silver",
        "gold_rule": is_address_like,
        "silver_rule": lambda t: len(t) > 4,
    },
    "storage_instructions": {
        "column": "packaging_text",
        "multilang": True,
        "tier_policy": "gold_silver",
        "gold_rule": is_sentence_like,
        "silver_rule": lambda t: len(t.split()) >= 3,
    },
    "nutrient_claims": {
        "column": "labels", # String field in your schema
        "multilang": False,
        "tier_policy": "gold_silver",
        "gold_rule": lambda t: 2 <= len(t.split()) <= 12,
        "silver_rule": lambda t: True,
    },
    "marketing_claims": {
        "column": "generic_name", # Struct list in your schema
        "multilang": True,
        "tier_policy": "gold_silver",
        "gold_rule": lambda t: 3 <= len(t.split()) <= 15,
        "silver_rule": lambda t: True,
    },
    "usage_instructions": {
        "column": "serving_size", # String in your schema
        "multilang": False,
        "tier_policy": "gold_silver",
        "gold_rule": lambda t: len(t.split()) >= 3,
        "silver_rule": lambda t: True,
    },
    "warning_statement": {
        "column": "traces_tags", # List of strings in your schema
        "multilang": False,
        "tier_policy": "gold_silver",
        "gold_rule": lambda t: len(t) > 4,
        "silver_rule": lambda t: True,
    },
    "website": {
        "column": "link",
        "multilang": False,
        "tier_policy": "gold_only",
        "gold_rule": lambda t: bool(URL_RE.search(t)),
    },
}

# ============================================================
# EXTRACTION LOGIC
# ============================================================
dataset = ds.dataset(PARQUET_PATH, format="parquet")
# Use a set to avoid duplicate column requests
unique_cols = list(set(["lang"] + [cfg["column"] for cfg in SEMANTIC_CONTRACTS.values()]))

scanner = dataset.scanner(columns=unique_cols, batch_size=10_000)

results = {field: {"gold": defaultdict(list), "silver": defaultdict(list)} for field in SEMANTIC_CONTRACTS}
diagnostics = {"field_language_coverage": defaultdict(Counter), "tier_counts": defaultdict(Counter)}

for batch in scanner.to_batches():
    cols = {name: batch.column(name) for name in batch.schema.names}
    for i in range(batch.num_rows):
        row_lang = cols["lang"][i].as_py()
        if row_lang not in LANGUAGES: continue

        for field, cfg in SEMANTIC_CONTRACTS.items():
            raw = cols[cfg["column"]][i].as_py()
            texts = []

            # 1. Handle Multilang Structs (e.g., product_name, generic_name)
            if cfg["multilang"] and isinstance(raw, list):
                texts = [e["text"].strip() for e in raw if isinstance(e, dict) and e.get("lang") == row_lang and e.get("text")]
            
            # 2. Handle Plain Strings (e.g., quantity, serving_size)
            elif isinstance(raw, str):
                texts = [raw.strip()]
            
            # 3. Handle List of Tags/Strings (e.g., allergens_tags, traces_tags)
            elif isinstance(raw, list):
                texts = [str(item).strip() for item in raw if item]
            
            for text in texts:
                if not text: continue
                diagnostics["field_language_coverage"][field][row_lang] += 1
                
                is_gold = cfg.get("gold_rule", lambda x: False)(text)
                is_silver = cfg.get("silver_rule", lambda x: False)(text)

                if is_gold and cfg["tier_policy"] != "silver_only":
                    results[field]["gold"][row_lang].append(text)
                    diagnostics["tier_counts"][field]["gold"] += 1
                elif is_silver and cfg["tier_policy"] != "gold_only":
                    results[field]["silver"][row_lang].append(text)
                    diagnostics["tier_counts"][field]["silver"] += 1
# ============================================================
# DERIVED FIELD: CONTACT INFORMATION (POST EXTRACTION)
# ============================================================

results["contact_information"] = {
    "gold": defaultdict(list),
    "silver": defaultdict(list),
}

diagnostics["tier_counts"]["contact_information"] = Counter()

# We only derive from fields that actually exist in parquet
SOURCE_FIELDS = [
    "storage_instructions",  # derived from packaging_text
    "address",               # derived from manufacturing_places
    "website",               # derived from link
    "marketing_claims",      # often contains URLs
]

for source_field in SOURCE_FIELDS:
    if source_field not in results:
        continue

    # Silver → derived silver
    for lang, texts in results[source_field]["silver"].items():
        for t in texts:
            if contains_contact(t):
                results["contact_information"]["silver"][lang].append(t)
                diagnostics["tier_counts"]["contact_information"]["silver"] += 1

    # Gold → derived gold
    for lang, texts in results[source_field]["gold"].items():
        for t in texts:
            if contains_contact(t):
                results["contact_information"]["gold"][lang].append(t)
                diagnostics["tier_counts"]["contact_information"]["gold"] += 1

# ============================================================
# RECOVERY & BALANCING
# ============================================================
for field in results:
    gold_count = diagnostics["tier_counts"][field]["gold"]
    if gold_count < GOLD_PROMOTION_THRESHOLD:
        for lang in results[field]["silver"]:
            silver_texts = results[field]["silver"][lang]
            promo_count = max(1, len(silver_texts) // 5)
            promoted = silver_texts[:promo_count]
            results[field]["gold"][lang].extend(promoted)
            diagnostics["tier_counts"][field]["gold"] += len(promoted)

def priority_sample(lang_map, target):
    all_samples = []
    for lang, texts in lang_map.items():
        if len(texts) <= MIN_PER_LANG:
            all_samples.extend(texts)
    
    current_count = len(all_samples)
    if current_count < target:
        remaining_pool = [t for lang, texts in lang_map.items() if len(texts) > MIN_PER_LANG for t in texts]
        random.shuffle(remaining_pool)
        needed = target - current_count
        all_samples.extend(remaining_pool[:needed])
    
    random.shuffle(all_samples)
    return all_samples[:target]

final_output = {
    field: {
        "gold": priority_sample(tiers["gold"], TARGET_PER_FIELD),
        "silver": priority_sample(tiers["silver"], TARGET_PER_FIELD),
    }
    for field, tiers in results.items()
}

OUTPUT_JSON.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
DIAGNOSTICS_JSON.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n--- FINAL AUDIT ---")
for field, data in final_output.items():
    print(f"{field:22s} | gold: {len(data['gold']):5d} | silver: {len(data['silver']):5d}")
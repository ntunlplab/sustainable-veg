import os
import json
import pickle
from rank_bm25 import BM25Okapi

from path_utils import load_config
from utils import tokenize_zh, normalize_text


# ---------------------------------------------------------
# Parse SASB metric file: {code: "topic。metric"}
# ---------------------------------------------------------
def load_sasb_labels(metrics_file):
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_list = json.load(f)

    labels = []
    for m in metrics_list:
        code = m.get("code", "")
        topic = m.get("topic", "")
        text = m.get("metric", "").strip()
            
        combined_text = f"{topic}。{text}" if topic != "一般資訊" else text  
        combined_text = normalize_text(combined_text)
        labels.append((code, combined_text))

    return labels


# ---------------------------------------------------------
# Encode SASB text into sparse embeddings
# ---------------------------------------------------------
def encode_sasb(sasb_labels):
    tokenized_corpus = [tokenize_zh(text) for _, text in sasb_labels]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# ---------------------------------------------------------
# Build a single SASB database
# ---------------------------------------------------------
def build_single_database(metrics_file, config):
    category = os.path.basename(metrics_file).split(".")[0]
    print(f"\n=== {category} ===")

    # Load SASB metrics → labels
    sasb_labels = load_sasb_labels(metrics_file)

    for code, text in sasb_labels:
        print(f"{code}: {text}")

    # Encode SASB texts
    bm25 = encode_sasb(sasb_labels)
    # Save
    out_dir = os.path.join(config["paths"]["sasb_index"], category)
    os.makedirs(out_dir, exist_ok=True)

    data = {
        "bm25": bm25,
        "documents": sasb_labels, 
    }
    with open(os.path.join(out_dir, config["paths"]["vector_data"]), "wb") as f:
        pickle.dump(data, f)

    print(f"{category} database built successfully.")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    config = load_config()

    # Build DB for each SASB metric file
    metrics_dir = config["paths"]["sasb_metrics"]
    for file_name in os.listdir(metrics_dir):
        metrics_file = os.path.join(metrics_dir, file_name)
        build_single_database(metrics_file, config)


if __name__ == "__main__":
    main()

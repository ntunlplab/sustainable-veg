import os
import json
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from path_utils import load_config


# ---------------------------------------------------------
# Parse SASB metric file: {code: "topic。metric"}
# ---------------------------------------------------------
def load_sasb_labels(metrics_file):
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_list = json.load(f)

    labels = {}
    for m in metrics_list:
        code = m.get("code", "")
        topic = m.get("topic", "")
        metric_text = m.get("metric", "")
        labels[code] = f"{topic}。{metric_text}"

    return labels


# ---------------------------------------------------------
# Encode SASB text into embedding vectors
# ---------------------------------------------------------
def encode_sasb(model, sasb_labels):
    vector_data = []

    for code, text in sasb_labels.items():
        vec = model.encode(text, normalize_embeddings=True)
        vector_data.append({
            "code": code,
            "text": text,
            "vector": vec
        })

    return vector_data


# ---------------------------------------------------------
# Build FAISS index from embeddings
# ---------------------------------------------------------
def build_faiss_index(vector_data):
    embedding_dim = vector_data[0]["vector"].shape[0]
    index = faiss.IndexFlatIP(embedding_dim)

    vectors = np.array([item["vector"] for item in vector_data], dtype="float32")
    index.add(vectors)

    return index


# ---------------------------------------------------------
# Save FAISS index + vectors to disk
# ---------------------------------------------------------
def save_database(out_dir, index, vector_data, index_filename, vector_filename):
    os.makedirs(out_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(out_dir, index_filename))
    with open(os.path.join(out_dir, vector_filename), "wb") as f:
        pickle.dump(vector_data, f)


# ---------------------------------------------------------
# Build a single SASB database
# ---------------------------------------------------------
def build_single_database(model, metrics_file, config):
    category = os.path.basename(metrics_file).split(".")[0]
    print(f"\n=== {category} ===")

    # Load SASB metrics → labels
    sasb_labels = load_sasb_labels(metrics_file)

    for code, text in sasb_labels.items():
        print(f"{code}: {text}")

    # Encode SASB texts
    vector_data = encode_sasb(model, sasb_labels)
    print(f"Total embeddings: {len(vector_data)}")

    # Build FAISS index
    index = build_faiss_index(vector_data)

    # Save
    out_dir = os.path.join(config["paths"]["sasb_database"], category)
    save_database(
        out_dir=out_dir,
        index=index,
        vector_data=vector_data,
        index_filename=config["paths"]["faiss_index"],
        vector_filename=config["paths"]["vector_data"]
    )

    print(f"SASB database built successfully, total vectors: {index.ntotal}")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    config = load_config()

    # Load embedding model
    model = SentenceTransformer(config["model"]["name"])

    # Build DB for each SASB metric file
    metrics_dir = config["paths"]["sasb_metrics"]
    for file_name in os.listdir(metrics_dir):
        metrics_file = os.path.join(metrics_dir, file_name)
        build_single_database(model, metrics_file, config)


if __name__ == "__main__":
    main()

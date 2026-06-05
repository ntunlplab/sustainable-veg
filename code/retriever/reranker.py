import argparse
import os
import json
import csv
import pickle
import yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from pdf_extract_region import mark_pdf_regions
import tqdm


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# Load FAISS index + vector data for a given SASB report
# ---------------------------------------------------------W
def load_faiss_database(base_path, sasb_report, index_name, vector_name):
    db_dir = os.path.join(base_path, sasb_report)
    index = faiss.read_index(os.path.join(db_dir, index_name))
    with open(os.path.join(db_dir, vector_name), "rb") as f:
        vectors = pickle.load(f)
    return index, vectors


# ---------------------------------------------------------
# Process a single OCR page: perform dense retrieval and produce a list of detection results
# ---------------------------------------------------------
def process_page(model, reranker, index, vector_data, ocr_objects, threshold, top_k):
    targets = []

    for obj in ocr_objects:
        
        if args.ocr == "olm":
            text = obj.get("text", "").strip()
        else:
            text = obj.get("content", "").strip()
        if not text:
            continue

        # ---------- Stage 1: Dense Retrieval ----------
        query_vec = model.encode(text, normalize_embeddings=True)
        query_vec = np.array([query_vec], dtype="float32")

        distances, indices = index.search(query_vec, top_k)

        cand_indices = indices[0]
        cand_scores = distances[0]

        # 組 candidate metric texts
        candidates = []
        for idx in cand_indices:
            candidates.append(vector_data[idx])  # {code, text}

        # ---------- Stage 2: Reranker ----------
        pairs = [(cand["text"], text) for cand in candidates]
        rerank_scores = reranker.predict(pairs)

        best_i = int(np.argmax(rerank_scores))
        best_score = float(rerank_scores[best_i])
        best_code = candidates[best_i]["code"]

        # print(f"[RERANK] score={best_score:.4f} code={best_code}")

        if best_score < threshold:
            continue

        bbox = obj.get("bbox_pdf")
        if bbox:
            bbox_str = ",".join(map(str, bbox))
            targets.append(f"{bbox_str}:{best_code}")

    return targets



# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main(args):
    config = load_config()

    # 1. Load embedding model
    model = SentenceTransformer(config["model"]["name"])
    # Load reranker (stage 2)
    reranker = CrossEncoder(config["reranker"]["name"])

    # 2. Load training data list
    # if args.test:
    #     data_path = config["paths"]["test_data"]
    # else:
    #     data_path = config["paths"]["train_data"]
    test_path = config["paths"]["test_data"]
    train_path = config["paths"]["train_data"]
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    # merge test and train data
    # data = train_data + test_data
    if args.test:
        data = test_data
    else:
        data = train_data

    # 3. Prepare output CSV
    csv_path = config["paths"]["results_output"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["ID", "TARGET"])
        writer.writeheader()

        # 4. Process each sample
        for item in tqdm.tqdm(data):
            sample_id = item["id"]
            page = item["page"]
            esg_file = item["esg_report"]
            sasb_report = item["sasb_report"].split(".")[0]

            # Load FAISS & vector data for the SASB group
            index, vector_data = load_faiss_database(
                base_path=config["paths"]["sasb_database"],
                sasb_report=sasb_report,
                index_name=config["paths"]["faiss_index"],
                vector_name=config["paths"]["vector_data"],
            )

            # Load OCR JSON
            # if args.test:
            #     esg_file = os.path.basename(esg_file)
            
            esg_file = os.path.basename(esg_file)
            
            report_folder = os.path.splitext(esg_file)[0]
            
            if args.ocr == "olm":
                ocr_path = f"{config['paths']['ocr_output']}/{report_folder}/{page}/output.json"
            else:
                ocr_path = f"{config['paths']['ocr_output']}/{report_folder}/{page}/chunks.json"

            if not os.path.exists(ocr_path):
                print(f"[WARN] Missing OCR file: {ocr_path}")
                writer.writerow({"ID": sample_id, "TARGET": "NONE"})
                continue

            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            # print(f"\n=== Processing {esg_file} | page {page} ===")

            if args.ocr == "olm":
                ocr_objects = ocr_data.get("objects", [])
            else:
                ocr_objects = ocr_data

            targets = process_page(
                model=model,
                reranker=reranker,
                index=index,
                vector_data=vector_data,
                # ocr_objects=ocr_data.get("objects", []),
                ocr_objects=ocr_objects,
                threshold=config["reranker"]["threshold"],
                top_k=config["reranker"]["top_k"]
            )


            # Write to CSV
            target_str = ";".join(targets) if targets else "NONE"
            writer.writerow({"ID": sample_id, "TARGET": target_str})

            # Visualize result if any target exists
            if targets:
                pdf_path = os.path.join(config["paths"]["esg_reports_pdf"], esg_file)
                output_dir = os.path.join(config["paths"]["marked_results_output"], report_folder)
                os.makedirs(output_dir, exist_ok=True)

                out_img = os.path.join(output_dir, f"{page}.png")
                mark_pdf_regions(pdf_path, page, targets, out_img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Use inference mode")
    parser.add_argument("--ocr", default="olm", help="OCR method: olm or chandra")
    args = parser.parse_args()

    main(args)

import argparse
import os
import json
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

from path_utils import load_config
from utils import load_faiss_index , mark_pdf_regions


# ---------------------------------------------------------
# Process a single OCR page: perform dense retrieval and produce a list of detection results
# ---------------------------------------------------------
def process_page(model, index, vector_data, ocr_objects, threshold):
    targets = []

    for obj in ocr_objects:
        text = obj.get("text", "").strip()
        if not text:
            continue

        query_vec = model.encode(text, normalize_embeddings=True)
        query_vec = np.array([query_vec], dtype="float32")

        # Retrieve top-5 candidates
        distances, indices = index.search(query_vec, 5)
        top_idx = indices[0][0]
        top_score = distances[0][0]

        print(f"top-score: {top_score:.4f}")

        if top_score > threshold:
            bbox = obj.get("bbox_pdf")
            if bbox:
                bbox_str = ",".join(map(str, bbox))
                targets.append(f"{bbox_str}:{vector_data[top_idx]['code']}")

    return targets


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main(args):
    config = load_config()

    # 1. Load embedding model
    model = SentenceTransformer(config["model"]["name"])

    # 2. Load training data list
    if args.test:
        data_path = config["paths"]["test_data"]
    else:
        data_path = config["paths"]["train_data"]
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Prepare output CSV
    csv_path = config["paths"]["results_output"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["ID", "TARGET"])
        writer.writeheader()

        # 4. Process each sample
        for item in data:
            sample_id = item["id"]
            page = item["page"]
            esg_file = item["esg_report"]
            sasb_report = item["sasb_report"].split("/")[-1].replace("SASB-", "").replace(".pdf", "")

            # Load FAISS & vector data for the SASB group
            index, vector_data = load_faiss_index(
                base_path=config["paths"]["sasb_database"],
                sasb_report=sasb_report,
                index_name=config["paths"]["faiss_index"],
                vector_name=config["paths"]["vector_data"],
            )

            # Load OCR JSON
            if args.test:
                esg_file = os.path.basename(esg_file)
            report_folder = os.path.splitext(esg_file)[0]
            ocr_path = f"{config['paths']['ocr_output']}/{report_folder}/{page}/output.json"

            if not os.path.exists(ocr_path):
                print(f"[WARN] Missing OCR file: {ocr_path}")
                writer.writerow({"ID": sample_id, "TARGET": "NONE"})
                continue

            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            print(f"\n=== Processing {esg_file} | page {page} ===")

            targets = process_page(
                model=model,
                index=index,
                vector_data=vector_data,
                ocr_objects=ocr_data.get("objects", []),
                threshold=config["threshold"]
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
    args = parser.parse_args()

    main(args)
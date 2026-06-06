import os
import argparse
import json
import csv
import numpy as np
from bs4 import BeautifulSoup
import time

from path_utils import load_config
from utils import tokenize_zh, normalize_text, load_bm25_index, mark_pdf_regions


# ---------------------------------------------------------
# OCR page processing (olm)
# ---------------------------------------------------------
def process_page_o(bm25, documents, ocr_path, threshold=19):
    targets = []

    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    for chunk in ocr_data["objects"]:
        # text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        text = chunk.get("text", "").strip()
        if not text:
            continue

        query_tokens = tokenize_zh((normalize_text(text)))   
        scores = bm25.get_scores(query_tokens)

        # scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        # top_idx = int(np.argmax(scores_norm))
        # top_score = float(scores_norm[top_idx])

        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])

        print(f"top-score: {top_score:.4f}")

        if top_score > threshold:
            bbox = chunk.get("bbox_pdf")
            if bbox:
                bbox_str = ",".join(map(str, bbox))
                targets.append(f"{bbox_str}:{documents[top_idx][0]}")  # code

    return targets


# ---------------------------------------------------------
# OCR page processing (Chandra)
# ---------------------------------------------------------
def process_page_c(bm25, documents, ocr_path, threshold=13):
    targets = []

    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    for chunk in ocr_data:
        if chunk["label"] not in ["Section-Header", "Text", "Table", "Figure", "Footnote"]:
            continue

        text = chunk["content"]

        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator="\n")

        query_tokens = tokenize_zh((normalize_text(text)))   
        scores = bm25.get_scores(query_tokens)

        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])

        print(f"top-score: {top_score:.4f}")

        if top_score > threshold:
            bbox = chunk.get("bbox_pdf")
            if bbox:
                bbox_str = ",".join(map(str, bbox))
                targets.append(f"{bbox_str}:{documents[top_idx][0]}")  # code

    return targets


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main(args):
    config = load_config()
    ocr_method = "chandra" if args.chandra else "olm"

    data_path = config["paths"]["data"]

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    csv_path = config["paths"]["results_output"].replace("{strategy}", "bm25")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["ID", "TARGET"])
        writer.writeheader()

        start_time = time.time()
        for item in data:
            sample_id = item["id"]
            page = item["page"]
            esg_file = os.path.basename(item["esg_report"])
            report_folder = os.path.splitext(esg_file)[0]
            sasb_report = item["sasb_report"].split("/")[-1].replace("SASB-", "").replace(".pdf", "")

            bm25, documents = load_bm25_index(
                config["paths"]["sasb_index"],
                sasb_report,
                config["paths"]["vector_data"],
            )

            print(f"\n=== Processing {esg_file} | page {page} ===")
            if ocr_method == "olm":
                ocr_path = f"{config['paths']['ocr_output']}/{report_folder}/{page}/output.json"
                targets = process_page_o(bm25, documents, ocr_path)
            elif ocr_method == "chandra":
                ocr_path = f"{config['paths']['chandra_ocr_output']}/{report_folder}/{page}/output.json"
                targets = process_page_c(bm25, documents, ocr_path)

            target_str = ";".join(targets) if targets else "NONE"
            writer.writerow({"ID": sample_id, "TARGET": target_str})

            # Visualization: GT + Prediction
            pdf_path = os.path.join(config["paths"]["esg_reports_pdf"], esg_file)
            out_dir = os.path.join(
                config["paths"]["marked_results_output"].replace("{strategy}", "bm25"),
                report_folder
            )
            os.makedirs(out_dir, exist_ok=True)
            out_img = os.path.join(out_dir, f"{page}.png")

            gt_boxes = item.get("label", [])
            mark_pdf_regions(pdf_path, page, gt_boxes=gt_boxes, pred_boxes=targets, output_path=out_img)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"[Total Time]: {total_time:.2f}s")
        print(f"[Average Time]: {total_time/len(data):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chandra", action="store_true", help="Use Chandra OCR instead of olmOCR (default)")
    args = parser.parse_args()

    main(args)

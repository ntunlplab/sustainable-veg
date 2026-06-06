import argparse
import os
import json
import csv

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

from path_utils import load_config
from utils import load_faiss_index, mark_pdf_regions


# ---------------------------------------------------------
# Path normalization helpers
# ---------------------------------------------------------
def normalize_esg_file(esg_report: str) -> str:
    """
    Accept:
      - reports/NPC_全國加油站.pdf
      - data/reports/NPC_全國加油站.pdf
      - NPC_全國加油站.pdf

    Return:
      - NPC_全國加油站.pdf
    """
    return os.path.basename(esg_report)


def normalize_report_folder(esg_report: str) -> str:
    """
    OCR output folder name should be PDF stem only.
    Example:
      reports/NPC_全國加油站.pdf -> NPC_全國加油站
    """
    return os.path.splitext(normalize_esg_file(esg_report))[0]


def normalize_sasb_report(sasb_report: str) -> str:
    """
    Important:
    build_database.py uses the metrics filename as database folder name.

    If metrics file is:
      retriever/metrics/SASB-EM-RM.json

    Then database folder is:
      SASB-EM-RM

    Therefore DO NOT remove 'SASB-'.

    Accept:
      - sasb/SASB-EM-RM.pdf
      - data/sasb/SASB-EM-RM.pdf
      - SASB-EM-RM.pdf

    Return:
      - SASB-EM-RM
    """
    return os.path.splitext(os.path.basename(sasb_report))[0]


# ---------------------------------------------------------
# OCR object helpers
# ---------------------------------------------------------
def get_ocr_text(obj, ocr_method: str) -> str:
    """
    olmOCR:
      obj["text"]

    Chandra:
      obj["content"]

    Fallback:
      try both.
    """
    if ocr_method == "olm":
        return str(obj.get("text", "")).strip()
    if ocr_method == "chandra":
        return str(obj.get("content", "")).strip()

    return str(obj.get("text", obj.get("content", ""))).strip()


def load_ocr_objects(ocr_path: str, ocr_method: str):
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    if ocr_method == "olm":
        return ocr_data.get("objects", [])

    # Chandra chunks.json is usually a list
    if isinstance(ocr_data, list):
        return ocr_data

    # fallback if someone stores Chandra chunks in a dict
    return ocr_data.get("objects", ocr_data.get("chunks", []))


def build_ocr_path(config, report_folder: str, page: int, ocr_method: str) -> str:
    if ocr_method == "olm":
        filename = "output.json"
    elif ocr_method == "chandra":
        filename = "chunks.json"
    else:
        raise ValueError(f"Unknown OCR method: {ocr_method}")

    return os.path.join(
        config["paths"]["ocr_output"],
        report_folder,
        str(page),
        filename,
    )


# ---------------------------------------------------------
# Dense-only retrieval
# ---------------------------------------------------------
def process_page_dense_only(
    model,
    index,
    vector_data,
    ocr_objects,
    threshold,
    ocr_method,
):
    targets = []

    for obj in ocr_objects:
        text = get_ocr_text(obj, ocr_method)
        if not text:
            continue

        bbox = obj.get("bbox_pdf")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        query_vec = model.encode(text, normalize_embeddings=True)
        query_vec = np.array([query_vec], dtype="float32")

        distances, indices = index.search(query_vec, 1)
        top_idx = int(indices[0][0])
        top_score = float(distances[0][0])

        print(f"[DENSE] score={top_score:.4f} code={vector_data[top_idx]['code']}")

        if top_score > threshold:
            code = vector_data[top_idx]["code"]
            bbox_str = ",".join(map(str, bbox))
            targets.append(f"{bbox_str}:{code}")

    return list(dict.fromkeys(targets))


# ---------------------------------------------------------
# Dense retrieval + CrossEncoder reranker
# ---------------------------------------------------------
def process_page_with_reranker(
    model,
    reranker,
    index,
    vector_data,
    ocr_objects,
    threshold,
    top_k,
    ocr_method,
):
    targets = []

    for obj in ocr_objects:
        text = get_ocr_text(obj, ocr_method)
        if not text:
            continue

        bbox = obj.get("bbox_pdf")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        # ---------- Stage 1: Dense Retrieval ----------
        query_vec = model.encode(text, normalize_embeddings=True)
        query_vec = np.array([query_vec], dtype="float32")

        distances, indices = index.search(query_vec, top_k)

        cand_indices = indices[0]
        candidates = []

        for idx in cand_indices:
            idx = int(idx)
            if idx < 0 or idx >= len(vector_data):
                continue
            candidates.append(vector_data[idx])  # {"code": ..., "text": ..., "vector": ...}

        if not candidates:
            continue

        # ---------- Stage 2: CrossEncoder Reranker ----------
        # bge-reranker convention: (query, passage)
        # query = OCR text, passage = SASB metric description
        pairs = [(text, cand["text"]) for cand in candidates]

        try:
            rerank_scores = reranker.predict(pairs)
        except Exception as e:
            print(f"[WARN] Reranker failed: {e}")
            continue

        best_i = int(np.argmax(rerank_scores))
        best_score = float(rerank_scores[best_i])
        best_code = candidates[best_i]["code"]

        print(f"[RERANK] score={best_score:.4f} code={best_code}")

        if best_score < threshold:
            continue

        bbox_str = ",".join(map(str, bbox))
        targets.append(f"{bbox_str}:{best_code}")

    # deduplicate while preserving order
    return list(dict.fromkeys(targets))


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main(args):
    config = load_config()

    # 1. Load embedding model
    model = SentenceTransformer(config["model"]["name"])

    # 2. Optional reranker
    reranker = None
    if args.rerank:
        reranker_name = config.get("reranker", {}).get(
            "name",
            "BAAI/bge-reranker-v2-m3",
        )
        print(f"[INFO] Loading reranker: {reranker_name}")
        reranker = CrossEncoder(reranker_name)

    # 3. Load merged data
    data_path = config["paths"]["data"]
    print(f"[INFO] Loading merged data: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 4. Output CSV
    csv_path = config["paths"]["results_output"]
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # 5. Threshold / top-k
    dense_threshold = float(config.get("threshold", 0.2))

    reranker_cfg = config.get("reranker", {})
    reranker_threshold = float(reranker_cfg.get("threshold", 0.0))
    reranker_top_k = int(reranker_cfg.get("top_k", 5))

    print(f"[INFO] OCR method: {args.ocr}")
    print(f"[INFO] Rerank enabled: {args.rerank}")
    if args.rerank:
        print(f"[INFO] Reranker top_k={reranker_top_k}, threshold={reranker_threshold}")
    else:
        print(f"[INFO] Dense threshold={dense_threshold}")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["ID", "TARGET"])
        writer.writeheader()

        for item in tqdm(data):
            sample_id = str(item["id"])
            page = int(item["page"])

            esg_file = normalize_esg_file(item["esg_report"])
            report_folder = normalize_report_folder(item["esg_report"])
            sasb_report = normalize_sasb_report(item["sasb_report"])

            print(
                f"\n=== Processing ID={sample_id} | "
                f"{esg_file} | page {page} | {sasb_report} ==="
            )

            # Load FAISS database.
            # sasb_report keeps 'SASB-' because build_database.py saves folder by metrics filename.
            try:
                index, vector_data = load_faiss_index(
                    base_path=config["paths"]["sasb_database"],
                    sasb_report=sasb_report,
                    index_name=config["paths"]["faiss_index"],
                    vector_name=config["paths"]["vector_data"],
                )
            except Exception as e:
                print(f"[WARN] Failed to load FAISS DB for {sasb_report}: {e}")
                writer.writerow({"ID": sample_id, "TARGET": "NONE"})
                continue

            # Load OCR
            ocr_path = build_ocr_path(
                config=config,
                report_folder=report_folder,
                page=page,
                ocr_method=args.ocr,
            )

            if not os.path.exists(ocr_path):
                print(f"[WARN] Missing OCR file: {ocr_path}")
                writer.writerow({"ID": sample_id, "TARGET": "NONE"})
                continue

            try:
                ocr_objects = load_ocr_objects(ocr_path, args.ocr)
            except Exception as e:
                print(f"[WARN] Failed to load OCR file {ocr_path}: {e}")
                writer.writerow({"ID": sample_id, "TARGET": "NONE"})
                continue

            # Retrieval
            if args.rerank:
                targets = process_page_with_reranker(
                    model=model,
                    reranker=reranker,
                    index=index,
                    vector_data=vector_data,
                    ocr_objects=ocr_objects,
                    threshold=reranker_threshold,
                    top_k=reranker_top_k,
                    ocr_method=args.ocr,
                )
            else:
                targets = process_page_dense_only(
                    model=model,
                    index=index,
                    vector_data=vector_data,
                    ocr_objects=ocr_objects,
                    threshold=dense_threshold,
                    ocr_method=args.ocr,
                )

            # Write CSV
            target_str = ";".join(targets) if targets else "NONE"
            writer.writerow({"ID": sample_id, "TARGET": target_str})

            # Visualization
            if targets:
                pdf_path = os.path.join(config["paths"]["esg_reports_pdf"], esg_file)
                output_dir = os.path.join(
                    config["paths"]["marked_results_output"],
                    report_folder,
                )
                os.makedirs(output_dir, exist_ok=True)

                out_img = os.path.join(output_dir, f"{page}.png")

                try:
                    mark_pdf_regions(pdf_path, page, targets, out_img)
                except Exception as e:
                    print(f"[WARN] Failed to draw marked PDF regions: {e}")

    print(f"\n[DONE] Saved results to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ocr",
        default="olm",
        choices=["olm", "chandra"],
        help="OCR method: olm uses output.json/text; chandra uses chunks.json/content",
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable dense retrieval + CrossEncoder reranker",
    )

    args = parser.parse_args()
    main(args)
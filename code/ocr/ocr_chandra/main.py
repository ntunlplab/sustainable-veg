import os
import json
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_chunks
import fitz
from PIL import Image, ImageDraw


# ------------------ 模型載入 ------------------
model_name = "datalab-to/chandra"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
).eval()
model.processor = AutoProcessor.from_pretrained(model_name)

# ------------------ 全域參數 ------------------
MAX_SIZE = 1536
DPI = 200

# ------------------ 建立 tasks ------------------
def build_tasks_from_train_test(
    report_dir: str,
    train_json_path: str,
    test_json_path: str,
) -> List[Tuple[str, str, int]]:
    """
    從 train.json 與 test.json 讀取所有 esg_report + page 組合，
    回傳唯一 (pdf_path, pdf_name, page_idx) list
    """
    pairs_set = set()

    def _add_from_file(json_path: str):
        if not os.path.exists(json_path):
            print(f"[MAIN] JSON file not found: {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[MAIN] JSON file {json_path} is not a list, skip.")
            return

        for item in data:
            if not isinstance(item, dict):
                continue
            esg_report = item.get("esg_report")
            page = item.get("page")
            if esg_report is None or page is None:
                continue

            try:
                page_idx = int(page)
            except Exception:
                continue
            if page_idx <= 0:
                continue

            # 處理 pdf_path
            if isinstance(esg_report, str):
                if esg_report.startswith("data/reports/"):
                    esg_report = esg_report[len("data/reports/"):]
                elif esg_report.startswith("reports/"):
                    esg_report = esg_report[len("reports/"):]
                pdf_path = os.path.join(report_dir, esg_report)
            else:
                continue

            pairs_set.add((pdf_path, page_idx))
            print(f"[MAIN] Found task: {pdf_path} page {page_idx}")

    _add_from_file(train_json_path)
    _add_from_file(test_json_path)

    tasks: List[Tuple[str, str, int]] = []
    for pdf_path, page_idx in sorted(pairs_set):
        pdf_name = os.path.basename(pdf_path)
        tasks.append((pdf_path, pdf_name, page_idx))

    print(f"[MAIN] Built {len(tasks)} unique (pdf, page) tasks from train/test.")
    return tasks

# ------------------ bbox 還原 ------------------
def restore_bbox(chunks, orig_size, resized_size):
    ow, oh = orig_size
    rw, rh = resized_size
    sx = ow / rw
    sy = oh / rh

    restored = []
    for c in chunks:
        x1, y1, x2, y2 = c["bbox"]
        restored.append({
            **c,
            "bbox": [round(x1 * sx, 2), round(y1 * sy, 2), round(x2 * sx, 2), round(y2 * sy, 2)]
        })
    return restored

# ------------------ image bbox 轉 pdf bbox ------------------
def image_to_pdf_bbox(bbox, dpi=200):
    scale = 72 / dpi
    return [round(v * scale, 6) for v in bbox]


# ------------------ 主處理函式 ------------------
def process_task(pdf_path, pdf_name, page_index, output_base="./ocr_output"):

    pdf_dir = Path(output_base) / Path(pdf_name).stem / str(page_index)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {pdf_name} page {page_index} ===")

    doc = fitz.open(pdf_path)
    page = doc[page_index-1]  # 1-based

    pix = page.get_pixmap(dpi=DPI)
    orig_w, orig_h = pix.width, pix.height
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # Resize
    scale = MAX_SIZE / max(orig_w, orig_h)
    resized = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)

    # Run Chandra
    batch = [BatchInputItem(image=resized, prompt_type="ocr_layout")]
    result = generate_hf(batch, model)[0]
    chunks_resized = parse_chunks(result.raw, resized)

    # bbox 還原到原始 image (200dpi) 座標
    chunks_img = restore_bbox(chunks_resized, (orig_w, orig_h), resized.size)

    # 轉成 PDF 72 dpi 座標
    chunks_pdf = []
    for c in chunks_img:
        pdf_bbox = image_to_pdf_bbox(c["bbox"], dpi=DPI)
        chunks_pdf.append({
            **c,
            "bbox_pdf": pdf_bbox     # 新增 PDF 座標
        })


    # Save JSON
    with open(pdf_dir / "output.json", "w", encoding="utf-8") as f:
        json.dump(chunks_pdf, f, ensure_ascii=False, indent=2)


    # 畫原圖 bbox
    draw = ImageDraw.Draw(img)
    for c in chunks_img:
        draw.rectangle(c["bbox"], outline="red", width=3)
    img.save(pdf_dir / "bbox.png")

    print("✔ Completed:", pdf_name, "page", page_index)


# ------------------ 主程式 ------------------
def main():
    REPORT_DIR = "../../../data/reports"
    TRAIN_JSON = "../../../data/Sustainable-VEG.json"
    TEST_JSON = "../../../data/Sustainable-VEG.json"
    OUTPUT_BASE = "./ocr_output"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    tasks = build_tasks_from_train_test(REPORT_DIR, TRAIN_JSON, TEST_JSON)
    for pdf_path, pdf_name, page_index in tasks:
        process_task(pdf_path, pdf_name, page_index, output_base=OUTPUT_BASE)

if __name__ == "__main__":
    main()
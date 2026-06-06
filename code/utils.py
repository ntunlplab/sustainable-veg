import os
import re
import jieba
import faiss
import pickle

from pathlib import Path
import io
from PIL import Image
import fitz  # PyMuPDF


# ---------------------------------------------------------
# Normalize Chinese text
# ---------------------------------------------------------
_PUNCT_MAP = {
    ",": "，",
    ":": "：",
    ";": "；",
    "?": "？",
    "!": "！",
    "(": "（",
    ")": "）",
}

def normalize_text(text):
    if not text:
        return ""

    # Unify punctuation to full-width
    text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)

    # Remove other symbols
    text = re.sub(r"[®]", "", text)

    # Remove purely numbered or single-letter brackets, e.g., (1), (a)
    text = re.sub(r"（\s*(?:\d|[a-zA-Z])\s*）", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", "", text).strip()

    # Ensure the text ends with a period
    if not text.endswith("。"):
        text += "。"

    return text


def tokenize_zh(text):
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)  # Keep Chinese characters, English letters, and digits
    tokens = list(jieba.cut(text))
    
    return tokens


# ---------------------------------------------------------
# Load FAISS index + vector data
# ---------------------------------------------------------
def load_faiss_index(base_path, sasb_report, index_name, vector_name):
    db_dir = os.path.join(base_path, sasb_report)
    index = faiss.read_index(os.path.join(db_dir, index_name))
    with open(os.path.join(db_dir, vector_name), "rb") as f:
        vectors = pickle.load(f)
    return index, vectors


# ---------------------------------------------------------
# Load BM25 index + document data
# ---------------------------------------------------------
def load_bm25_index(base_path, sasb_report, vector_name):
    db_dir = os.path.join(base_path, sasb_report)

    with open(os.path.join(db_dir, vector_name), "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    documents = data["documents"]

    return bm25, documents


# ---------------------------------------------------------
# PDF Labeling/Visualization Tool (GT + Prediction Integration)
# ---------------------------------------------------------
def _parse_single_box(spec: str):
    if ":" in spec:
        coords_part, label = spec.split(":", 1)
        label = label.strip() or "BOX"
    else:
        coords_part, label = spec, "BOX"

    coords = coords_part.split(",")
    if len(coords) != 4:
        raise ValueError("座標格式錯誤，應為 'x1,y1,x2,y2[:label]'")

    x1, y1, x2, y2 = map(float, coords)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("座標無效：需滿足 x1<x2 且 y1<y2")
    if min(x1, y1, x2, y2) < 0:
        raise ValueError("座標不能為負數")

    return x1, y1, x2, y2, label


# ---------------------------------------------------------
# Draw bounding box with text label on a PDF page
# ---------------------------------------------------------
def _draw_box_with_label(
    page, x1, y1, x2, y2, label,
    box_color=(1,0,0),
    label_bg=(1,0.85,0.85),
    text_color=(0,0,0),
    box_width=2,
    fontsize=10,
    padding=3):

    rect = fitz.Rect(x1, y1, x2, y2)
    page.draw_rect(rect, color=box_color, width=box_width)

    est_text_w = max(40, int(0.6 * fontsize * len(label)) + 2 * padding)
    est_text_h = fontsize + 2 * padding

    page_rect = page.rect
    label_top_y = y1 - est_text_h
    if label_top_y < page_rect.y0:
        label_top_y = y1

    label_rect = fitz.Rect(x1, label_top_y, x1 + est_text_w, label_top_y + est_text_h)

    if label_rect.x1 > page_rect.x1:
        shift = page_rect.x1 - label_rect.x1
        label_rect = label_rect + (shift, 0, shift, 0)
    if label_rect.y1 > page_rect.y1:
        dy = page_rect.y1 - label_rect.y1
        label_rect = label_rect + (0, dy, 0, dy)

    page.draw_rect(label_rect, color=box_color, fill=label_bg, width=0.5)
    text_x = label_rect.x0 + padding
    text_y = label_rect.y0 + padding + fontsize * 0.8
    page.insert_text((text_x, text_y), label, fontsize=fontsize, color=text_color)


# ---------------------------------------------------------
# Mark PDF regions with GT (Green) and Predictions (Red)
# ---------------------------------------------------------
def mark_pdf_regions(pdf_path: str, page_num: int, gt_boxes=None, pred_boxes=None, output_path=None):
    gt_boxes = gt_boxes or []
    pred_boxes = pred_boxes or []

    pdf_document = fitz.open(pdf_path)
    try:
        page = pdf_document[page_num-1]

        # 畫 GT（綠色）
        for spec in gt_boxes:
            x1, y1, x2, y2, label = _parse_single_box(spec)
            _draw_box_with_label(page, x1, y1, x2, y2, label,
                                 box_color=(0,1,0), label_bg=(0.85,1,0.85))
        # 畫 Prediction（紅色）
        for spec in pred_boxes:
            x1, y1, x2, y2, label = _parse_single_box(spec)
            _draw_box_with_label(page, x1, y1, x2, y2, label,
                                 box_color=(1,0,0), label_bg=(1,0.85,0.85))

        pix = page.get_pixmap(matrix=fitz.Matrix(1,1))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        if output_path is None:
            stem = Path(pdf_path).stem
            output_path = f"{stem}_p{page_num}.png"
        img.save(output_path, "PNG")
        return output_path
    finally:
        pdf_document.close()
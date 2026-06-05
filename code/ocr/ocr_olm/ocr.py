import os
import json
import base64
import subprocess
from io import BytesIO
from multiprocessing import Process, Manager, Queue
from typing import List, Dict, Any, Tuple
import time

import torch
from PIL import Image
import fitz 
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def pdf_page_to_base64(pdf_path: str, page_idx: int, dpi: int = 150) -> Tuple[str, Tuple[int, int]]:
    """
    Render a single PDF page to PNG (in-memory) and return (base64_str, (width, height)).
    page_idx is 1-based.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_idx - 1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return b64, (pix.width, pix.height)
    finally:
        doc.close()


def convert_bbox_to_pdf_points(
    bbox_px: List[float],
    page_rect: fitz.Rect,
    render_w: int,
    render_h: int,
) -> List[float]:
    """
    Convert bbox in rendered image pixels (top-left origin) → PDF coordinate space (points),
    still using top-left origin because pdf_extract_region.py assumes that.
    """
    if len(bbox_px) != 4:
        return [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = bbox_px
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    sx = page_rect.width / float(render_w)
    sy = page_rect.height / float(render_h)

    x1p = x1 * sx
    x2p = x2 * sx
    y1p = y1 * sy
    y2p = y2 * sy

    return [x1p, y1p, x2p, y2p]


def clean_and_fix_bbox(bbox: List[float], page_rect: fitz.Rect) -> List[float]:
    """
    Ensure bbox is valid: sorted, clipped to page, and has positive size.
    If invalid, return [0,0,0,0].
    """
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0.0, min(x1, page_rect.width))
    x2 = max(0.0, min(x2, page_rect.width))
    y1 = max(0.0, min(y1, page_rect.height))
    y2 = max(0.0, min(y2, page_rect.height))

    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return [0.0, 0.0, 0.0, 0.0]

    return [x1, y1, x2, y2]

def parse_model_json(raw: str) -> Dict[str, Any] | None:
    """
    Try to parse model output as strict JSON.
    """
    if not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    return data


def draw_boxes_on_pdf(
    pdf_path: str,
    page_idx: int,
    objects: List[Dict[str, Any]],
    output_png: str,
) -> None:
    """
    Call existing data/pdf_extract_region.py to draw boxes.
    This script MUST NOT be modified (per your requirement).
    """
    cmd = ["python", "../../../data/pdf_extract_region.py", pdf_path, str(page_idx)]

    for obj in objects:
        bbox = obj.get("bbox_pdf", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue
        label = str(obj.get("id", ""))
        box_arg = f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}:{label}"
        cmd += ["--box", box_arg]

    cmd += ["-o", output_png]
    subprocess.run(cmd)

def extract_table_and_chart_images(
    pdf_path: str,
    page_idx: int,
    objects: List[Dict[str, Any]],
    output_dir: str,
    opened_pdfs: Dict[str, fitz.Document],
) -> None:
    
    if not objects:
        return

    if pdf_path not in opened_pdfs:
        opened_pdfs[pdf_path] = fitz.open(pdf_path)
    doc = opened_pdfs[pdf_path]
    page = doc[page_idx - 1]

    pic_idx = 1
    for obj in objects:
        obj_type = str(obj.get("type", "")).lower()
        if obj_type not in ("table", "chart", "table_cell"):
            continue

        bbox_pdf = obj.get("bbox_pdf")
        if not isinstance(bbox_pdf, (list, tuple)) or len(bbox_pdf) != 4:
            continue

        x1, y1, x2, y2 = bbox_pdf
        if x2 <= x1 or y2 <= y1:
            continue

        rect = fitz.Rect(x1, y1, x2, y2)
        try:
            pix = page.get_pixmap(clip=rect)
        except Exception:
            continue

        pic_path = os.path.join(output_dir, f"picture_{pic_idx}.png")
        try:
            pix.save(pic_path)
            pic_idx += 1
        except Exception:
            continue
        
        json_path = os.path.join(output_dir, f"picture_{pic_idx-1}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": obj.get("id"),
                "bbox_pdf": bbox_pdf,
                "type": obj_type,
            }, f, ensure_ascii=False, indent=2)



def worker_loop(
    gpu_id: int,
    task_queue: Queue,
    anchor_prompt_json: str,
    output_root: str,
    batch_size: int = 3,
    max_retries: int = 1,
) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    print(f"[GPU {gpu_id}] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-2-7B-1025",
        dtype=torch.bfloat16,
    ).eval().to(device)
    
    print(f"[GPU {gpu_id}] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast=False,
    )
    
    print(f"[GPU {gpu_id}] Worker started.")
    start_time = time.time()

    opened_pdfs: Dict[str, fitz.Document] = {}

    while True:
        batch_tasks = []
        for _ in range(batch_size):
            try:
                task = task_queue.get_nowait()
            except Exception:
                break
            if task is None:
                continue
            batch_tasks.append(task)

        if not batch_tasks:
            break

        texts: List[str] = []
        images: List[Image.Image] = []
        meta: List[Dict[str, Any]] = []

        for (pdf_path, pdf_name, page_idx) in batch_tasks:
            
            if pdf_path not in opened_pdfs:
                opened_pdfs[pdf_path] = fitz.open(pdf_path)
            pdf_doc = opened_pdfs[pdf_path]

            image_base64, (render_w, render_h) = pdf_page_to_base64(pdf_path, page_idx)
            page_rect = pdf_doc[page_idx - 1].rect
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

            prompt = anchor_prompt_json.format(width=render_w, height=render_h)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            texts.append(text)
            images.append(main_image)
            meta.append(
                {
                    "pdf_path": pdf_path,
                    "pdf_name": pdf_name,
                    "page_idx": page_idx,
                    "render_w": render_w,
                    "render_h": render_h,
                    "page_rect": page_rect,
                }
            )

        if not texts:
            continue

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=4096,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, prompt_len:]

        decoded_list = processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        for decoded, info in zip(decoded_list, meta):
            pdf_path = info["pdf_path"]
            pdf_name = info["pdf_name"]
            page_idx = info["page_idx"]
            render_w = info["render_w"]
            render_h = info["render_h"]
            page_rect = info["page_rect"]

            pdf_stem = os.path.splitext(os.path.basename(pdf_name))[0]
            
            print(f"[GPU {gpu_id}] Processing {pdf_name} page {page_idx}...")

            page_dir = os.path.join(output_root, pdf_stem, str(page_idx))
            os.makedirs(page_dir, exist_ok=True)

            raw_path = os.path.join(page_dir, "output_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(decoded)
                
            meta_path = os.path.join(page_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "render_w": render_w,
                    "render_h": render_h,
                    "page_width": float(page_rect.width),
                    "page_height": float(page_rect.height),
                }, f, ensure_ascii=False, indent=2)

            page_data = None
            attempt_raw = decoded
            for _ in range(max_retries):
                page_data = parse_model_json(attempt_raw)
                break 

            if page_data is None:
                print(f"[GPU {gpu_id}] JSON parse failed for {pdf_name} page {page_idx}. Saved raw only.")
                continue

            objects = page_data.get("objects", [])
            if not isinstance(objects, list):
                objects = []
                page_data["objects"] = objects

            valid_objects: List[Dict[str, Any]] = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                bbox = obj.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue

                bbox_pdf = convert_bbox_to_pdf_points(
                    bbox, page_rect, render_w, render_h
                )
                bbox_pdf = clean_and_fix_bbox(bbox_pdf, page_rect)
                if bbox_pdf == [0.0, 0.0, 0.0, 0.0]:
                    continue

                obj["bbox_pdf"] = bbox_pdf
                valid_objects.append(obj)

            page_data["objects"] = valid_objects

            json_path = os.path.join(page_dir, "output.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, ensure_ascii=False, indent=2)

            if valid_objects:
                bbox_png_path = os.path.join(page_dir, "bbox.png")
                draw_boxes_on_pdf(pdf_path, page_idx, valid_objects, bbox_png_path)

                extract_table_and_chart_images(
                    pdf_path,
                    page_idx,
                    valid_objects,
                    page_dir,
                    opened_pdfs,
                )

            cost_time = time.time() - start_time
            print(f"[GPU {gpu_id}] Processed {pdf_name} page {page_idx}: saved outputs at {page_dir}. Time has taken: {cost_time:.2f} seconds.")

    for doc in opened_pdfs.values():
        try:
            doc.close()
        except Exception:
            pass

    print(f"[GPU {gpu_id}] Worker finished.")
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    print(f"[GPU {gpu_id}] Total elapsed time: {total_elapsed:.2f}")


def build_tasks_from_train_test(
    report_dir: str,
    train_json_path: str,
    test_json_path: str,
) -> List[Tuple[str, str, int]]:
    
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

            if isinstance(esg_report, str):
                if esg_report.startswith("ocr/ocr_olm/data/reports/"):
                    pdf_path = esg_report
                elif esg_report.startswith("data/reports/"):
                    pdf_path = os.path.join("ocr/ocr_olm/", esg_report)
                elif esg_report.startswith("reports/"):
                    pdf_path = os.path.join("ocr/ocr_olm/data", esg_report)
                else:
                    pdf_path = os.path.join(report_dir, esg_report)
            else:
                continue

            todo_set = []
            todo_set.append((pdf_path, page_idx))
            
            if (pdf_path, page_idx) not in todo_set:
                continue
                
            print(f"[MAIN] Found task: {pdf_path} page {page_idx}")
            pairs_set.add((pdf_path, page_idx))
            
            print(f"[MAIN] Total tasks collected: {len(pairs_set)}")

    _add_from_file(train_json_path)
    # _add_from_file(test_json_path)

    tasks: List[Tuple[str, str, int]] = []
    for pdf_path, page_idx in sorted(pairs_set):
        pdf_name = os.path.basename(pdf_path)
        tasks.append((pdf_path, pdf_name, page_idx))

    print(f"[MAIN] Built {len(tasks)} unique (pdf, page) tasks from train/test.")
    return tasks

def main():
    print("=== OCR PROCESSING STARTED ===")
    REPORT_DIR = "../../../data/reports"
    TRAIN_JSON = "../../../data/Sustainable-VEG.json"
    TEST_JSON = "../../../data/Sustainable-VEG.json"
    OUTPUT_BASE = "./ocr_output"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    anchor_prompt_json = """
You are a precise OCR and layout extraction system.

The image you will analyze has:
- width: {width} pixels
- height: {height} pixels

The coordinate origin (0,0) is at the TOP-LEFT corner.

Output a JSON object **strictly matching this structure**:

{{
  "page": {{
    "width": {width},
    "height": {height}
  }},
  "objects": [
    {{
      "id": int,
      "text": "string",
      "bbox": [x1, y1, x2, y2],
      "type": "line" | "paragraph" | "header" | "table_cell"
    }}
  ]
}}

Guidelines:
- bbox MUST use these exact pixel dimensions.
- x increases to the right; y increases downward.
- Never normalize; always use raw pixel values.
- Extract all visible text.
- No hallucination.
- Output ONLY valid JSON parsable by json.loads().
- The response MUST begin with '{{' and end with '}}'.
- Remember to use \n for line breaks within text fields.
- Remove redundant \n or spaces in text fields.

Now analyze the following image:
"""

    tasks = build_tasks_from_train_test(REPORT_DIR, TRAIN_JSON, TEST_JSON)
    if not tasks:
        print("[MAIN] No tasks found from train/test JSON files. Exit.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPU detected.")

    print(f"[MAIN] Detected GPUs: {num_gpus}")
    print(f"[MAIN] Total tasks: {len(tasks)}")

    manager = Manager()
    task_queue: Queue = manager.Queue()

    # enqueue tasks
    for pdf_path, pdf_name, page_idx in tasks:
        if not os.path.exists(pdf_path):
            print(f"[MAIN] PDF not found, skip: {pdf_path}")
            continue
        task_queue.put((pdf_path, pdf_name, page_idx))
    print("[MAIN] All tasks enqueued.")

    batch_size = 1
    max_retries = 1

    processes: List[Process] = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker_loop,
            args=(
                gpu_id+2,
                task_queue,
                anchor_prompt_json,
                OUTPUT_BASE,
                batch_size,
                max_retries,
            ),
        )
        p.start()
        processes.append(p)
        print(f"[MAIN] Started worker on GPU {gpu_id}")

    for p in processes:
        p.join()

    print("=== ALL TASKS DONE ===")


if __name__ == "__main__":
    main()

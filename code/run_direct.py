import os
import json
import csv
import argparse
import time
import random
from pathlib import Path

from tqdm import tqdm
import fitz

from google import genai
from google.genai import types
from dotenv import load_dotenv

from path_utils import CODE_DIR, load_config


load_dotenv()
api_key = os.getenv("API_KEY")


# ---------------------------------------------------------
# Args
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="code/prompt/prompt_vlm.txt",
        help="prompt template file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="render dpi",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.8,
        help="fixed sleep after each page",
    )
    parser.add_argument(
        "--max_retry",
        type=int,
        default=16,
        help="max retries when rate-limited",
    )
    parser.add_argument(
        "--page_is_1_based",
        action="store_true",
        help="dataset page is 1-based",
    )
    parser.set_defaults(page_is_1_based=True)

    return parser.parse_args()


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


def normalize_sasb_report(sasb_report: str) -> str:
    """
    Important:
    Metrics files are named like:
      SASB-EM-RM.json

    So DO NOT remove 'SASB-'.

    Accept:
      - sasb/SASB-EM-RM.pdf
      - data/sasb/SASB-EM-RM.pdf
      - SASB-EM-RM.pdf

    Return:
      - SASB-EM-RM
    """
    return os.path.splitext(os.path.basename(sasb_report))[0]


# ---------------------------------------------------------
# Dataset / metrics
# ---------------------------------------------------------
def read_dataset_json(dataset_path):
    path = Path(dataset_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sasb_stringify(metrics_data):
    lines = []

    for m in metrics_data:
        # Expected keys: code, topic, metric, category, unit
        lines.append(
            f"- Code: {m['code']}; "
            f"Topic: {m.get('topic', '')}; "
            f"Description: {m.get('metric', '')}; "
            f"Category: {m.get('category', '')}; "
            f"Unit: {m.get('unit', '')}."
        )

    return "\n".join(lines)


def apply_prompt_template(template_path, metrics_data):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    return template.replace("{metrics_data}", sasb_stringify(metrics_data))


def load_metrics(metrics_dir, sasb_report):
    """
    dataset sasb_report:
      sasb/SASB-EM-RM.pdf
      SASB-EM-RM.pdf

    metrics file:
      SASB-EM-RM.json
    """
    base = normalize_sasb_report(sasb_report)
    metric_path = os.path.join(metrics_dir, f"{base}.json")

    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"metrics not found: {metric_path}")

    with open(metric_path, "r", encoding="utf-8") as f:
        return json.load(f), metric_path


# ---------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------
def render_pdf_page_to_png_bytes(pdf_path, page_index_0, dpi=120):
    """
    Returns:
      png_bytes, width_px, height_px, page_width_pt, page_height_pt

    The VLM should output bbox in rendered image pixel coordinates.
    Later we convert pixel bbox back to PDF point coordinates for scoring.
    """
    doc = fitz.open(pdf_path)

    try:
        if page_index_0 < 0 or page_index_0 >= doc.page_count:
            raise IndexError(
                f"page_index {page_index_0} out of range "
                f"(0..{doc.page_count - 1}) for {pdf_path}"
            )

        page = doc.load_page(page_index_0)
        page_rect = page.rect

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        png_bytes = pix.tobytes("png")

        return (
            png_bytes,
            pix.width,
            pix.height,
            page_rect.width,
            page_rect.height,
        )

    finally:
        doc.close()


# ---------------------------------------------------------
# Response parsing
# ---------------------------------------------------------
def parse_metrics_response(text):
    """
    Expected:
    <metrics>
    {"items":[{"bbox":[...],"metric":"..."}, ...]}
    </metrics>
    """
    if not isinstance(text, str):
        return None, "not_string"

    start_tag = "<metrics>"
    end_tag = "</metrics>"

    s = text.find(start_tag)
    e = text.find(end_tag)

    if s == -1 or e == -1 or e <= s:
        return None, "missing_tags"

    payload = text[s + len(start_tag):e].strip()

    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None, "bad_json"

    items = obj.get("items", [])
    if not isinstance(items, list):
        return None, "items_not_list"

    return items, None


def clamp_bbox(b, w, h):
    x1, y1, x2, y2 = b

    x1 = max(0, min(w - 1, float(x1)))
    y1 = max(0, min(h - 1, float(y1)))
    x2 = max(0, min(w - 1, float(x2)))
    y2 = max(0, min(h - 1, float(y2)))

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def pixel_bbox_to_pdf_bbox(bbox_px, width_px, height_px, page_width_pt, page_height_pt):
    """
    Convert rendered image pixel coordinates to PDF point coordinates.
    This is necessary because the score.py expects PDF coordinates.
    """
    x1, y1, x2, y2 = bbox_px

    sx = page_width_pt / float(width_px)
    sy = page_height_pt / float(height_px)

    return [
        x1 * sx,
        y1 * sy,
        x2 * sx,
        y2 * sy,
    ]


# ---------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------
def call_vision_bbox_metric_gemini(
    client,
    model,
    prompt,
    png_bytes,
    stats,
    max_retry=8,
):
    image_part = types.Part(
        inline_data=types.Blob(
            data=png_bytes,
            mime_type="image/png",
        )
    )
    text_part = types.Part(text=prompt)

    for attempt in range(max_retry):
        try:
            t0 = time.perf_counter()

            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[text_part, image_part],
                    )
                ],
            )

            t1 = time.perf_counter()
            stats["api_s"] += t1 - t0
            stats["calls"] += 1

            return resp.text or ""

        except Exception as e:
            stats["retries"] += 1

            wait = 1.0 + random.uniform(0.1, 0.3)
            stats["wait_s"] += wait

            print(f"[Gemini Retry] {e} → sleep {wait:.2f}s "
                  f"(attempt {attempt + 1}/{max_retry})")

            time.sleep(wait)

    raise RuntimeError("Gemini retry exceeded")


# ---------------------------------------------------------
# Output conversion
# ---------------------------------------------------------
def json_format_to_submission_csv(pred_format_json_path, csv_path):
    """
    pred_format_json:
      {
        "000": ["x1,y1,x2,y2:METRIC", ...] or ["NONE"]
      }

    CSV:
      ID,TARGET
    """
    with open(pred_format_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    csv_dir = os.path.dirname(str(csv_path))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "TARGET"])
        writer.writeheader()

        for idx, entries in data.items():
            cleaned = []

            for e in entries:
                e = str(e).strip()
                if e.upper() == "NONE":
                    continue
                if e.upper().endswith(":NONE"):
                    continue
                cleaned.append(e)

            target = "NONE" if len(cleaned) == 0 else ";".join(cleaned)
            writer.writerow({"ID": idx, "TARGET": target})


# ---------------------------------------------------------
# Main process
# ---------------------------------------------------------
def process(data, args, client, config):
    predictions_format = {}
    raw_responses = {}

    stats = {
        "wait_s": 0.0,
        "api_s": 0.0,
        "render_s": 0.0,
        "calls": 0,
        "retries": 0,
        "pages": 0,
    }

    t_total0 = time.perf_counter()

    for item in tqdm(data):
        instance_id = str(item["id"])

        page_int = int(item["page"])
        page_index = page_int - 1 if args.page_is_1_based else page_int

        esg_report_name = normalize_esg_file(item["esg_report"])
        pdf_path = os.path.join(
            config["paths"]["esg_reports_pdf"],
            esg_report_name,
        )

        metrics_data, _ = load_metrics(
            config["paths"]["sasb_metrics"],
            item["sasb_report"],
        )

        prompt = apply_prompt_template(args.prompt, metrics_data)

        t_r0 = time.perf_counter()
        (
            png_bytes,
            width_px,
            height_px,
            page_width_pt,
            page_height_pt,
        ) = render_pdf_page_to_png_bytes(
            pdf_path,
            page_index,
            dpi=args.dpi,
        )
        t_r1 = time.perf_counter()
        stats["render_s"] += t_r1 - t_r0

        resp_text = call_vision_bbox_metric_gemini(
            client=client,
            model=args.model,
            prompt=prompt,
            png_bytes=png_bytes,
            stats=stats,
            max_retry=args.max_retry,
        )

        raw_responses[instance_id] = resp_text

        items, err = parse_metrics_response(resp_text)

        if err is not None or items is None:
            predictions_format[instance_id] = ["NONE"]
        else:
            out = []

            for it in items:
                metric = str(it.get("metric", "")).strip()
                bbox = it.get("bbox", None)

                if not metric or metric.upper() == "NONE":
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                bbox_px = clamp_bbox(bbox, width_px, height_px)
                bbox_pdf = pixel_bbox_to_pdf_bbox(
                    bbox_px,
                    width_px=width_px,
                    height_px=height_px,
                    page_width_pt=page_width_pt,
                    page_height_pt=page_height_pt,
                )

                out.append(
                    f"{bbox_pdf[0]},{bbox_pdf[1]},{bbox_pdf[2]},{bbox_pdf[3]}:{metric}"
                )

            out = list(dict.fromkeys(out))
            predictions_format[instance_id] = out if out else ["NONE"]

        stats["pages"] += 1

        if args.throttle > 0:
            stats["wait_s"] += float(args.throttle)
            time.sleep(float(args.throttle))

    t_total1 = time.perf_counter()

    total_s = t_total1 - t_total0
    non_wait_s = total_s - stats["wait_s"]

    print("\n====== Timing Summary ======")
    print(f"Pages processed: {stats['pages']}")
    print(f"Total wall time: {total_s:.2f}s")
    print(f"  Wait time:     {stats['wait_s']:.2f}s")
    print(f"  Non-wait time: {non_wait_s:.2f}s")
    print(f"    - Render:    {stats['render_s']:.2f}s")
    print(f"    - API:       {stats['api_s']:.2f}s")
    print(f"Calls: {stats['calls']}, Retries: {stats['retries']}")

    if stats["pages"] > 0:
        print(
            f"Avg wall/page: {total_s / stats['pages']:.2f}s | "
            f"avg wait/page: {stats['wait_s'] / stats['pages']:.2f}s | "
            f"avg api/page: {stats['api_s'] / stats['pages']:.2f}s | "
            f"avg render/page: {stats['render_s'] / stats['pages']:.2f}s"
        )

    stats["total_wall_s"] = total_s
    stats["non_wait_s"] = non_wait_s

    return predictions_format, raw_responses, stats


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
def main():
    args = parse_args()
    config = load_config()

    if not api_key:
        raise RuntimeError("Missing API_KEY in .env")

    client = genai.Client(api_key=api_key)

    data = read_dataset_json(config["paths"]["data"])

    pred_format, raw, stats = process(
        data=data,
        args=args,
        client=client,
        config=config,
    )

    out_json = CODE_DIR / "predictions_format_gemini_pagebbox.json"
    out_raw = CODE_DIR / "raw_responses_gemini_pagebbox.json"
    out_csv = CODE_DIR / "submission_gemini_pagebbox.csv"
    out_timing = CODE_DIR / "timing_gemini_pagebbox.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pred_format, f, ensure_ascii=False, indent=2)

    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    with open(out_timing, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    json_format_to_submission_csv(out_json, out_csv)

    print("\nWrote:")
    print(" ", out_json)
    print(" ", out_raw)
    print(" ", out_csv)
    print(" ", out_timing)


if __name__ == "__main__":
    main()
# exp_page_bbox_full.py
# Usage example:
#   export OPENAI_API_KEY="YOUR_KEY"
#   python exp_page_bbox_full.py --split train --dpi 120 --model gpt-4o-mini --prompt prompt_page_bbox.txt
#
# Output:
#   {split}_predictions_format_{model}_pagebbox.json
#   {split}_raw_responses_{model}_pagebbox.json
#   {split}_submission_{model}_pagebbox.csv
#   {split}_timing_{model}_pagebbox.json

import os
import json
import csv
import argparse
import base64
import time
import random
from tqdm import tqdm

import fitz  # PyMuPDF
# import openai
# from openai import OpenAI
from google import genai
from google.genai import types

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="../metrics", help="path to metrics json files")
    # parser.add_argument("--metrics", type=str, default="../IRFinal-main/extract_sasb", help="path to metrics json files")
    parser.add_argument("--reports", type=str, default="../../data/reports", help="path to pdf report files")
    parser.add_argument("--dataset", type=str, default="../../data", help="path to dataset (train.json/test.json)")
    parser.add_argument("--prompt", type=str, default="prompt2-4.txt", help="prompt template file")
    parser.add_argument("--split", type=str, default="Sustainable-VEG", choices=["Sustainable-VEG", "Sustainable-VEG"])
    # parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview")

    parser.add_argument("--dpi", type=int, default=120, help="render dpi (lower = cheaper/faster)")
    parser.add_argument("--throttle", type=float, default=0.8, help="fixed sleep after each page (seconds)")
    parser.add_argument("--max_retry", type=int, default=16, help="max retries when rate-limited")
    parser.add_argument("--page_is_1_based", action="store_true", help="dataset page is 1-based (default True)")
    parser.set_defaults(page_is_1_based=True)
    return parser.parse_args()


def read_dataset_json(dataset_path, split):
    path = os.path.join(dataset_path, f"{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sasb_stringify(metrics_data):
    lines = []
    for m in metrics_data:
        # expected keys: code, topic, metric, category, unit
        lines.append(
            f"- Code: {m['code']}; Topic: {m['topic']}; Description: {m['metric']}; "
            f"Category: {m['category']}; Unit: {m['unit']}."
        )
        
        # expected keys: code, description
        # lines.append(f"- Code: {m['code']}; Description: {m['description']}.")
    return "\n".join(lines)


def apply_prompt_template(template_path, metrics_data):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("{metrics_data}", sasb_stringify(metrics_data))


def render_pdf_page_to_png_bytes(pdf_path, page_index_0, dpi=120):
    """
    Returns (png_bytes, width_px, height_px)
    bbox output from the model should be in this pixel coordinate space.
    """
    doc = fitz.open(pdf_path)
    if page_index_0 < 0 or page_index_0 >= doc.page_count:
        raise IndexError(f"page_index {page_index_0} out of range (0..{doc.page_count-1}) for {pdf_path}")

    page = doc.load_page(page_index_0)
    zoom = dpi / 72.0  # 72 dpi = 1x in PDF points
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    return png_bytes, pix.width, pix.height


def parse_metrics_response(text):
    """
    Expect:
    <metrics>
    {"items":[{"bbox":[...],"metric":"..."}, ...]}
    </metrics>
    """
    start_tag = "<metrics>"
    end_tag = "</metrics>"
    s = text.find(start_tag)
    e = text.find(end_tag)
    if s == -1 or e == -1:
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
    # ensure ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def load_metrics(metrics_dir, sasb_report):
    """
    dataset sasb_report could be 'SASB-EM-RM.pdf' or 'EM-RM.pdf'
    metrics file assumed: '{base}.json'
    """
    base = os.path.splitext(os.path.basename(sasb_report))[0]
    base = base.replace("SASB-", "")  # normalize
    metric_path = os.path.join(metrics_dir, f"{base}.json")
    # metric_path = os.path.join(metrics_dir, f"{base}.jsonl")
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"metrics not found: {metric_path}")
    with open(metric_path, "r", encoding="utf-8") as f:
        return json.load(f), metric_path
        # lines = f.readlines()
        # data = [json.loads(line) for line in lines]
        # return data, metric_path


def call_vision_bbox_metric(client, model, prompt, png_bytes, stats, max_retry=8):
    """
    stats will be updated:
      stats["wait_s"]: total sleep time due to rate limit + throttle (you add throttle elsewhere)
      stats["api_s"]: time spent inside the API call (excluding sleeps)
      stats["calls"]: successful calls count
      stats["retries"]: rate limit retry count
    """
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    for attempt in range(max_retry):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            }
                        ],
                    }
                ],
            )
            t1 = time.perf_counter()
            stats["api_s"] += (t1 - t0)
            stats["calls"] += 1
            return resp.choices[0].message.content

        except openai.RateLimitError as e:
            stats["retries"] += 1

            # Parse suggested wait time (ms) from the error message if present
            wait = 1.0
            msg = str(e)
            if "Please try again in" in msg:
                try:
                    wait_ms = float(msg.split("in")[1].split("ms")[0].strip())
                    wait = wait_ms / 1000.0
                except Exception:
                    pass

            # jitter
            wait += random.uniform(0.10, 0.30)
            stats["wait_s"] += wait
            print(f"[RateLimit] sleep {wait:.2f}s (attempt {attempt+1}/{max_retry})")
            time.sleep(wait)

    raise RuntimeError("Rate limit retry exceeded")

def call_vision_bbox_metric_gemini(
    client,
    model,
    prompt,
    png_bytes,
    stats,
    max_retry=8
):
    image_part = types.Part.from_bytes(
        data=png_bytes,
        mime_type="image/png"
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
            stats["api_s"] += (t1 - t0)
            stats["calls"] += 1

            return resp.text

        except Exception as e:
            stats["retries"] += 1
            wait = 1.0 + random.uniform(0.1, 0.3)
            stats["wait_s"] += wait
            print(f"[Gemini Retry] {e} → sleep {wait:.2f}s")
            time.sleep(wait)

    raise RuntimeError("Gemini retry exceeded")


def json_format_to_submission_csv(pred_format_json_path, csv_path):
    """
    pred_format_json: { "000": ["x1,y1,x2,y2:METRIC", ...] or ["NONE"], ... }
    csv TARGET expects:
      - if none: "NONE"
      - else join by ';'
    """
    with open(pred_format_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "TARGET"])
        w.writeheader()
        for idx, entries in data.items():
            # remove entries that end with :NONE or :None
            cleaned = []
            for e in entries:
                if e.strip().upper().endswith(":NONE"):
                    continue
                cleaned.append(e)

            if len(cleaned) == 0:
                target = "NONE"
            elif len(cleaned) == 1 and cleaned[0].strip().upper() == "NONE":
                target = "NONE"
            else:
                # If someone stored ["NONE", ...] accidentally
                cleaned = [e for e in cleaned if e.strip().upper() != "NONE"]
                target = "NONE" if len(cleaned) == 0 else ";".join(cleaned)

            w.writerow({"ID": idx, "TARGET": target})


def process(data, args, client):
    predictions_format = {}   # ID -> ["x1,y1,x2,y2:METRIC", ...] or ["NONE"]
    raw_responses = {}

    stats = {
        "wait_s": 0.0,     # sleeping time (rate-limit waits + throttle waits)
        "api_s": 0.0,      # time spent in API calls (excluding waits)
        "render_s": 0.0,   # time for PDF rendering
        "calls": 0,        # successful calls
        "retries": 0,      # rate-limit retries
        "pages": 0,        # processed items
    }

    t_total0 = time.perf_counter()

    for item in tqdm(data):
        instance_id = item["id"]

        # dataset page can be "1", 1, etc.
        page_val = item["page"]
        page_int = int(page_val)
        page_index = page_int - 1 if args.page_is_1_based else page_int
        
        esg_report_name = item["esg_report"]
        
        if esg_report_name.startswith("data/reports/"):
            esg_report_name = esg_report_name[len("data/reports/"):]

        pdf_path = os.path.join(args.reports, esg_report_name)
        metrics_data, _ = load_metrics(args.metrics, item["sasb_report"].replace(".pdf", ""))
        prompt = apply_prompt_template(args.prompt, metrics_data)

        # render
        t_r0 = time.perf_counter()
        png_bytes, w, h = render_pdf_page_to_png_bytes(pdf_path, page_index, dpi=args.dpi)
        t_r1 = time.perf_counter()
        stats["render_s"] += (t_r1 - t_r0)

        # call vision model with rate-limit retry tracking
        # resp_text = call_vision_bbox_metric(
        #     client, args.model, prompt, png_bytes, stats, max_retry=args.max_retry
        # )
        resp_text = call_vision_bbox_metric_gemini(
            client, args.model, prompt, png_bytes, stats, max_retry=args.max_retry
        )

        raw_responses[instance_id] = resp_text

        items, err = parse_metrics_response(resp_text)
        if err is not None or items is None:
            predictions_format[instance_id] = ["NONE"]
        else:
            out = []
            for it in items:
                metric = it.get("metric", "")
                bbox = it.get("bbox", None)

                if not metric or str(metric).strip().upper() == "NONE":
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                bbox = clamp_bbox(bbox, w, h)
                out.append(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}:{metric}")

            # dedup, keep order
            out = list(dict.fromkeys(out))
            predictions_format[instance_id] = out if len(out) > 0 else ["NONE"]

        stats["pages"] += 1

        # fixed throttle (count as wait)
        if args.throttle > 0:
            stats["wait_s"] += float(args.throttle)
            time.sleep(float(args.throttle))

    t_total1 = time.perf_counter()
    total_s = t_total1 - t_total0
    non_wait_s = total_s - stats["wait_s"]

    print("\n====== Timing Summary ======")
    print(f"Pages processed: {stats['pages']}")
    print(f"Total wall time: {total_s:.2f}s")
    print(f"  Wait time (sleep): {stats['wait_s']:.2f}s")
    print(f"  Non-wait time:     {non_wait_s:.2f}s")
    print(f"    - PDF render:    {stats['render_s']:.2f}s")
    print(f"    - API time:      {stats['api_s']:.2f}s")
    print(f"Calls succeeded: {stats['calls']}, RateLimit retries: {stats['retries']}")
    if stats["pages"] > 0:
        print(
            f"Avg wall/page: {total_s/stats['pages']:.2f}s | "
            f"avg wait/page: {stats['wait_s']/stats['pages']:.2f}s | "
            f"avg api/page: {stats['api_s']/stats['pages']:.2f}s | "
            f"avg render/page: {stats['render_s']/stats['pages']:.2f}s"
        )

    stats["total_wall_s"] = total_s
    stats["non_wait_s"] = non_wait_s
    return predictions_format, raw_responses, stats


def main():
    args = parse_args()

    # Use env var key (do NOT hardcode keys in code)
    api_key = "YOUR_KEY"  
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    # client = OpenAI(api_key=api_key)
    client = genai.Client(
        api_key=api_key
    )

    data = read_dataset_json(args.dataset, args.split)

    pred_format, raw, stats = process(data, args, client)

    out_json = f"{args.split}_predictions_format_gemini_pagebbox.json"
    out_raw = f"{args.split}_raw_responses_gemini_pagebbox.json"
    out_csv = f"{args.split}_submission_gemini_pagebbox.csv"
    out_timing = f"{args.split}_timing_gemini_pagebbox.json"

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

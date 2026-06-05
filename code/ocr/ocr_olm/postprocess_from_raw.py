import os
import json
import fitz
import subprocess


# ============================================================
# === SAME HELPERS FROM ocr.py ===============================
# ============================================================

def convert_bbox_to_pdf_points(bbox_px, page_rect, render_w, render_h):
    if len(bbox_px) != 4:
        return [0,0,0,0]

    try:
        x1, y1, x2, y2 = map(float, bbox_px)
    except:
        return [0,0,0,0]

    sx = page_rect.width  / float(render_w)
    sy = page_rect.height / float(render_h)

    return [
        x1 * sx,
        y1 * sy,
        x2 * sx,
        y2 * sy,
    ]


def clean_and_fix_bbox(bbox, page_rect):
    if len(bbox) != 4:
        return [0,0,0,0]

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except:
        return [0,0,0,0]

    # sort
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    # clip
    x1 = max(0, min(x1, page_rect.width))
    x2 = max(0, min(x2, page_rect.width))
    y1 = max(0, min(y1, page_rect.height))
    y2 = max(0, min(y2, page_rect.height))

    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return [0,0,0,0]

    return [x1, y1, x2, y2]


def parse_model_json(raw):
    try:
        return json.loads(raw)
    except:
        return None


# ============================================================
# DRAW BBOX
# ============================================================

def draw_boxes_on_pdf(pdf_path, page_idx, objects, output_png):
    cmd = ["python", "ocr/data/pdf_extract_region.py", pdf_path, str(page_idx)]
    for obj in objects:
        bbox = obj.get("bbox_pdf", [0,0,0,0])
        if len(bbox) != 4: continue
        x1,y1,x2,y2 = bbox
        if x2 <= x1 or y2 <= y1: continue
        cmd += ["--box", f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}:{obj.get('id','')}"]
    cmd += ["-o", output_png]
    subprocess.run(cmd)


# ============================================================
# EXTRACT REGION
# ============================================================

def extract_region_images(pdf_path, page_idx, objects, output_dir):
    doc = fitz.open(pdf_path)
    page = doc[page_idx - 1]

    pic_idx = 1
    for obj in objects:
        t = obj.get("type","").lower()
        if t not in ("table", "chart", "table_cell"): continue

        bbox = obj.get("bbox_pdf")
        if not bbox or len(bbox)!=4: continue

        x1,y1,x2,y2 = bbox
        if x2 <= x1 or y2 <= y1: continue

        rect = fitz.Rect(x1,y1,x2,y2)
        try:
            pix = page.get_pixmap(clip=rect)
        except:
            continue

        out_path = os.path.join(output_dir, f"picture_{pic_idx}.png")
        try:
            pix.save(out_path)
            pic_idx += 1
        except:
            continue
        
        json_path = os.path.join(output_dir, f"picture_{pic_idx-1}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": obj.get("id"),
                "bbox_pdf": bbox,
                "type": t
            }, f, ensure_ascii=False, indent=2)

    doc.close()


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page_dir(page_dir):

    raw_path  = os.path.join(page_dir, "output_raw.txt")
    meta_path = os.path.join(page_dir, "meta.json")
    json_path = os.path.join(page_dir, "output.json")

    if not os.path.exists(meta_path):
        print(f"[ERR] missing meta.json: {page_dir}")
        return

    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    render_w   = meta["render_w"]
    render_h   = meta["render_h"]
    page_rect  = fitz.Rect(0, 0, meta["page_width"], meta["page_height"])

    # ------------------------------------------------------------------
    # Step 1: If no output.json → parse raw to build it (do not overwrite)
    # ------------------------------------------------------------------
    if not os.path.exists(json_path):
        if not os.path.exists(raw_path):
            print(f"[SKIP] no raw + no json: {page_dir}")
            return

        raw = open(raw_path, "r", encoding="utf-8").read()
        data = parse_model_json(raw)
        if data is None:
            print(f"[FAIL] cannot parse json: {page_dir}")
            return

        if not isinstance(data.get("objects"), list):
            data["objects"] = []

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[STEP1] output.json created: {page_dir}")

    # ------------------------------------------------------------------
    # Step 2: For every object with bbox → recompute bbox_pdf (ALWAYS overwrite)
    # ------------------------------------------------------------------
    data = json.load(open(json_path, "r", encoding="utf-8"))
    objects = data.get("objects", [])

    updated = False

    for obj in objects:

        if "bbox" not in obj:
            continue

        bbox_px = obj["bbox"]
        if not isinstance(bbox_px, (list, tuple)) or len(bbox_px) != 4:
            continue

        # === Recompute bbox_pdf with the SAME logic as ocr.py ===
        bbox_pdf = convert_bbox_to_pdf_points(bbox_px, page_rect, render_w, render_h)
        bbox_pdf = clean_and_fix_bbox(bbox_pdf, page_rect)

        if bbox_pdf != [0,0,0,0]:
            obj["bbox_pdf"] = bbox_pdf
        else:
            # if invalid, remove bbox_pdf
            obj.pop("bbox_pdf", None)

        updated = True

    if updated:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[STEP2] bbox_pdf recomputed: {page_dir}")

    # ------------------------------------------------------------------
    # Draw & Extract Regions
    # ------------------------------------------------------------------

    pdf_name = page_dir.split(os.sep)[-2]
    pdf_path = f"../../../data/reports/{pdf_name}.pdf"
    page_idx = int(os.path.basename(page_dir))

    valid_objects = [o for o in objects if "bbox_pdf" in o]

    if valid_objects:
        draw_boxes_on_pdf(
            pdf_path, page_idx,
            valid_objects,
            os.path.join(page_dir, "bbox.png")
        )

        extract_region_images(
            pdf_path,
            page_idx,
            valid_objects,
            page_dir
        )

        print(f"[DONE] {page_dir}: bbox + region extracted")


# ============================================================
# MAIN WALKER
# ============================================================

def main():
    ROOT = "./ocr_output"

    for pdf_dir in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, pdf_dir)
        if not os.path.isdir(p):
            continue

        for pg in sorted(os.listdir(p)):
            page_dir = os.path.join(p, pg)
            if not os.path.isdir(page_dir):
                continue

            process_page_dir(page_dir)


if __name__ == "__main__":
    main()

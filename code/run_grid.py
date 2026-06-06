"""Predict SASB labels for test.json using a 50px alphabet/number grid with Gemini."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
import math
from tqdm import tqdm

from path_utils import load_config

load_dotenv()

@dataclass
class Record:
    """Minimal record used for test/train items."""

    id: str
    page: int
    company: str
    esg_report: str
    sasb_report: str
    labels: List[str]


class GridBox(BaseModel):
    """Grid-aligned bounding box using alphabet columns and numeric rows (model output)."""

    x1: str = Field(description="Left column letter (A, B, C...)")
    y1: int = Field(description="Top row number starting at 1")
    x2: str = Field(description="Right column letter (>= x1)")
    y2: int = Field(description="Bottom row number (>= y1)")
    code: str = Field(description="SASB metric code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict grid-based SASB labels with Gemini.")
    parser.add_argument("--grid_size", type=int, default=25, help="Grid size in pixels (default 25)")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", help="Gemini model name")
    parser.add_argument("--max_retries", type=int, default=1, help="Maximum number of retries for failed API calls (default 3)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF page rendering (default 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Gemini API (default 42)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay in seconds between API calls (default 0.3)")
    return parser.parse_args()


def load_metrics(metrics_path: str) -> List[Dict[str, Any]]:
    """Load SASB metrics from a JSON file (code + metric)."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metrics file: {e}")
    if not isinstance(metrics, list):
        raise ValueError(f"Metrics file should contain a JSON array, got {type(metrics)}")
    # Filter to only include metrics with code and metric fields
    filtered_metrics = []
    for metric in metrics:
        if isinstance(metric, dict) and "code" in metric and "metric" in metric:
            filtered_metrics.append(metric)
    return filtered_metrics


def render_pdf_page_to_pil(pdf_path: str, page_number: int, dpi: int = 150) -> Image.Image:
    """
    Render a single page from PDF to PIL RGB image at specified DPI.
    page_number is 1-based.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"Page {page_number} out of range for document {pdf_path}")
        page = doc[page_index]
        
        # Calculate scaling factor: PDF default is 72 DPI, so scale by dpi/72
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Set DPI metadata on the image
        img.info["dpi"] = (dpi, dpi)
        
        return img
    finally:
        doc.close()


def load_records(path: Path) -> List[Record]:
    """Load records from JSON file."""
    data = json.loads(path.read_text())
    records: List[Record] = []
    for item in data:
        records.append(
            Record(
                id=str(item.get("id")),
                page=int(item.get("page")),
                company=item.get("company", ""),
                esg_report=item.get("esg_report"),
                sasb_report=item.get("sasb_report"),
                labels=item.get("label", []),
            )
        )
    return records


def col_to_alpha(col_idx: int) -> str:
    """Convert zero-based column index to spreadsheet-style letters."""
    if col_idx < 0:
        raise ValueError("Column index must be non-negative")
    col_number = col_idx + 1
    letters = ""
    while col_number:
        col_number, rem = divmod(col_number - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def alpha_to_col(alpha: str) -> int:
    """Convert column letters (A, B, AA, AB...) to zero-based index."""
    s = alpha.strip().upper()
    if not s.isalpha():
        raise ValueError(f"Invalid column label: {alpha}")
    col = 0
    for ch in s:
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col - 1


def parse_label_to_pixel(label_str: str) -> Dict[str, Any]:
    """Parse 'x1,y1,x2,y2:code' into pixel-based dict with code."""
    coords, code = label_str.split(":")
    x1, y1, x2, y2 = [float(v) for v in coords.split(",")]
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "code": code}


def overlay_grid(image: Image.Image, grid_size: int = 25, dpi: int = 150) -> Image.Image:
    """Draw a grid with alphabet x-axis and numeric y-axis labels.
    """
    img = image.convert("RGBA")
    width, height = img.size
    
    # Scale grid size to match the high-DPI image
    scale = dpi / 72.0
    scaled_grid_size = grid_size * scale
    
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Use larger font size - scale with DPI for better visibility
    font_size = max(18, int(12 * scale))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (OSError, IOError):
            # Fallback to default font if system fonts not available
            font = ImageFont.load_default()
    line_color = (0, 0, 0, 255)  # Black
    label_color = (0, 0, 0, 255)  # Black

    num_cols = math.ceil(width / scaled_grid_size)
    num_rows = math.ceil(height / scaled_grid_size)

    for col in range(num_cols + 1):
        x = col * scaled_grid_size
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
        if col < num_cols:
            label = col_to_alpha(col)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = min(max(x + scaled_grid_size / 2 - tw / 2, 0), width - tw)
            draw.text((tx, 2), label, font=font, fill=label_color)

    for row in range(num_rows + 1):
        y = row * scaled_grid_size
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
        if row < num_rows:
            label = str(row + 1)
            bbox = draw.textbbox((0, 0), label, font=font)
            th = bbox[3] - bbox[1]
            ty = min(max(y + scaled_grid_size / 2 - th / 2, 0), height - th)
            draw.text((2, ty), label, font=font, fill=label_color)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    return combined


def image_to_part(image: Image.Image) -> types.Part:
    """Convert PIL image to Gemini part."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return types.Part.from_bytes(data=buf.read(), mime_type="image/png")


def resolve_report_path(report: str, reports_dir: str | None = None) -> Path:
    """Find the actual path to an ESG report PDF.

    Tries, in order: the path as-is, <reports_dir>/<basename>, and the
    legacy data/reports/<report> fallback.
    """
    path = Path(report)
    if path.exists():
        return path
    if reports_dir:
        cand = Path(reports_dir) / Path(report).name
        if cand.exists():
            return cand
    alt = Path("data/reports") / report
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Report not found: {report}")


def sasb_text_from_report(sasb_report: str, metrics_dir: str = "metrics",
                          max_items: int = 30) -> str:
    """Build a SASB metric reference text block from <metrics_dir>/*.json.

    sasb_report looks like 'sasb/SASB-EM-RM.pdf'; metrics files are named
    'EM-RM.json', so we strip the 'SASB-' prefix from the stem.
    """
    stem = Path(sasb_report).stem.replace("SASB-", "")
    metrics_path = Path(metrics_dir) / f"{stem}.json"
    try:
        metrics = load_metrics(str(metrics_path))
    except FileNotFoundError:
        return f"(Missing metrics file at {metrics_path})"
    lines = []
    for metric in metrics[:max_items]:
        code = metric.get("code", "")
        metric_desc = metric.get("metric", "")
        lines.append(f"{code}: {metric_desc}")
    return "\n".join(lines)


def build_prompt_text(sasb_text: str) -> str:
    """Instruction prompt for Gemini (expects grid letters/numbers)."""
    return f"""You are given a single ESG report page.
- X axis uses alphabetic columns starting at A, B, C, etc.
- Y axis uses numeric rows starting at 1, 2, 3, etc.
Return the grid-aligned boxes that fully covers each SASB region as JSON array objects with fields:
  x1 (letter), y1 (number), x2 (letter), y2 (number), code.
x1, y1 represents the left-top corner of the box, and x2, y2 represents the right-bottom corner of the box.
Boxes must not overlap with each other. Each box should cover a distinct, non-overlapping SASB region.
Only include regions that clearly correspond to SASB metrics in the reference below.
Don't return multiple neighboring regions with the same code, return the most representative one.
If no SASB regions are found, return an empty array [].

SASB reference:
{sasb_text}

Respond only with JSON for the image."""


def validate_grid_boxes(boxes: Optional[List[Dict[str, Any]]]) -> bool:
    """Validate that grid boxes have appropriate structure and values."""
    # None or empty list is valid (no boxes found)
    if boxes is None:
        return True
    if not isinstance(boxes, list):
        return False
    # Empty list is valid
    if len(boxes) == 0:
        return True
    
    for box in boxes:
        if not isinstance(box, dict):
            return False
        
        # Check required fields exist
        required_fields = ["x1", "y1", "x2", "y2", "code"]
        if not all(field in box for field in required_fields):
            return False
        
        x1 = box.get("x1", "")
        y1 = box.get("y1", 0)
        x2 = box.get("x2", "")
        y2 = box.get("y2", 0)
        code = box.get("code", "")
        
        # Check code is non-empty string
        if not isinstance(code, str) or not code.strip():
            return False
        
        # Check x1, x2 are valid column letters
        try:
            col1 = alpha_to_col(str(x1))
            col2 = alpha_to_col(str(x2))
        except (ValueError, TypeError, AttributeError):
            return False
        
        # Check y1, y2 are valid positive integers
        try:
            y1_int = int(y1)
            y2_int = int(y2)
            if y1_int < 1 or y2_int < 1:
                return False
        except (ValueError, TypeError):
            return False
        
        # Check x1 <= x2 (column-wise)
        if col1 > col2:
            return False
        
        # Check y1 <= y2 (row-wise)
        if y1_int > y2_int:
            return False
    
    # Check for overlapping boxes
    if len(boxes) > 1:
        for i in range(len(boxes)):
            box_i = boxes[i]
            try:
                col1_i = alpha_to_col(str(box_i.get("x1", "")))
                col2_i = alpha_to_col(str(box_i.get("x2", "")))
                row1_i = int(box_i.get("y1", 0)) - 1
                row2_i = int(box_i.get("y2", 0)) - 1
            except (ValueError, TypeError):
                continue
            
            for j in range(i + 1, len(boxes)):
                box_j = boxes[j]
                try:
                    col1_j = alpha_to_col(str(box_j.get("x1", "")))
                    col2_j = alpha_to_col(str(box_j.get("x2", "")))
                    row1_j = int(box_j.get("y1", 0)) - 1
                    row2_j = int(box_j.get("y2", 0)) - 1
                except (ValueError, TypeError):
                    continue
                
                # Check if boxes overlap
                # Boxes overlap if they intersect in both x and y dimensions
                x_overlap = not (col2_i < col1_j or col2_j < col1_i)
                y_overlap = not (row2_i < row1_j or row2_j < row1_i)
                
                if x_overlap and y_overlap:
                    return False  # Found overlapping boxes
    
    return True


def predict_record(
    client: genai.Client,
    record: Record,
    sasb_text: str,
    model: str,
    grid_size: int,
    max_retries: int = 3,
    dpi: int = 150,
    seed: int = 42,
    reports_dir: str | None = None,
) -> List[Dict[str, Any]]:
    """Call Gemini for one record and parse grid-based response with retry logic."""
    report_path = resolve_report_path(record.esg_report, reports_dir)
    image = render_pdf_page_to_pil(str(report_path), record.page, dpi=dpi)
    image = overlay_grid(image, grid_size=grid_size, dpi=dpi)

    prompt = build_prompt_text(sasb_text)

    contents: List[Any] = []
    contents.append(prompt)
    contents.append(image_to_part(image))

    config = types.GenerateContentConfig(
        seed=seed,
        response_mime_type="application/json",
        response_schema=list[GridBox],
    )
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            
            # Parse response
            if getattr(response, "parsed", None) is not None:
                boxes = [box.model_dump() for box in response.parsed]  # type: ignore[assignment]
            else:
                # Handle None or empty response text
                response_text = getattr(response, "text", None) or ""
                if not response_text.strip():
                    boxes = []
                else:
                    boxes = json.loads(response_text)
            
            # Handle None - convert to empty list
            if boxes is None:
                boxes = []
            
            # Validate the response structure and values
            if not validate_grid_boxes(boxes):
                raise ValueError(f"Invalid grid boxes structure or values: {boxes}")
            
            return boxes
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff: wait 1s, 2s, 4s...
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                # Last attempt failed, raise the exception
                raise last_exception


def format_pixel_box(box: Dict[str, Any]) -> str:
    """Convert pixel box dict to 'x1,y1,x2,y2:code' string."""
    x1 = box.get("x1", "")
    y1 = box.get("y1", "")
    x2 = box.get("x2", "")
    y2 = box.get("y2", "")
    code = box.get("code", "")
    prefix = f"{x1},{y1},{x2},{y2}"
    return f"{prefix}:{code}" if code else prefix


def load_processed_ids(output_path: Path) -> set[str]:
    """Load already processed record IDs from existing CSV file."""
    processed_ids = set()
    if output_path.exists():
        try:
            with output_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    id_val = row.get("ID", "").strip('"')
                    if id_val:
                        processed_ids.add(id_val)
        except Exception as e:
            print(f"[WARN] Failed to read existing output file: {e}")
    return processed_ids


def save_prediction(output_path: Path, record_id: str, labels: List[Dict[str, Any]], grid_size: int) -> None:
    """Save a single prediction to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert grid predictions to pixel coordinates
    target = "NONE"
    if labels:
        pixel_labels = []
        for box in labels:
            try:
                col1 = alpha_to_col(str(box.get("x1", "")))
                col2 = alpha_to_col(str(box.get("x2", "")))
                row1 = int(box.get("y1", 0)) - 1
                row2 = int(box.get("y2", 0)) - 1
                pixel_labels.append(
                    {
                        "x1": max(0, col1) * grid_size,
                        "y1": max(0, row1) * grid_size,
                        "x2": (col2 + 1) * grid_size,
                        "y2": (row2 + 1) * grid_size,
                        "code": box.get("code", ""),
                    }
                )
            except Exception:
                continue

        formatted = [format_pixel_box(box) for box in pixel_labels]
        target = ";".join(formatted) if formatted else "NONE"
    
    # Append to CSV file
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header only if this is a new file
        if not file_exists:
            writer.writerow(["ID", "TARGET"])
        writer.writerow([record_id, target])


def main(args: argparse.Namespace) -> None:
    # API key from .env (API_KEY is the sustainable-veg convention; fall back
    # to the team-16 names for compatibility).
    api_key = (os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
               or os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        raise RuntimeError("Set API_KEY (or GEMINI_API_KEY) in .env for Gemini access")
    client = genai.Client(api_key=api_key)

    # All paths come from config.yaml (consistent with run_direct/rag/bm25/...)
    config = load_config()
    data_path = Path(config["paths"]["data"])
    metrics_dir = config["paths"]["sasb_metrics"]
    reports_dir = config["paths"]["esg_reports_pdf"]
    output_path = Path(config["paths"]["results_output"].replace("{strategy}", "grid"))

    test_records = load_records(data_path)

    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        print(f"Found {len(processed_ids)} already processed records. Will skip them.")

    remaining_records = [rec for rec in test_records if rec.id not in processed_ids]

    if not remaining_records:
        print("All records have already been processed!")
        return

    print(f"Processing {len(remaining_records)} remaining records out of "
          f"{len(test_records)} total. -> {output_path}")

    for rec in tqdm(remaining_records, desc="Processing records"):
        sasb_text = sasb_text_from_report(rec.sasb_report, metrics_dir=metrics_dir)
        try:
            boxes = predict_record(
                client=client,
                record=rec,
                sasb_text=sasb_text,
                model=args.model,
                grid_size=args.grid_size,
                max_retries=args.max_retries,
                dpi=args.dpi,
                seed=args.seed,
                reports_dir=reports_dir,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[WARN] Failed record {rec.id} after {args.max_retries} retries: {exc}")
            boxes = []

        # Save immediately after processing each record
        save_prediction(output_path, rec.id, boxes, args.grid_size)

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"Completed processing. Results saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

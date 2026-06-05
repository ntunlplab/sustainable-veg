#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from bs4 import BeautifulSoup

def html_to_semantic_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    result = []

    tables = soup.find_all("table")
    if tables:
        for table_idx, table in enumerate(tables):
            rows = table.find_all("tr")
            result.append(f"[表格 {table_idx+1}]")

            for ri, row in enumerate(rows):
                cells = row.find_all(["td", "th"])
                row_texts = []

                for ci, cell in enumerate(cells):
                    text = cell.get_text(" ", strip=True)

                    rowspan = cell.get("rowspan")
                    colspan = cell.get("colspan")

                    meta = []
                    if rowspan:
                        meta.append(f"跨 {rowspan} 行")
                    if colspan:
                        meta.append(f"跨 {colspan} 列")

                    meta_str = f" ({' '.join(meta)})" if meta else ""
                    row_texts.append(f"欄位{ci+1}: {text}{meta_str}")

                result.append(" ・ " + "； ".join(row_texts))

    else:
        tds = soup.find_all(["td", "th"])
        if tds:
            result.append("[表格欄位]")
            for idx, cell in enumerate(tds):
                text = cell.get_text(" ", strip=True)
                rowspan = cell.get("rowspan")
                colspan = cell.get("colspan")

                meta = []
                if rowspan:
                    meta.append(f"跨 {rowspan} 行")
                if colspan:
                    meta.append(f"跨 {colspan} 列")

                meta_str = f" ({' '.join(meta)})" if meta else ""
                result.append(f"欄位{idx+1}: {text}{meta_str}")

    paragraphs = soup.find_all(["p", "div"])
    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        if txt:
            result.append(f"[段落] {txt}")

    if not result:
        plain_text = soup.get_text(" ", strip=True)
        if plain_text:
            result.append(plain_text)

    return "\n".join(result).strip()


def process_output_json(path: str):
    data = json.load(open(path, "r", encoding="utf-8"))

    changed = False

    for obj in data.get("objects", []):
        content = obj.get("text", "")
        if "<" in content and ">" in content:
            obj["content_semantic"] = html_to_semantic_text(content)
            changed = True
        else:
            obj["content_semantic"] = content

    out_path = path.replace("ocr_output", "ocr_output_final")
    out_dir = out_path[:-12]
    
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    json.dump(data, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[✓] Converted → {out_path}")


def walk_ocr_root(root="./ocr_output"):
    for pdf in os.listdir(root):
        pdf_dir = os.path.join(root, pdf)
        if not os.path.isdir(pdf_dir):
            continue

        for page in os.listdir(pdf_dir):
            page_dir = os.path.join(pdf_dir, page)
            if not os.path.isdir(page_dir):
                continue

            json_path = os.path.join(page_dir, "output.json")
            if os.path.exists(json_path):
                process_output_json(json_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_root", default="./ocr_output")
    args = parser.parse_args()

    walk_ocr_root(args.ocr_root)

    print("ALL DONE")

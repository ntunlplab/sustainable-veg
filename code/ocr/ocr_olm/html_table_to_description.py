#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# 1. 判斷是否為 HTML table
# =========================================================

def is_html_table(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return "<table" in t and "<tr" in t and "<td" in t


# =========================================================
# 2. Prompt（固定，不要亂動）
# =========================================================

TABLE_TO_TEXT_PROMPT = """你是一個企業永續報告理解助手。

請將下列 HTML 表格改寫成一段「中文語意描述」，說明這個表格在說什麼。

規則：
- 不要輸出任何 HTML
- 不要逐列照抄
- 保留所有重要數字、單位、公司名稱與專有名詞
- 不要加入原文沒有的推論或評論
- 使用正式、完整的中文句子

HTML 表格：
{table}

請只輸出描述文字。
"""


# =========================================================
# 3. LLM 包裝
# =========================================================

class HTMLNormalizer:
    def __init__(self, model_name: str):
        print(f"[MODEL] Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()

    @torch.inference_mode()
    def normalize(self, html: str) -> str:
        prompt = TABLE_TO_TEXT_PROMPT.format(table=html)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
        )

        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return text.strip()


# =========================================================
# 4. 處理單一 output.json
# =========================================================

def process_one_file(
    in_path: str,
    out_path: str,
    normalizer: HTMLNormalizer,
):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects", [])

    for obj in objects:
        text = obj.get("text", "")

        if is_html_table(text):
            try:
                normalized = normalizer.normalize(text)
            except Exception as e:
                print("[WARN] Failed to normalize table, fallback to original:", e)
                normalized = text
        else:
            normalized = text

        obj["text"] = normalized

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def batch_process(
    input_root: str,
    output_root: str,
    model_name: str,
):
    normalizer = HTMLNormalizer(model_name)

    total = 0

    for root, _, files in os.walk(input_root):
        if "output.json" not in files:
            continue

        rel = os.path.relpath(root, input_root)
        in_path = os.path.join(root, "output.json")
        out_path = os.path.join(output_root, rel, "output.json")

        print(f"[PROCESS] {in_path}")
        process_one_file(in_path, out_path, normalizer)
        total += 1

    print("\n==============================")
    print(f"[DONE] Total pages processed: {total}")
    print("==============================")


# =========================================================
# 6. CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="./ocr_output",
        help="original OCR output root",
    )
    parser.add_argument(
        "--output_root",
        default="./ocr_output_html",
        help="normalized OCR output root",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="LLM for HTML normalization",
    )
    args = parser.parse_args()

    batch_process(
        input_root=args.input_root,
        output_root=args.output_root,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

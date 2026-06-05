#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

ROOT = "./ocr_output"

attr_pattern = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def replace_attr(match):
        attr = match.group(1) 
        value = match.group(2) 
        value_escaped = value.replace('"', '\\"')
        return f'{attr}="{value_escaped}"'.replace('"', '\\"') \
               if '"' in value else f'{attr}=\\"{value}\\"'

    fixed = attr_pattern.sub(lambda m: f'{m.group(1)}=\\"{m.group(2)}\\"', text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(fixed)

    print(f"[FIXED] {path}")


def scan_and_process():
    for root, dirs, files in os.walk(ROOT):
        for fn in files:
            if fn == "output_raw.txt":
                process_file(os.path.join(root, fn))


if __name__ == "__main__":
    scan_and_process()
    print("\nALL DONE")

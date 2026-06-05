import os
import json
import csv

from pdf_extract_region import mark_pdf_regions
import yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# ----------------------------
# 1. Read train.json
# ----------------------------
with open(config["paths"]["train_data"], "r", encoding="utf-8") as f:
    data = json.load(f)


# ----------------------------
# 2. Convert to CSV
# ----------------------------
csv_file = "train.csv"
csv_columns = ["ID", "TARGET"]

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    
    for item in data:
        if item["label"]:
            label_str = ";".join(item["label"])

            pdf_path = os.path.join(config["paths"]["esg_reports_pdf"], item["esg_report"])
            report_folder = os.path.splitext(item["esg_report"])[0]
            out_folder = os.path.join("gt", report_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_path = os.path.join(out_folder, f"{item["page"]}.png")

            mark_pdf_regions(pdf_path, item["page"], item["label"], out_path)

        else:
            label_str = "NONE"
        
        writer.writerow({
            "ID": item["id"],
            "TARGET": label_str
        })
        
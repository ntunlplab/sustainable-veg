import os
import json
import csv
import argparse
import time
import base64
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from tqdm import tqdm
from utils import load_faiss_index, mark_pdf_regions
from path_utils import load_config
from utils import normalize_text

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")

TOTAL_CHUNK_SIZE = 350


def parse_args():
    parser = argparse.ArgumentParser(description="rag")
    parser.add_argument("--chandra", action="store_true", help="Use Chandra OCR instead of olmOCR (default)")
    parser.add_argument("--prompt", type=str, default=str("code/prompt/prompt_rag.txt"), help="path to prompt template file")
    return parser.parse_args()


def read_dataset_json(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    return data


def sasb_stringify(metrics_data, top_n_metrics=None):
    # "code": "EM-RM-110a.1",
    # "topic": "溫室氣體排放",
    # "metric": "範疇 1 排放之全球總排放量",
    # "category": "量化",
    # "unit": "公噸 (t) 二氧化碳當量"

    tmp_results = []
    # turn each metric to string
    for metric in metrics_data:
        if top_n_metrics is not None:
            if metric["code"] not in top_n_metrics:
                continue
        code = metric["code"]
        topic = metric["topic"]
        description = metric["metric"]
        category = metric["category"]
        unit = metric["unit"]

        metric_str = f"- Code: {code}; Topic: {topic}; Description: {description}; Category: {category}; Unit: {unit}."
        tmp_results.append(metric_str)

    return "\n".join(tmp_results)


def apply_prompt_template(template_path, box_content, metrics_data, top_n_metrics=None):
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders in the template
    prompt = template.replace("{box_content}", box_content)
    prompt = prompt.replace("{metrics_data}", sasb_stringify(metrics_data, top_n_metrics=top_n_metrics))

    return prompt


def generate_answer(client, image_path, question, model_name="gpt-4o-mini"):
    if image_path is not None:
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        response = client.chat.completions.create(
            model=model_name,  # or latest vision-capable model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

    answer = response.choices[0].message.content
    return answer


def llm_predict(client, prompt, image_path=None):
    # print(prompt)

    response = generate_answer(client, image_path, prompt)
    # print("LLM Response:")
    # print(response)

    # exit(0)
    return response


def parse_response(response):
    # look for json enclosed in <metric> <metric/>
    start_tag = "<metric>"
    end_tag = "</metric>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)

    if start_index == -1 or end_index == -1:
        return None
    json_str = response[start_index + len(start_tag):end_index].strip()
    try:
        parsed_json = json.loads(json_str)
        # print(parsed_json)
    except json.JSONDecodeError:
        return None
    
    try:
        metric_code = parsed_json["metric"]
        if metric_code == "NONE":
            return None
        return f"{metric_code}"
    except:
        return None


def get_topk(model, index, vector_data, ocr_path, top_k=5):
    targets = []

    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    if "ocr_olm" in ocr_path:
        for chunk in ocr_data["objects"]:
            obj = {}

            bbox = chunk.get("bbox_pdf", "")
            if not bbox:
                continue
            bbox_str = list(map(str, bbox))
            obj["bbox_pdf"] = bbox_str

            text = chunk.get("text", "").strip()
            if not text:
                continue
            obj["text"] = normalize_text(text)

            query_vec = model.encode(obj["text"], normalize_embeddings=True)
            query_vec = np.array([query_vec], dtype="float32")

            _, indices = index.search(query_vec, top_k)
            top_idx = indices[0]
            obj["retrieval"] = [vector_data[top_idx[i]]["code"] for i in range(top_k)]

            targets.append(obj)
    elif "ocr_chandra" in ocr_path:
        for chunk in ocr_data:
            if chunk["label"] not in ["Section-Header", "Text", "Table", "Figure", "Footnote"]:
                continue

            obj = {}

            bbox = chunk.get("bbox_pdf", "")
            if not bbox:
                continue
            bbox_str = list(map(str, bbox))
            obj["bbox_pdf"] = bbox_str

            text = chunk["content"]

            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator="\n")
            if not text:
                continue
            obj["text"] = normalize_text(text)

            query_vec = model.encode(obj["text"], normalize_embeddings=True)
            query_vec = np.array([query_vec], dtype="float32")

            _, indices = index.search(query_vec, top_k)
            top_idx = indices[0]
            obj["retrieval"] = [vector_data[top_idx[i]]["code"] for i in range(top_k)]

            targets.append(obj)
    else:
        print(f"[Error]: Unknown path: {ocr_path}")

    return targets


def process(data, args, client, config):
    model = SentenceTransformer(config["model"]["name"])

    counter = 0
    predictions = {}
    predictions_format = {}
    raw_responses = {}
    for item in tqdm(data):
        page = item["page"]
        instance_id = item["id"]
        esg_report_path = os.path.basename(item["esg_report"])
        esg_report_name = esg_report_path.replace(".pdf", "")
        sasb_report = item["sasb_report"].split("/")[-1].replace("SASB-", "").replace(".pdf", "")

        index, vector_data = load_faiss_index(
            config["paths"]["sasb_database"],
            sasb_report,
            config["paths"]["faiss_index"],
            config["paths"]["vector_data"],
        )

        ocr_path_base = config["paths"]["chandra_ocr_output"] if args.chandra else config["paths"]["ocr_output"]
        ocr_path = f"{ocr_path_base}/{esg_report_name}/{page}/output.json"
        ocr_data = get_topk(model, index, vector_data, ocr_path, 5)

        metric_path = f"{config['paths']['sasb_metrics']}/{sasb_report}.json"
        with open(metric_path, 'r') as f:
            metrics_data = json.load(f)

        all_parsed = []
        all_formated = []
        all_raw_responses = []
        for i, chunk in enumerate(ocr_data):

            box_content = chunk["text"]
            pdf_bbox = chunk["bbox_pdf"]
            top_n_codes = chunk["retrieval"]

            prompt = apply_prompt_template(args.prompt, box_content, metrics_data, top_n_metrics=top_n_codes)
            response = llm_predict(client, prompt)
            parsed = parse_response(response)

            # # print for debugging
            # print(f"Instance ID: {instance_id}, Chunk {i}")
            # print("========== Prompt ==========")
            # print(prompt)
            # print("======== Response =========")
            # print(response)
            # print("======== Parsed Metric ========")
            # print(parsed)
            # print("\n\n")
            # print("-------------------------------")
            # print(f"Progress: {counter + 1}/{TOTAL_CHUNK_SIZE}")


            if parsed == None:
                continue

            all_parsed.append(parsed)
            all_raw_responses.append(response)

            formated = f"{pdf_bbox[0]},{pdf_bbox[1]},{pdf_bbox[2]},{pdf_bbox[3]}:{parsed}"
            all_formated.append(formated)

            counter += 1
            # if counter == 3:
            #     exit(0)
            # if counter % 10 == 0:
            #     # wait a few seconds to avoid rate limit
            #     print("Waiting for 5 seconds to avoid rate limit...")
            #     time.sleep(5)

        predictions[instance_id] = all_parsed
        predictions_format[instance_id] = all_formated
        raw_responses[instance_id] = all_raw_responses

        # Visualization: GT + Prediction
        pdf_path = os.path.join(config["paths"]["esg_reports_pdf"], esg_report_path)
        out_dir = os.path.join(config["paths"]["marked_results_output"].replace("{strategy}", "rag"), esg_report_name)
        os.makedirs(out_dir, exist_ok=True)
        out_img = os.path.join(out_dir, f"{page}.png")

        gt_boxes = item.get("label", [])
        pred_boxes = [pred for pred in all_formated if not pred.endswith("None")]
        mark_pdf_regions(pdf_path, page, gt_boxes=gt_boxes, pred_boxes=pred_boxes, output_path=out_img)

        # break

    return predictions, predictions_format, raw_responses


def save_to_csv(predictions_format, dst_path):
    with open(dst_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # write header
        csvwriter.writerow(["ID", "TARGET"])
        for report_id, predictions in predictions_format.items():
            # skip if prediction in predictions ends with NONE
            predictions = [pred for pred in predictions if not pred.endswith("None")]

            # join predictions with ;
            pred_str = ";".join(predictions)
            if pred_str == "":
                pred_str = "NONE"
            csvwriter.writerow([report_id, pred_str])


def main():
    args = parse_args()
    config = load_config()

    data = read_dataset_json(config["paths"]["data"])
    # data = data[:1]  # for testing

    client = OpenAI(api_key=api_key)

    print("Processing data...")
    start_time = time.time()
    predictions, predictions_format, raw_responses = process(data, args, client, config)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"[Total Time]: {total_time:.2f}s")
    print(f"[Average Time]: {total_time/len(data):.2f}s")

    # # print first 10 predictions for test
    # print("Predictions (first 10):")
    # for k, v in list(predictions_format.items())[:10]:
    #     print(f"{k}: {v}")

    # # save all 4 predictions to json
    # with open("./results/test_predictions_4o-mini_v2-2.json", 'w') as f:
    #     json.dump(predictions, f, indent=4)
    # with open("./results/test_predictions_format_4o-mini_v2-2.json", 'w') as f:
    #     json.dump(predictions_format, f, indent=4)
    # with open("./results/test_raw_responses_4o-mini_v2-2.json", 'w') as f:
    #     json.dump(raw_responses, f, indent=4)
    save_to_csv(predictions_format, config["paths"]["results_output"].replace("{strategy}", "rag"))


if __name__ == "__main__":
    main()
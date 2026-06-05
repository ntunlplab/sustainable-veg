import json
import csv
import argparse
import os
import time
import base64

from tqdm import tqdm
from PIL import Image

from openai import OpenAI

API_KEY = "OPENAI_API_KEY"

TOTAL_CHUNK_SIZE = 350

def parse_args():
    parser = argparse.ArgumentParser(description="exp1")
    parser.add_argument("--metrics", type=str, default="../metrics", help="path to metrics json files")
    parser.add_argument("--reports", type=str, default="../../data/reports", help="path to pdf report files")
    parser.add_argument("--dataset", type=str, default="../../data", help="path to dataset")
    parser.add_argument("--boxex", type=str, default="../retriever/top5_results", help="path to OCR boxex files")
    parser.add_argument("--prompt", type=str, default="prompt2-1.txt", help="path to prompt template file")
    return parser.parse_args()

def read_dataset_json(dataset_path):
    # train and test
    train_path = f"{dataset_path}/Sustainable-VEG.json"
    test_path = f"{dataset_path}/Sustainable-VEG.json"
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    return train_data, test_data

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
    # prompt = prompt.replace("{metrics_data}", metrics_data) # need to stringify
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
    
def process(data, args, client):
    counter = 0
    predictions = {}
    predictions_format = {}
    raw_responses = {}
    for item in tqdm(data):
        page = item["page"]
        instance_id = item["id"]
        esg_pdf_path = f"{args.reports}/{item['esg_report']}"
        esg_report_name = item["esg_report"].replace(".pdf", "")
        sasb_metrics = item["sasb_report"].replace("SASB-", "").replace(".pdf", "")

        metric_path = f"{args.metrics}/{sasb_metrics}.json"

        ocr_path = f"{args.boxex}/{esg_report_name}/{page}"
        bbox_image = f"{ocr_path}/bbox.png"
        bbox_info = f"{ocr_path}/output.json"

        with open(bbox_info, 'r') as f:
            ocr_data = json.load(f)
            ocr_data = ocr_data["objects"]

        with open(metric_path, 'r') as f:
            metrics_data = json.load(f)

        all_parsed = []
        all_formated = []
        all_raw_responses = []
        for i, chunk in enumerate(ocr_data):
            # if chunk doesn't have key "bbox", continue
            if "bbox_pdf" not in chunk:
                continue
            # read cropped image
            # filename = f"{ocr_path}/chunk_{i}.png"
            # img = Image.open(filename)
            box_content = chunk["text"]
            pdf_bbox = chunk["bbox_pdf"]
            top_n_raw = chunk["retrieval"][:3]
            top_n_codes = [item["code"] for item in top_n_raw]
            # label = chunk["label"]

            # ignore_label = ["Page-Footer", "Page-Header", "Page-header", "Footnote", "Section-Header"]

            # if label in ignore_label:
            #     continue

            prompt = apply_prompt_template(args.prompt, box_content, metrics_data, top_n_metrics=top_n_codes)
            # print(prompt)
            # exit(0)
            response = llm_predict(client, prompt)
            parsed = parse_response(response)

            # print for debugging
            print(f"Instance ID: {instance_id}, Chunk {i}")
            print("========== Prompt ==========")
            print(prompt)
            print("======== Response =========")
            print(response)
            print("======== Parsed Metric ========")
            print(parsed)
            print("\n\n")
            print("-------------------------------")
            print(f"Progress: {counter + 1}/{TOTAL_CHUNK_SIZE}")

            # exit(0)

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


    return predictions, predictions_format, raw_responses

def main():
    args = parse_args()
    train_data, test_data = read_dataset_json(args.dataset)

    # test_data = test_data[:1]  # for testing

    client = OpenAI(api_key=API_KEY)

    # print("Processing train data...")
    # train_predictions, train_predictions_format = process(train_data, args, client)
    print("Processing test data...")
    test_predictions, test_predictions_format, raw_responses = process(test_data, args, client)

    # print first 10 predictions for train and test
    # print("Train Predictions (first 10):")
    # for k, v in list(train_predictions_format.items())[:10]:
    #     print(f"{k}: {v}")

    print("Test Predictions (first 10):")
    for k, v in list(test_predictions_format.items())[:10]:
        print(f"{k}: {v}")

    # save all 4 predictions to json
    # with open("train_predictions.json", 'w') as f:
    #     json.dump(train_predictions, f, indent=4)
    with open("test_predictions_4o-mini_v2-2.json", 'w') as f:
        json.dump(test_predictions, f, indent=4)
    # with open("train_predictions_format.json", 'w') as f:
    #     json.dump(train_predictions_format, f, indent=4)
    with open("test_predictions_format_4o-mini_v2-2.json", 'w') as f:
        json.dump(test_predictions_format, f, indent=4)
    with open("test_raw_responses_4o-mini_v2-2.json", 'w') as f:
        json.dump(raw_responses, f, indent=4)

if __name__ == "__main__":
    main()
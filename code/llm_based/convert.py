import json
import csv
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON predictions to CSV")
    parser.add_argument("--src", type=str, default="/nfs/nas-8.1/gtyi/IR/codes/test_predictions_format_4o-mini_v2-2.json", help="Path to the source JSON file")
    parser.add_argument("--dst", type=str, default=None, help="Path to the destination CSV file")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dropout")
    return parser.parse_args()

# src_path = "/nfs/nas-8.1/gtyi/IR/codes/test_predictions_format_4o-mini_v2-1.json"

def main():
    args = parse_args()
    src_path = args.src
    dst_path = args.dst if args.dst is not None else src_path.replace(".json", ".csv")

    # apply seed
    random.seed(args.seed)

    with open(src_path, 'r') as f:
        data = json.load(f)

    # convert to csv
    with open(dst_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # write header
        csvwriter.writerow(["ID", "TARGET"])
        for report_id, predictions in data.items():
            # skip if prediction in predictions ends with NONE
            predictions = [pred for pred in predictions if not pred.endswith("None")]

            # apply drop out
            if args.dropout > 0.0:
                predictions = [pred for pred in predictions if random.random() > args.dropout]

            # join predictions with ;
            pred_str = ";".join(predictions)
            if pred_str == "":
                pred_str = "NONE"
            csvwriter.writerow([report_id, pred_str])

if __name__ == "__main__":
    main()
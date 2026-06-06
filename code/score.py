import os
import json
import pandas as pd
from datetime import datetime

from path_utils import load_config


IOU_THR = 0.5
BOX_SEP = ";"


def _safe_split_boxes(s):
    """Split a 'boxes' cell into a list of box strings."""
    if s is None:
        return []
    if isinstance(s, float):
        # NaN -> no boxes
        return []
    s = str(s).strip()
    if not s or s.upper() == "NONE":
        return []
    # split by ';'
    parts = [p.strip() for p in s.split(BOX_SEP)]
    return [p for p in parts if p]


def _parse_box_str(box_str):
    """
    '<x1>,<y1>,<x2>,<y2>:<code>' -> (x1,y1,x2,y2,code)
    coordinates are float; code is string
    """
    if ":" in box_str:
        coord, code = box_str.split(":", 1)
        code = code.strip()
    else:
        raise Exception(f"Invalid box coord: {box_str}, your submisson should contain ':'")
    xs = [v.strip() for v in coord.split(",")]
    if len(xs) != 4:
        raise Exception(f"Invalid box coord: {box_str}, you should contain 4 coordinates and separate them by ','")
    x1, y1, x2, y2 = map(float, xs)
    if not (x1 < x2 and y1 < y2):
        raise Exception(f"Invalid box coord: {box_str}, x1 should be lower than x2 and y1 should be lower than y2")
    return (x1, y1, x2, y2, code)


def _to_xywh(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(iou_mat, thr):
    """
    iou_mat: 2D list [len(GT)] x [len(Pred)]
    return: (matches, used_g, used_p)
      matches: list of (gi, pj, iou)
    """
    m = len(iou_mat)
    n = len(iou_mat[0]) if m else 0
    triples = []
    for gi in range(m):
        for pj in range(n):
            triples.append((iou_mat[gi][pj], gi, pj))
    triples.sort(reverse=True)  # largest IoU first
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pj in triples:
        if iou < thr:
            break
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        matches.append((gi, pj, iou))
    return matches, used_g, used_p


def _evaluate_image(gt_boxes, pred_boxes, iou_thr=IOU_THR):
    """
    gt_boxes / pred_boxes: list of (x1,y1,x2,y2,code)
    returns: (tp, fp, fn)
    """
    # bucket by class
    gt_bins = {}
    for (x1, y1, x2, y2, c) in gt_boxes:
        gt_bins.setdefault(c, []).append(_to_xywh(x1, y1, x2, y2))

    pred_bins = {}
    for (x1, y1, x2, y2, c) in pred_boxes:
        pred_bins.setdefault(c, []).append(_to_xywh(x1, y1, x2, y2))

    tp = fp = fn = 0

    for cls in set(gt_bins.keys()).union(set(pred_bins.keys())):
        g = gt_bins.get(cls, [])
        p = pred_bins.get(cls, [])
        # IoU matrix
        iou_mat = [[_iou_xywh(gi, pj) for pj in p] for gi in g]
        matches, used_g, used_p = _greedy_match(iou_mat, iou_thr)
        tp += len(matches)
        fp += max(0, len(p) - len(used_p))
        fn += max(0, len(g) - len(used_g))

    return tp, fp, fn


def _prf(tp, fp, fn):
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
    return P, R, F1


def _build_index(df):
    """
    returns dict: image_id -> list[(x1,y1,x2,y2,code)]
    """
    index = {}
    for _, row in df.iterrows():
        img = str(row["ID"]).strip()
        boxes_raw = _safe_split_boxes(row.get("TARGET", ""))
        boxes = []
        for b in boxes_raw:
            parsed = _parse_box_str(b)
            if parsed is not None:
                boxes.append(parsed)
        index[img] = boxes
    return index


def _convert_to_solution(json_path, csv_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        labels = item.get("label", [])
        
        if labels:
            # 將所有的 label 用分號 ';' 串接起來
            target = ";".join(str(l) for l in labels)
        else:
            target = "NONE"

        rows.append({
            "ID": item["id"],
            "TARGET": target,
            "Usage": "test"
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def score(solution_df: pd.DataFrame, submission_df: pd.DataFrame) -> float:
    sol_idx = _build_index(solution_df)
    sub_idx = _build_index(submission_df)

    imgs = sorted(set(sol_idx.keys()).union(set(sub_idx.keys())))

    total_tp = total_fp = total_fn = 0

    for img in imgs:
        gt_boxes = sol_idx.get(img, [])
        pd_boxes = sub_idx.get(img, [])

        tp, fp, fn = _evaluate_image(gt_boxes, pd_boxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    _, _, f1 = _prf(total_tp, total_fp, total_fn)
    return f1


def log(msg, log_file="score.log"):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    config = load_config()

    gt_json = config["paths"]["data"]
    gt_csv = "data/solution.csv"
    results_dir = "results"
    
    log_file = "score.log"
    log(f"\n===== SCORE RUN {datetime.now()} =====", log_file)

    for exp_name in sorted(os.listdir(results_dir)):
        pred_path = os.path.join(results_dir, exp_name, "prediction.csv")

        if not os.path.exists(pred_path):
            continue

        _convert_to_solution(gt_json, gt_csv)

        solution_df = pd.read_csv(gt_csv)
        submission_df = pd.read_csv(pred_path)

        f1_score = score(solution_df, submission_df)

        msg = f"{exp_name} F1 = {f1_score:.4f}"
        log(msg, log_file)


if __name__ == "__main__":
    main()
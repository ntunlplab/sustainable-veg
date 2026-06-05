#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score submission against solution for region extraction task.

Usage: python score.py <solution_train.csv> <submission_train.csv> <solution_test.csv> <submission_test.csv>
"""

import pandas as pd


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

def score(solution_train: pd.DataFrame, submission_train: pd.DataFrame, solution_test: pd.DataFrame, submission_test: pd.DataFrame) -> float:
    sol_idx_train = _build_index(solution_train)
    sub_idx_train = _build_index(submission_train)
    sol_idx_test = _build_index(solution_test)
    sub_idx_test = _build_index(submission_test)

    imgs_train = sorted(set(sol_idx_train.keys()).union(set(sub_idx_train.keys())))
    imgs_test = sorted(set(sol_idx_test.keys()).union(set(sub_idx_test.keys())))
    total_tp = total_fp = total_fn = 0

    for img in imgs_train:
        gt_boxes = sol_idx_train.get(img, [])
        pd_boxes = sub_idx_train.get(img, [])
        tp, fp, fn = _evaluate_image(gt_boxes, pd_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    for img in imgs_test:
        gt_boxes = sol_idx_test.get(img, [])
        pd_boxes = sub_idx_test.get(img, [])
        tp, fp, fn = _evaluate_image(gt_boxes, pd_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    P, R, F1 = _prf(total_tp, total_fp, total_fn)
    
    return F1

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python score.py <solution_train.csv> <submission_train.csv> <solution_test.csv> <submission_test.csv>")
        sys.exit(1)

    solution_train_path = sys.argv[1]
    submission_train_path = sys.argv[2]
    solution_test_path = sys.argv[3]
    submission_test_path = sys.argv[4]

    solution_train_df = pd.read_csv(solution_train_path)
    submission_train_df = pd.read_csv(submission_train_path)
    solution_test_df = pd.read_csv(solution_test_path)
    submission_test_df = pd.read_csv(submission_test_path)

    f1_score = score(solution_train_df, submission_train_df, solution_test_df, submission_test_df)
    print(f"F1 Score: {f1_score:.4f}")
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from itertools import permutations
from math import isnan


def IoU(gate1, gate2):
    poly1 = Polygon(gate1.reshape(4, 2))
    poly2 = Polygon(gate2.reshape(4, 2))
    if poly2.is_valid == False:
        # invalid polygon
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    iou = intersection_area / union_area

    return iou


def match_gates(IoU_matrix):
    num_gt, num_pred = IoU_matrix.shape
    best_match = []
    best_score = 0

    # IoU matrix is always fat (more columns than rows)
    if num_gt > num_pred:
        IoU_matrix = IoU_matrix.T

    # go through all permutations of the gates find the best according to total IoU
    for perm in permutations(range(max(num_gt, num_pred)), min(num_gt, num_pred)):
        current_score = sum(IoU_matrix[i, j] for i, j in enumerate(perm))
        if current_score > best_score:
            best_score = current_score
            best_match = perm

    best_iou_gt = np.zeros(num_gt, dtype=np.float32)
    best_iou_pred = np.zeros(num_pred, dtype=np.float32)
    for i, j in enumerate(best_match):
        gt_idx = j if num_gt > num_pred else i
        pred_idx = i if num_gt > num_pred else j
        best_iou_gt[gt_idx] = IoU_matrix[i, j]
        best_iou_pred[pred_idx] = IoU_matrix[i, j]

    return best_iou_gt, best_iou_pred, best_score


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """
    Calculate the mean F1 score for gate detection predictions based on Intersection over Union (IoU).
    This function compares the ground truth and predicted gate region.
    It computes the IoU for each pair of ground truth and predicted boxes, matches them based on the highest IoU,
    and calculates the F1 score at different IoU thresholds.
    Args:
        solution (pd.DataFrame): DataFrame containing the ground truth bounding boxes with a column named "PredictionString".
        submission (pd.DataFrame): DataFrame containing the predicted bounding boxes with a column named "PredictionString".
        row_id_column_name (str): The name of the column containing the row identifiers, which will be removed from both DataFrames.
    Returns:
        float: The mean F1 score across different IoU thresholds.
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    gt_string = solution["PredictionString"]
    pred_string = submission["PredictionString"]

    iou_dict = {
        "gt": [],
        "pred": [],
    }
    for gt, pred in zip(gt_string, pred_string):
        gt = np.array(list(map(float, gt.split(" "))), dtype=np.float32).reshape(-1, 12)
        if isinstance(pred, float) and isnan(pred):
            pred = np.empty((0, 12), dtype=np.float32)
        if isinstance(pred, str):
            if pred.strip() == "":
                pred = np.empty((0, 12), dtype=np.float32)
            else:
                pred = np.array(
                    list(map(float, pred.split(" "))), dtype=np.float32
                ).reshape(-1, 12)

        # cap num predictions to 10
        pred = pred[:10]

        # remove visibility flag
        gt = gt[:, [0, 1, 3, 4, 6, 7, 9, 10]]
        pred = pred[:, [0, 1, 3, 4, 6, 7, 9, 10]]

        # calculate IoU for all pairs
        pred_IoU = np.zeros((len(gt), len(pred)), dtype=np.float32)
        for i, gate1 in enumerate(gt):
            for j, gate2 in enumerate(pred):
                iou = IoU(gate1, gate2)
                pred_IoU[i, j] = iou

        # match gates according to highest IoU
        iou_gt, iou_pred, _ = match_gates(pred_IoU)
        iou_dict["gt"].append(iou_gt)
        iou_dict["pred"].append(iou_pred)

    # compute f1 (precision and recall) at different IoU thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)
    f1_score = np.empty_like(thresholds)

    for i, threshold in enumerate(thresholds):
        gt_match = np.concatenate([iou > threshold for iou in iou_dict["gt"]])
        pred_match = np.concatenate([iou > threshold for iou in iou_dict["pred"]])

        recall = gt_match.sum() / len(gt_match)
        precision = pred_match.sum() / max(len(pred_match), 1)

        f1_score[i] = 2 * recall * precision / (recall + precision + 1e-6)

    return f1_score.mean()


if __name__ == "__main__":
    sol = pd.read_csv("solution.csv")
    sub = pd.read_csv("submission.csv")
    print(score(sol, sub, "Id"))

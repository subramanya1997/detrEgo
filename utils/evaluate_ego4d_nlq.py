#! /usr/bin/env python
"""
Script to evaluate performance of any model for Ego4d Episodic Memory.

Natural Language Queries (NLQ)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import terminaltables


def display_results(results, mIoU, thresholds, topK, title=None):
    display_data = [
        [f"Rank@{ii}\nmIoU@{jj}" for ii in topK for jj in thresholds] + ["mIoU"]
    ]
    results *= 100
    mIoU *= 100
    display_data.append(
        [
            f"{results[jj][ii]:.02f}"
            for ii in range(len(topK))
            for jj in range(len(thresholds))
        ]
        + [f"{mIoU:.02f}"]
    )
    table = terminaltables.AsciiTable(display_data, title)
    for ii in range(len(thresholds) * len(topK)):
        table.justify_columns[ii] = "center"
    return table.table


def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def evaluate_nlq_performance(
    predictions, thresholds, topK, per_instance=False
):
    """Evalutes the performances."""
    num_queries = 0
    results = [[[] for _ in topK] for _ in thresholds]
    average_IoU = []
    num_instances = 0
    for v, pred in predictions.items():
        for pred_datum in pred:
            num_queries += 1
            if len(pred_datum["pred_timespan"]) == 0:
                continue
            overlap = compute_IoU(
                pred_datum["pred_timespan"],
                [pred_datum["gt_timespan"]]
            )
            w_is_nan = np.isnan(overlap)
            overlap[w_is_nan] = 0.0
            average_IoU.append(np.mean(np.sort(overlap[0])))
            # average_IoU.append(np.mean(np.sort(overlap[0])[-3:]))
            for tt, threshold in enumerate(thresholds):
                for rr, KK in enumerate(topK):
                    results[tt][rr].append((overlap > threshold)[:KK].any())
            num_instances += 1

    mean_results = np.array(results).mean(axis=-1)
    mIoU = np.mean(average_IoU)
    print(f"Evaluated: {num_instances} / {num_queries} instances")
    if per_instance:
        per_instance_results = {
            "overlap": overlap,
            "average_IoU": average_IoU,
            "results": results,
        }
        return mean_results, mIoU, per_instance_results
    else:
        return mean_results, mIoU

#! /usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt


def pre_rec(tup):
    """
    Returns a tuple of (precision, recall) for a given input of (tp, tn, fp, fn)
    @param tup - A tuple of (tp, tn, fp, fn)
    """
    tp, tn, fp, fn = tup
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (precision, recall)
    except ZeroDivisionError:
        return (np.nan, np.nan)


def categorize_actual_preds(actuals, preds, threshold=0.5):
    """
    Find the tp, tn, fp, fn for a model with a given threshold
    @param actuals
    @param preds
    @param threshold - default = 0.5
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # print(actuals.shape)
    if actuals.size != preds.size:
        print("Shapes of actuals and preds not equal")
        return tp, tn, fp, fn
    if np.count_nonzero(actuals) == 0:
        # print("[WARNING] No positive predictions")
        return tp, tn, fp, fn

    if actuals.ndim == 2:
        actuals = np.squeeze(actuals, axis=0)
        preds = np.squeeze(preds, axis=0)

    true_indices = []
    false_indices = []
    for i, val in enumerate(actuals):
        if val:
            true_indices.append(i)
        else:
            false_indices.append(i)

    # Calculate tp and fn
    for i in true_indices:
        act = actuals[i]
        pred = preds[i]
        if pred >= threshold:
            tp += 1
        else:
            fn += 1

    # Calculate tn and fp
    for i in false_indices:
        pred = preds[i]
        if pred >= threshold:
            fp += 1
        else:
            tn += 1

    return (tp, tn, fp, fn)


def plot_roc_for_feature_id(ids, actuals, preds, feature_id_to_name_map=None):
    """
    Plots the roc graph for multiple input feature ids
    @param ids: int, tuple or list of feature ids
    @actuals: list of true labels
    @preds: list of predicted labels
    @feature_id_to_map_map: dictionary of feature_id to feature_name mapping
    """
    loop = []
    if isinstance(ids, int):
        loop = range(ids)
    elif isinstance(ids, tuple):
        low = min(ids[0], ids[1])
        high = max(ids[0], ids[1])
        loop = range(low, high)
    elif isinstance(ids, list):
        loop = ids
    plt.figure(figsize=(12, 12))
    for i in loop:
        fpr, tpr, thres = roc_curve(actuals[:, i], preds[:, i], drop_intermediate=False)
        if feature_id_to_name_map:
            label = feature_id_to_name_map.get(i) + "(" + str(i) + ")"
        else:
            label = str(i)
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], ls="--", label="No Skills")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()


def result_classifications(actuals, preds, threshold=0.5):
    """
    @return: returns a dict of feature_id to a tuple of (tp, tn, fp, fn)
    """
    classifications = dict()
    # true_labels = test_gen.labels
    for i in range(preds.shape[1]):
        classifications[i] = categorize_actual_preds(actuals[:, i], preds[:, i], threshold=threshold)

    return classifications


def calculate_pres_recall(actuals, preds, classifications=None, threshold=0.5):
    precision_recall_report = dict()
    if not classifications:
        classifications = result_classifications(actuals, preds, threshold=threshold)
    for i in classifications.keys():
        precision_recall_report[i] = pre_rec(classifications[i])

    return precision_recall_report


def calculate_roc(tp, tn, fp, fn):
    trp = 0.0
    fpr = 0.0
    try:
        tpr = (tp) / (tp + fn)
        fpr = (fp) / (fp + tn)

    except Exception:
        print("Some error occurred!")
    return tpr, fpr


def roc_curve(actuals, preds, thresholds):
    tprs = []
    fprs = []
    for t in thresholds:
        tp, tn, fp, fn = categorize_actual_preds(actuals, preds, t)
        tpr, fpr = calculate_roc(tp, tn, fp, fn)
        tprs.append(tpr)
        fprs.append(fpr)

    tprs = np.expand_dims(np.array(tprs), axis=0)
    fprs = np.expand_dims(np.array(fprs), axis=0)

    return tprs, fprs, thresholds


def plot_roc(tpr, fpr, threshold=0.5):
    if isinstance(tpr, list):
        tpr = np.array(tpr)
        tpr = np.expand_dims(tpr, axis=0)
    if isinstance(fpr, list):
        fpr = np.array(fpr)
        fpr = np.expand_dims(fpr, axis=0)

    plt.figure(figsize=(12, 12))

    for i in range(tpr.shape[0]):
        plt.plot(fpr[i, :], tpr[i, :], marker='o', label=str(i))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def generate_data(n):
    actuals = np.array([random.randint(0, 1) for i in range(n)])
    preds = np.array([random.random() for i in range(n)])

    return actuals, preds


if __name__ == '__main__':
    thresholds = [i / 100 for i in range(101)]
    n = 120

    a1, p1 = generate_data(100)
    a2, p2 = generate_data(100)

    tprs = []
    fprs = []
    t1, f1, _ = roc_curve(a1, p1, thresholds)
    t2, f2, _ = roc_curve(a2, p2, thresholds)
    tprs = np.concatenate([t1, t2], axis=0)
    fprs = np.concatenate([f1, f2], axis=0)
    plot_roc(tprs, fprs)

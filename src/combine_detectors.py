import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def filter_weights(weights, k):
    if k == len(weights):
        return weights
    # Find indices of top-k elements
    top_k_indices = np.argsort(weights)[-k:]

    # Zero out elements not in top-k
    averaged_probabilities_filtered = np.zeros_like(weights)
    averaged_probabilities_filtered[top_k_indices] = weights[top_k_indices]

    # Normalize probabilities so that they sum to 1
    averaged_probabilities_filtered = averaged_probabilities_filtered/sum(averaged_probabilities_filtered)

    return averaged_probabilities_filtered

# def wrapper_get_metrics(data_tuple):
# 	metrics_to_keep = ["AUC-ROC", "AUC-PR", "VUS-ROC", "VUS-PR"]

# 	score = data_tuple[0]
# 	label = data_tuple[1]

# 	result = get_metrics(score, label, metric="all", slidingWindow=10)

# 	return {key: result[key.replace('-', '_')] for key in metrics_to_keep}

from sklearn.metrics import roc_auc_score, average_precision_score

def wrapper_get_metrics(data_tuple):
    # Extract score (predictions) and label (true values)
    score = data_tuple[0]
    label = data_tuple[1]

    # Compute AUC-ROC and AUC-PR
    auc_roc = roc_auc_score(label, score)
    auc_pr = average_precision_score(label, score)

    # Return the computed metrics in a dictionary
    return {"AUC-ROC": auc_roc, "AUC-PR": auc_pr}


def row_combine_detectors(row, scores, labels, detectors=None, k=2):
    weights = row[detectors].values if detectors else row.values
    norm_weights = weights/sum(weights)
    filtered_weights = filter_weights(norm_weights, k=k)

    curr_score = np.average(scores[row['Time series']], weights=filtered_weights, axis=1)
    curr_label = labels[row['Time series']]
    # curr_metric = wrapper_get_metrics((curr_score, curr_label))

    # return curr_metric['AUC-PR']
    return average_precision_score(curr_label, curr_score)


def combine_detectors(df, scores, labels, detectors):
    weighted_scores = []
    
    tqdm.pandas()
    df['k=4'] = df.progress_apply(lambda row: row_combine_detectors(row, scores, labels, detectors, k=4), axis=1)

    return df
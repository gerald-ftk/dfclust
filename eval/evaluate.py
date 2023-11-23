import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Check if the script is inside the 'scripts' directory
if os.path.basename(current_dir) == "eval":
    root_dir = os.path.dirname(current_dir)
else:
    root_dir = current_dir

sys.path.append(root_dir)

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from dfclust.ogmc import OGMCGraph
from tqdm import tqdm


def load_data(file_path):
    with np.load(file_path) as data:
        true_labels = data["labels"]
        features = data["features"]
    return true_labels, features


def calculate_bcubed_metrics(true_labels, test_labels):
    precision = precision_score(true_labels, test_labels, average="weighted")
    recall = recall_score(true_labels, test_labels, average="weighted")
    f_measure = f1_score(true_labels, test_labels, average="weighted")
    return precision, recall, f_measure


def calculate_additional_metrics(true_labels, test_labels):
    ari = adjusted_rand_score(true_labels, test_labels)
    nmi = normalized_mutual_info_score(true_labels, test_labels)
    return ari, nmi


def generate_test_labels(features):
    ogmc = OGMCGraph()
    for row in tqdm(features, "Adding samples"):
        ogmc.add_sample(row)

    return ogmc._labels_with_noise

def b_cubed_precision_recall(y_true, y_pred):
    # Create a dictionary to hold counts
    true_cluster_counts = {}
    pred_cluster_counts = {}
    for t, p in zip(y_true, y_pred):
        true_cluster_counts[t] = true_cluster_counts.get(t, 0) + 1
        pred_cluster_counts[p] = pred_cluster_counts.get(p, 0) + 1

    # Calculate precision and recall per element
    precision_sum = 0
    recall_sum = 0
    for t, p in zip(y_true, y_pred):
        tp = len([1 for yt, yp in zip(y_true, y_pred) if yt == t and yp == p])
        precision_sum += tp / pred_cluster_counts[p]
        recall_sum += tp / true_cluster_counts[t]

    # Average over all elements
    precision = precision_sum / len(y_true)
    recall = recall_sum / len(y_true)

    return precision, recall


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("file_path")
    ap.add_argument("-c", "--cutoff", type=int, default=5000)
    args = ap.parse_args()

    # Load labels from the NPZ file
    true_labels, features = load_data(args.file_path)

    true_labels = true_labels[: args.cutoff]
    features = features[: args.cutoff]

    test_labels = generate_test_labels(features)

    # Calculate BCubed Metrics
    bcubed_precision, bcubed_recall, bcubed_f_measure = calculate_bcubed_metrics(
        true_labels, test_labels
    )
    print(f"BCubed Precision: {bcubed_precision}")
    print(f"BCubed Recall: {bcubed_recall}")
    print(f"BCubed F-Measure: {bcubed_f_measure}")

    # Calculate Additional Metrics (Optional)
    ari, nmi = calculate_additional_metrics(true_labels, test_labels)
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")

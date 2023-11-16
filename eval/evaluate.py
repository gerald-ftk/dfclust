import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Check if the script is inside the 'scripts' directory
if os.path.basename(current_dir) == "scripts":
    root_dir = os.path.dirname(current_dir)
else:
    root_dir = current_dir

sys.path.append(root_dir)

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from dfclust.ogmc import OGMCGraph

def load_data(file_path):
    with np.load(file_path) as data:
        true_labels = data['labels']
        features = data['features']
    return true_labels, features

def calculate_bcubed_metrics(true_labels, test_labels):
    precision = precision_score(true_labels, test_labels, average='weighted')
    recall = recall_score(true_labels, test_labels, average='weighted')
    f_measure = f1_score(true_labels, test_labels, average='weighted')
    return precision, recall, f_measure

def calculate_additional_metrics(true_labels, test_labels):
    ari = adjusted_rand_score(true_labels, test_labels)
    nmi = normalized_mutual_info_score(true_labels, test_labels)
    return ari, nmi

def generate_test_labels(features, cutoff):
    ogmc = OGMCGraph()
    for row in features[:cutoff]:
        ogmc.add_sample(row)

    return ogmc._labels_with_noise()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('file_path')
    ap.add_argument('-c', '-cutoff', type=int, default=5000)
    args = ap.parse_args()

    # Load labels from the NPZ file
    true_labels, features = load_data(args.file_path)

    test_labels = generate_test_labels(features, args.cutoff)

    # Calculate BCubed Metrics
    bcubed_precision, bcubed_recall, bcubed_f_measure = calculate_bcubed_metrics(true_labels, test_labels)
    print(f"BCubed Precision: {bcubed_precision}")
    print(f"BCubed Recall: {bcubed_recall}")
    print(f"BCubed F-Measure: {bcubed_f_measure}")

    # Calculate Additional Metrics (Optional)
    ari, nmi = calculate_additional_metrics(true_labels, test_labels)
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")

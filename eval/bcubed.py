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
from dfclust.ogmc import OGMCGraph
from tqdm import tqdm


def bcubed_score(l1: np.ndarray, l2: np.ndarray) -> float:
    """
    Calculate the BCubed F-score for evaluating clustering performance.

    This function computes the BCubed precision, recall, and F-score based on
    the clustering labels (l1) and the ground truth labels (l2). It handles
    different label sets and is suitable for evaluating clustering where the
    actual values of cluster labels may vary.

    Parameters:
    l1 (np.ndarray): The labels obtained from clustering.
    l2 (np.ndarray): The ground truth labels of the dataset.

    Returns:
    float: The BCubed F-score.

    Note:
    Both l1 and l2 should be 1D arrays of the same length.
    """
    unique_labels_l1 = np.unique(l1)
    unique_labels_l2 = np.unique(l2)

    clusters = {label: np.where(l1 == label)[0] for label in unique_labels_l1}
    truths = {label: np.where(l2 == label)[0] for label in unique_labels_l2}

    total_precision = 0
    total_recall = 0

    for idx in range(len(l1)):
        cluster_label = l1[idx]
        truth_label = l2[idx]

        intersection_size = len(np.intersect1d(clusters[cluster_label], truths[truth_label]))
        precision = intersection_size / len(clusters[cluster_label])
        recall = intersection_size / len(truths[truth_label])

        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / len(l1)
    avg_recall = total_recall / len(l1)

    if avg_precision + avg_recall == 0:
        return 0
    f_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    return f_score

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='data/4339.npz')
    args = ap.parse_args()

    with np.load(args.dataset) as f:
        features = f['features']
        labels = f['labels']

    ogmc = OGMCGraph()

    with tqdm(total=len(features), desc="BCubed F-score: N/A") as pbar:
        for i, feature in enumerate(features):
            ogmc.add_sample(feature)

            if i > 0:  # Compute F-score only if there is more than one label
                f_score = bcubed_score(ogmc._labels_with_noise[:i+1], labels[:i+1])
                pbar.set_description(f"BCubed F-score: {f_score:.4f}")

            # Update progress bar
            pbar.update(1)
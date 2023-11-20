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
from dfclust.ogmc import OGMCGraph
from sklearn.metrics.cluster import (
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
import numpy as np
from tqdm import tqdm


def load_npz_features(npz_path):
    data = np.load(npz_path)
    return data["features"]


def calculate_bcubed_score(labels, ground_truth):
    homogeneity = homogeneity_score(ground_truth, labels)
    completeness = completeness_score(ground_truth, labels)
    v_measure = v_measure_score(ground_truth, labels)
    return homogeneity, completeness, v_measure


def main(directory_path, cutoff=5000):
    graph = OGMCGraph()
    ground_truth = []
    class_labels = {}
    label_counter = 0

    # First pass to count total features for progress bar (up to cutoff)
    total_features = 0
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".npz"):
                feature_path = os.path.join(subdir, file)
                features = load_npz_features(feature_path)
                total_features += len(features)
                if total_features > cutoff:
                    total_features = cutoff
                    break
        if total_features >= cutoff:
            break

    # Second pass to add features (up to cutoff)
    features_added = 0
    with tqdm(total=total_features, desc="Adding samples") as pbar:
        for subdir, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".npz"):
                    feature_path = os.path.join(subdir, file)
                    features = load_npz_features(feature_path)

                    # Assign or retrieve the label for this class
                    if file not in class_labels:
                        class_labels[file] = label_counter
                        label_counter += 1
                    current_label = class_labels[file]

                    for feature in features:
                        graph.add_sample(feature)
                        ground_truth.append(current_label)
                        pbar.update(1)
                        features_added += 1
                        if features_added == cutoff:
                            break

                if features_added >= cutoff:
                    break

            if features_added >= cutoff:
                break

    labels = graph._labels_with_noise
    homogeneity, completeness, v_measure = calculate_bcubed_score(labels, ground_truth)
    print(f"B-Cubed Homogeneity: {homogeneity}")
    print(f"B-Cubed Completeness: {completeness}")
    print(f"B-Cubed V-measure: {v_measure}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dirpath", help="path to directory of npz's")
    ap.add_argument("-c", "--cutoff", type=int, default=5000)
    args = ap.parse_args()

    directory_path = args.dirpath
    main(directory_path, args.cutoff)

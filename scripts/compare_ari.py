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
from dfclust.ogmc import OGMCGraph
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm


def compare_clustering_with_labels(file_path: str, cutoff=5000) -> float:
    # Load the data and labels from the npz file
    data = np.load(file_path)
    samples = data["features"][:cutoff]
    true_labels = data["labels"][:cutoff]

    # Initialize and populate the OGMCGraph with the samples
    graph = OGMCGraph()
    for sample in tqdm(samples[:cutoff], desc="Adding samples"):
        graph.add_sample(sample)

    # Obtain the clustering labels from OGMCGraph
    predicted_labels = graph._labels_with_noise

    # Use the Adjusted Rand Score to compare the true labels and predicted labels
    score = adjusted_rand_score(true_labels, predicted_labels)

    return score


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cutoff", type=int, default=5000)
    args = ap.parse_args()

    file_path = f"{root_dir}/data/test.npz"
    score = compare_clustering_with_labels(file_path, cutoff=args.cutoff)
    print(f"Adjusted Rand Score: {score:.4f}")

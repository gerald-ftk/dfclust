import numpy as np
from dfclust.ogmc import OGMCGraph
from sklearn.metrics import adjusted_rand_score

def compare_clustering_with_labels(file_path: str, cutoff=5000) -> float:
    # Load the data and labels from the npz file
    data = np.load(file_path)
    samples = data['features'][:cutoff]
    true_labels = data['labels'][:cutoff]
    
    # Initialize and populate the OGMCGraph with the samples
    graph = OGMCGraph()
    for i, sample in enumerate(samples):
        graph.add_sample(sample)
        print(f'{i+1} samples added out of {len(samples)}\t\t\r', end='')
    print()
    
    # Obtain the clustering labels from OGMCGraph
    predicted_labels = graph._labels
    
    # Use the Adjusted Rand Score to compare the true labels and predicted labels
    score = adjusted_rand_score(true_labels, predicted_labels)
    
    return score

if __name__ == '__main__':
    # Path to the test.npz file
    file_path = "data/test.npz"
    score = compare_clustering_with_labels(file_path)
    print(f"Adjusted Rand Score: {score:.4f}")
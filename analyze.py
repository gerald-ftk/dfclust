import numpy as np
from scipy.spatial.distance import euclidean, cosine, pdist, squareform


def compute_statistics(features, labels):
    print("Starting computation...\n")

    # 1) Min/Max/Avg distance between labels
    print("Computing min/max/med distances within clusters...")
    dist_stats = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != -1:  # Exclude noise
            cluster_features = features[labels == label]
            pairwise_distances = squareform(pdist(cluster_features, "cosine"))
            min_dist = np.min(
                pairwise_distances
                + np.eye(pairwise_distances.shape[0]) * np.max(pairwise_distances)
            )
            max_dist = np.max(pairwise_distances)
            avg_dist = (
                np.median(pairwise_distances)
            )  # Divided by 2 because pairwise_distances counts each pair twice
            dist_stats[label] = (min_dist, max_dist, avg_dist)
    print("Min/max/med distances computation done.\n")

    # 2) Cosine/Euclidean distance between means of clusters
    print("Computing cosine and euclidean distances between cluster centroids...")
    centroids = {}
    for label in unique_labels:
        if label != -1:
            cluster_features = features[labels == label]
            centroids[label] = np.mean(cluster_features, axis=0)

    centroid_pairs = list(centroids.keys())
    centroid_distances = {}
    for i in range(len(centroid_pairs)):
        for j in range(i + 1, len(centroid_pairs)):
            label_i, label_j = centroid_pairs[i], centroid_pairs[j]
            euclidean_dist = euclidean(centroids[label_i], centroids[label_j])
            cosine_dist = cosine(centroids[label_i], centroids[label_j])
            centroid_distances[(label_i, label_j)] = (euclidean_dist, cosine_dist)
    print("Distances between cluster centroids computation done.\n")

    # 3) Maximum distance between features in the same cluster
    print("Computing maximum distances between features in the same cluster...")
    max_cluster_dist = {}
    for label in unique_labels:
        if label != -1:
            cluster_features = features[labels == label]
            pairwise_distances = squareform(pdist(cluster_features, "euclidean"))
            max_cluster_dist[label] = np.max(pairwise_distances)
    print("Maximum distance computation done.\n")

    # Print results
    print("Results:\n")
    print("Min/Max/Avg distances within clusters:")
    for label, (min_dist, max_dist, avg_dist) in dist_stats.items():
        print(
            f"Label {label}: Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}"
        )

    print("\nEuclidean & Cosine distances between cluster centroids:")
    for (label_i, label_j), (euclidean_dist, cosine_dist) in centroid_distances.items():
        print(
            f"Labels {label_i} & {label_j}: Euclidean: {euclidean_dist:.2f}, Cosine: {cosine_dist:.2f}"
        )

    print("\nMaximum distance between features in the same cluster:")
    for label, max_dist in max_cluster_dist.items():
        print(f"Label {label}: Max Distance: {max_dist:.2f}")

    print("\nComputation completed!")


# Run the function on the loaded data (if it was successfully loaded)
# compute_statistics(features, labels)


def main():
    # Load the features from the provided path
    data_path = "data/test.npz"
    print(f"Loading data from {data_path}...\n")
    data = np.load(data_path)
    features = data["features"]
    labels = data["labels"]

    compute_statistics(features, labels)


main()

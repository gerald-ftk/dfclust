import numpy as np
from scipy.spatial.distance import euclidean, cdist
from sklearn.preprocessing import normalize
from typing import List, Dict, Set


class OGMCluster:
    """A class to represent a single cluster within OGMC."""

    def __init__(self, graph: "OGMCGraph") -> None:
        """
        Initialize the OGMCluster.
        """
        self.sample_indices = []
        self.sum: np.ndarray = np.zeros(512)  # Sum of samples in the cluster
        self.graph = graph
        self.centroid = self.sum

    def add_sample_by_index(self, i: int) -> None:
        """
        Add a sample to this cluster from the index of parent Graph.

        Parameters:
        - i (int): Index of the sample in the parent graphs sample list.
        """

        self.sample_indices.append(i)
        self.sum += self.graph.samples[i]
        self.centroid = self.sum / len(self.sample_indices)

    def samples(self):
        return [self.graph.samples[i] for i in self.sample_indices]

    def __len__(self):
        return len(self.sample_indices)

    def __len__(self) -> int:
        return len(self.sample_indices) - 1


class OGMCGraph:
    """A class to represent the OGMC graph."""

    def __init__(
        self,
        fusion: float = 0.7,
        nsr: int = 5,
        thr_sc: float = 0.99,
        thr_wc: float = 1.12,
    ) -> None:
        """
        Initialize the OGMC graph.

        Parameters:
        - fusion (float): The fusion threshold for clustering.
        """
        self.samples: List[np.ndarray] = []  # List to store normalized samples

        # Store clusters with their id's in a dict
        self.clusters: Dict[int, OGMCluster] = {}

        # Thresholds
        self.fusion = fusion
        self.nsr = nsr
        self.thr_sc = thr_sc
        self.thr_wc = thr_wc

        # Increment this counter as new clusters are added
        self.id_counter = 0

        # Keep track of connections with a dict
        self.connections: Dict[int, Set[int]] = {}

    @property
    def _labels(self) -> np.ndarray:
        """
        Return the labels (i.e., cluster IDs) for each sample, taking connections into account.

        Returns:
        - np.ndarray: An array where the value at index i represents the cluster ID of the i-th sample.
        """
        labels = np.empty(len(self.samples), dtype=int)

        # Cache for clusters that have been already processed
        processed_clusters = set()

        for cluster_id, cluster in self.clusters.items():
            if cluster_id in processed_clusters:
                continue

            # If cluster is not None and it's not been processed
            if cluster:
                # Get the connected clusters for the current cluster
                connected_clusters = self.get_connected_clusters(cluster_id)

                # Add the current cluster to the list
                connected_clusters.add(cluster_id)

                # Find the cluster with the smallest ID among the connected clusters
                representative_id = min(connected_clusters)

                for conn_cluster_id in connected_clusters:
                    # Mark the cluster as processed
                    processed_clusters.add(conn_cluster_id)

                    # Assign the representative ID to all samples in the connected cluster
                    for sample_idx in self.clusters[conn_cluster_id].sample_indices:
                        labels[sample_idx] = representative_id

        return labels

    @property
    def _labels_with_noise(self, min_cluster_size=10):
        """
        Return labels for each sample, but if a label's count is less than X,
        replace it with -1 indicating noise.

        Parameters:
        - X: int, minimum count threshold below which labels are treated as noise.

        Returns:
        - np.ndarray: labels for each sample.
        """
        labels = (
            self._labels
        )  # Assuming _labels is a property/method that returns the original labels

        # Count occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Find labels that occur less than X times
        noise_labels = unique_labels[counts < min_cluster_size]

        # Replace infrequent labels with -1
        for noise_label in noise_labels:
            labels[labels == noise_label] = -1

        return labels

    def create_cluster(self, idx: int) -> int:
        """Create a new cluster, add a sample to it, and store the cluster in the graph.

        Parameters:
        - idx (int): The index of the sample to be added to the new cluster.

        Returns:
        - int: The ID of the newly created cluster.
        """
        cluster = OGMCluster(graph=self)
        cluster.add_sample_by_index(idx)

        cluster_id = self.id_counter
        self.clusters[cluster_id] = cluster
        self.connections[cluster_id] = set()
        self.id_counter += 1

        return cluster_id

    def add_sample(self, f: np.ndarray) -> int:
        """Add a sample to the graph, possibly creating a new cluster or updating an existing one.

        Parameters:
        - f (np.ndarray): The sample to be added.

        Returns:
        - int: The index of the added sample within the graph.
        """
        # Normalize the sample
        fn = normalize(f.reshape(1, -1))
        self.samples.append(fn[0])
        fn_idx = len(self.samples) - 1

        # If this is the first sample, create a new cluster and return
        if len(self.clusters) == 0:
            self.create_cluster(fn_idx)
            return fn_idx

        # Compute distances between the new sample (fn) and existing cluster centroids
        valid_keys, centroids = zip(
            *[(k, c.centroid) for k, c in self.clusters.items() if c is not None]
        )

        centroids = np.array(centroids)

        cdists = cdist(fn, centroids).flatten()
        dists = {i: dist for i, dist in zip(valid_keys, cdists)}

        min_idx = min(dists, key=dists.get)  # index of closest cluster centroid
        min_dist = dists[min_idx]  # distance to closest cluster centroid
        min_cluster = self.clusters[min_idx]  # reference to closest cluster centroid

        if min_dist >= self.fusion:
            new_cluster_idx = self.create_cluster(idx=fn_idx)
            if (min_dist < self.thr_wc) and (len(min_cluster) >= self.nsr):
                self.connect_clusters(new_cluster_idx, min_idx)
                self._check_connections()
            else:
                return fn_idx
        else:
            self.clusters[min_idx].add_sample_by_index(fn_idx)
            if len(self.clusters) > 1:
                self.recluster(min_idx)

        return fn_idx

    def fuse_clusters(self, i1: int, i2: int) -> None:
        """Fuse one cluster into another and set the fused cluster's entry to None.

        This method merges the samples of the second cluster into the first cluster,
        updates the cumulative sum of the first cluster, manages the connections,
        and then sets the second cluster's entry in the clusters dictionary to None
        (instead of removing it) to ensure the indices of the remaining clusters
        are not affected.

        Parameters:
        - i1 (int): The ID of the first cluster (the one that will remain after fusion).
        - i2 (int): The ID of the second cluster (the one that will be fused).

        Returns:
        - None
        """
        cluster1 = self.clusters[i1]
        cluster2 = self.clusters[i2]

        # 1. Merge samples of cluster2 into cluster1
        for sample_idx in cluster2.sample_indices:
            cluster1.add_sample_by_index(sample_idx)

        # 2. Update cumulative sum of cluster1 (it's done automatically in add_sample)

        # 3. Update connections
        for connected_cluster in set(self.connections[i2]):
            # If the connected cluster is not cluster1, update its connections
            if connected_cluster != i1:
                self.connections[connected_cluster].remove(i2)
                self.connections[connected_cluster].add(i1)
                self.connections[i1].add(connected_cluster)

        # Remove connections of cluster2
        del self.connections[i2]

        # 4. Set the cluster at this index to None to avoid changing indices of other clusters
        self.clusters[i2] = None

    def _check_connections(self):
        """Check and ensure all connections are still valid."""
        invalid_connections = []

        for cluster_idx, connected_clusters in self.connections.items():
            # Check if the cluster exists
            if cluster_idx >= len(self.clusters) or self.clusters[cluster_idx] is None:
                invalid_connections.append(cluster_idx)
                continue

            # Check connected clusters
            for connected_cluster_idx in connected_clusters:
                # Check if the connected cluster exists
                if (
                    connected_cluster_idx >= len(self.clusters)
                    or self.clusters[connected_cluster_idx] is None
                ):
                    invalid_connections.append((cluster_idx, connected_cluster_idx))
                    continue

                # Check if the connected cluster acknowledges the connection back
                if cluster_idx not in self.connections.get(
                    connected_cluster_idx, set()
                ):
                    invalid_connections.append((cluster_idx, connected_cluster_idx))

        # Removing invalid connections
        for connection in invalid_connections:
            if isinstance(connection, tuple):
                # If connection is a tuple, it refers to a bidirectional connection
                self.connections[connection[0]].discard(connection[1])
            else:
                # If connection is a single value, it means the cluster doesn't exist anymore
                del self.connections[connection]

        # if invalid_connections:
        #     print(invalid_connections)
        return invalid_connections  # Optionally return the list of invalid connections for debugging/info

    def connect_clusters(self, i1, i2):
        """Connect two clusters."""

        # Ensure clusters are not the same
        if i1 == i2:
            raise ValueError("A cluster cannot be connected to itself.")

        # Ensure both clusters have an entry in the connections dictionary
        if i1 not in self.connections and (i1 in self.clusters):
            self.connections[i1] = set()
        if i2 not in self.connections and (i2 in self.clusters):
            self.connections[i2] = set()

        # Establish the connection
        self.connections[i1].add(i2)
        self.connections[i2].add(i1)

    def disconnect_clusters(self, i1: int, i2: int):
        """Disconnect two clusters."""
        self.connections[i1].remove(i2)
        self.connections[i2].remove(i1)

    def get_connected_clusters(self, cluster_idx):
        """Return all clusters connected to the given cluster."""
        return self.connections.get(cluster_idx, [])

    def recluster(self, u_idx: int) -> None:
        """Trigger reclustering process

        Args:
            u_idx (int): The index of the cluster that was updated
        """

        u_clust = self.clusters[u_idx]

        # Compute distances between u_clust and existing cluster centroids

        # First, get the keys that are valid, as some may be None
        valid_keys, centroids = zip(
            *[(k, c.centroid) for k, c in self.clusters.items() if c is not None]
        )
        centroids = np.array(centroids)

        cdists = cdist(u_clust.centroid.reshape(1, -1), centroids).flatten()
        dists = {i: dist for i, dist in zip(valid_keys, cdists)}

        # Remove u_func's centroid from the dict
        dists.pop(u_idx)

        if not dists:
            return

        # Determine if the number of samples in the min_cluster is greater than the threshold
        min_idx = min(dists, key=dists.get)
        min_clust = self.clusters[min_idx]
        u_min_dist = dists[min_idx]

        if len(u_clust) >= self.nsr:
            if (u_min_dist <= self.fusion) and (len(min_clust) < self.nsr):
                # Fuse u_clust and min_clust
                self.fuse_clusters(u_idx, min_idx)
                self.recluster(u_idx)

            elif u_min_dist <= self.thr_sc:
                # Connect u_clust and min_clust
                self.connect_clusters(u_idx, min_idx)

            elif (u_min_dist <= self.thr_wc) and len(min_clust) < self.nsr:
                # Connect uclust and min_clust
                self.connect_clusters(u_idx, min_idx)

        elif u_min_dist <= self.fusion:
            # Fuse u_clust and min_clust
            self.fuse_clusters(u_idx, min_idx)

        elif (u_min_dist <= self.thr_wc) and len(min_clust) >= self.nsr:
            # Connect u_clust and min_clust
            self.connect_clusters(u_idx, min_idx)

        self._check_connections()


if __name__ == "__main__":
    with np.load("data/test.npz") as npz:
        features = npz["features"]

    graph = OGMCGraph()
    min_samples = 10
    print()
    for i, f in enumerate(features):
        graph.add_sample(f)
        unique, counts = np.unique(graph._labels, return_counts=True)
        count = np.sum(counts > min_samples)

        print(
            f"samples: {i+1}/{features.shape[0]}, clusters: {len(graph.clusters)}, "
            f"clusters above {min_samples} samples: {count}\t\r",
            end="",
        )

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
        self.sample_indices = set()
        self.sum: np.ndarray = np.zeros(512)  # Sum of samples in the cluster
        self.graph = graph
        self.centroid = self.sum
        self.connections: Dict[int, float] = {}

    def add_sample_by_index(self, i: int) -> None:
        """
        Add a sample to this cluster from the index of parent Graph.

        Parameters:
        - i (int): Index of the sample in the parent graphs sample list.
        """

        self.sample_indices.add(i)
        self.sum += self.graph.samples[i]
        self.centroid = self.sum / len(self.sample_indices)

    def merge_cluster(self, other_cluster: "OGMCluster") -> None:
        """
        Merge another cluster into this cluster.

        Parameters:
        - other_cluster (OGMCluster): The other cluster to merge into this one.

        Returns:
        - None
        """
        # Merge the sample indices from the other cluster into this one
        # This assumes you have a structure to hold sample indices in your cluster
        self.sample_indices.update(other_cluster.sample_indices)

        # Merge connections from the other cluster into this one
        for connected_label, distance in other_cluster.connections.items():
            self.add_connection(connected_label, distance)

    def add_connection(self, cluster: int, distance: float) -> None:
        self.connections[cluster] = distance

    def delete_connection(self, cluster: int) -> None:
        if cluster in self.connections:
            del self.connections[cluster]

    def get_connected_clusters(self):
        """Get a list of clusters that are connected to this cluster.

        Returns:
            A list of cluster indices that are connected to this cluster.
        """
        # Return the list of connected cluster indices
        return list(self.connections.keys())

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
        ncr: int = 5,
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
        self.ncr = ncr
        self.thr_sc = thr_sc
        self.thr_wc = thr_wc

        # Increment this counter as new clusters are added
        self.id_counter = 0

    @property
    def _labels(self) -> np.ndarray:
        """
        Return the labels (i.e., cluster IDs) for each sample, taking connections into account.

        Returns:
            np.ndarray: An array where the value at index i represents the cluster ID of the i-th sample.
        """
        labels = np.empty(len(self.samples), dtype=int)

        # Cache for clusters that have been already processed
        processed_clusters = set()

        for cluster_id, cluster in self.clusters.items():
            if cluster_id in processed_clusters:
                continue

            if cluster:
                # Get the connected clusters for the current cluster
                connected_cluster_ids = [
                    conn_id for conn_id, _ in cluster.connections.items()
                ]
                connected_cluster_ids.append(cluster_id)  # Include current cluster

                # Find the cluster with the smallest ID among the connected clusters
                representative_id = min(connected_cluster_ids)

                for conn_cluster_id in connected_cluster_ids:
                    # Mark the cluster as processed
                    processed_clusters.add(conn_cluster_id)

                    # Assign the representative ID to all samples in the connected cluster
                    for sample_idx in self.clusters[conn_cluster_id].sample_indices:
                        labels[sample_idx] = representative_id

        return labels

    @property
    def _labels_with_noise(self, min_cluster_size=10) -> np.ndarray:
        """
        Return labels for each sample, but if a label's count is less than min_cluster_size,
        replace it with -1 indicating noise.

        Parameters:
        - min_cluster_size: int, minimum count threshold below which labels are treated as noise.

        Returns:
        - np.ndarray: labels for each sample.
        """
        labels = self._labels  # Retrieve the original labels

        # Count occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Find labels that occur less than min_cluster_size times
        noise_labels = unique_labels[counts < min_cluster_size]

        # Replace infrequent labels with -1
        is_noise = np.isin(labels, noise_labels)
        labels[is_noise] = -1

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
        """
        Fuse one cluster into another and set the fused cluster's entry to None.

        This method merges the samples and connections of the second cluster into the first cluster,
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

        # Assuming cluster1 has a method merge_cluster that handles the merging of samples and connections.
        cluster1.merge_cluster(cluster2)

        # Now we need to update all connections in the graph that pointed to i2 to point to i1
        for cluster in self.clusters.values():
            if cluster and i2 in cluster.connections:
                distance = cluster.connections.pop(i2)
                cluster.add_connection(i1, distance)  # Update to the new connection

        # Finally, we mark the second cluster as None to indicate it's been fused
        self.clusters[i2] = None

    def _check_connections(self):
        """Check and ensure all connections are still valid."""
        invalid_connections = []

        for cluster_idx, cluster in self.clusters.items():
            if cluster is None:
                continue  # Skip deleted clusters

            # Create a list of connections to remove to avoid modifying the dictionary during iteration
            connections_to_remove = []

            for conn_idx, distance in cluster.connections.items():
                # Check if the connected cluster exists
                if conn_idx >= len(self.clusters) or self.clusters[conn_idx] is None:
                    connections_to_remove.append(conn_idx)
                    continue

                # Check if the connected cluster acknowledges the connection back
                connected_cluster = self.clusters[conn_idx]
                if cluster_idx not in connected_cluster.connections:
                    connections_to_remove.append(conn_idx)

            # Remove invalid connections for the current cluster
            for conn_idx in connections_to_remove:
                del cluster.connections[conn_idx]
                invalid_connections.append((cluster_idx, conn_idx))

            # Additionally, remove the invalid connections from the connected clusters
            for conn_idx in connections_to_remove:
                if (
                    conn_idx < len(self.clusters)
                    and self.clusters[conn_idx] is not None
                ):
                    self.clusters[conn_idx].connections.pop(cluster_idx, None)

        return invalid_connections

    def connect_clusters(self, i1, i2):
        """Connect two clusters."""

        dist = euclidean(self.clusters[i1].centroid, self.clusters[i2].centroid)
        # Ensure clusters are not the same
        if i1 == i2:
            raise ValueError("A cluster cannot be connected to itself.")

        # Establish the connection
        self.clusters[i1].connections[i2] = dist
        self.clusters[i2].connections[i1] = dist

    def disconnect_clusters(self, i1: int, i2: int):
        """Disconnect two clusters."""
        del self.clusters[i1].connections[i2]
        del self.clusters[i2].connections[i1]

    def get_connected_clusters(self, cluster_idx: int):
        """
        Return all clusters connected to the given cluster.

        Args:
            cluster_idx (int): The index of the cluster.

        Returns:
            List[Tuple[int, float]]: A list of tuples where each tuple contains the index of a connected cluster
                                     and the distance to that cluster.
        """
        cluster = self.clusters.get(cluster_idx)
        if not cluster:
            return []  # If the cluster doesn't exist, return an empty list

        # The items of the connections dictionary are tuples of (cluster_label, distance)
        return list(cluster.connections.items())

    def recluster(self, u_idx: int) -> None:
        """Trigger reclustering process

        Args:
            u_idx (int): The index of the cluster that was updated
        """
        u_clust = self.clusters[u_idx]

        # Compute distances between u_clust and existing cluster centroids
        # First, get the valid clusters as some may be None
        valid_clusters = {
            k: c for k, c in self.clusters.items() if c is not None and k != u_idx
        }

        # If there are no other clusters to compare with, return
        if not valid_clusters:
            return

        # Compute the distances from u_clust to all other clusters
        dists = {
            k: euclidean(u_clust.centroid, c.centroid)
            for k, c in valid_clusters.items()
        }

        # Determine if the number of samples in the min_cluster is greater than the threshold
        min_idx, u_min_dist = min(dists.items(), key=lambda item: item[1])
        min_clust = self.clusters[min_idx]

        if len(u_clust.sample_indices) >= self.nsr:
            if (u_min_dist <= self.fusion) and (
                len(min_clust.sample_indices) < self.nsr
            ):
                # Fuse u_clust and min_clust
                self.fuse_clusters(u_idx, min_idx)
                self.recluster(u_idx)
            elif u_min_dist <= self.thr_sc:
                # Connect u_clust and min_clust
                u_clust.add_connection(min_idx, u_min_dist)
                min_clust.add_connection(u_idx, u_min_dist)
            elif (u_min_dist <= self.thr_wc) and (
                len(min_clust.sample_indices) < self.nsr
            ):
                # Connect u_clust and min_clust
                u_clust.add_connection(min_idx, u_min_dist)
                min_clust.add_connection(u_idx, u_min_dist)

        elif u_min_dist <= self.fusion:
            # Fuse u_clust and min_clust
            self.fuse_clusters(u_idx, min_idx)

        elif (u_min_dist <= self.thr_wc) and (
            len(min_clust.sample_indices) >= self.nsr
        ):
            # Connect u_clust and min_clust
            u_clust.add_connection(min_idx, u_min_dist)
            min_clust.add_connection(u_idx, u_min_dist)

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

import numpy as np
from scipy.spatial.distance import euclidean, cdist
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple
import heapq


class OGConnections:
    def __init__(self, max_length: int) -> None:
        """
        Initialize the OGConnections object with a specified maximum length.

        :param max_length: The maximum number of connections (distance-label pairs) to store.
        """
        self.max_length = max_length
        self.data: List[Tuple[float, int]] = []

    def add(self, distance: float, label: int) -> None:
        """
        Add a new label-distance pair to the heap. If the heap is full, the pair with the
        smallest distance is replaced if the new distance is larger.

        :param label: The label of the new connection.
        :param distance: The distance associated with the label.
        """

        if len(self.data) < self.max_length:
            heapq.heappush(self.data, (distance, label))
        elif distance > self.data[0][0]:
            heapq.heapreplace(self.data, (distance, label))

    def get_sorted_data(self) -> List[Tuple[float, int]]:
        """
        Retrieve a sorted list of all label-distance pairs in the heap.

        :return: A list of tuples, each containing a distance and its associated label, sorted by distance.
        """
        return sorted(self.data)

    def get_all_labels(self) -> List[int]:
        """
        Retrieve all labels currently in the heap.

        :return: A list of labels.
        """
        return [label for _, label in self.data]


class OGMCluster:
    """A class to represent a single cluster within OGMC."""

    def __init__(self, graph: "OGMCGraph", nsr: int) -> None:
        """
        Initialize the OGMCluster.
        """
        self.sample_indices = set()
        self.sum: np.ndarray = np.zeros(512)  # Sum of samples in the cluster
        self.graph = graph
        self.centroid = self.sum
        self.connections = OGConnections(nsr)
        self._is_robust = False

    @property
    def is_robust(self):
        return self._is_robust

    @is_robust.setter
    def is_robust(self, value):
        if not isinstance(value, bool):
            raise ValueError("is_robust must be a boolean value")
        self._is_robust = value

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

        for i in other_cluster.sample_indices:
            self.add_sample_by_index(i)

        # Merge connections from the other cluster into this one
        for distance, connected_label in other_cluster.connections.get_sorted_data():
            if connected_label not in self.connections.get_all_labels():
                self.add_connection(distance, connected_label)

    def add_connection(self, distance: float, cluster: int) -> None:
        self.connections.add(distance, cluster)

    def get_connected_clusters(self):
        """Get a list of clusters that are connected to this cluster.

        Returns:
            A list of cluster indices that are connected to this cluster.
        """
        # Return the list of connected cluster indices
        return list(self.connections.get_all_labels())

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
        thr_f: float = 0.7,
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
        self.thr_f = thr_f  # fusion threshold
        self.nsr = nsr  # number of samples to be considered robust
        self.ncr = ncr  # maximum number of allowed connections
        self.thr_sc = thr_sc  # threshold for connecting 2 robust clusters
        self.thr_wc = thr_wc  # threshold to connect a robust to a non-robust

        # Increment this counter as new clusters are added
        self.id_counter = 0

    @property
    def _labels(self) -> np.ndarray:
        """
        Return the labels (i.e., cluster IDs) for each sample, taking connections into account.

        Each connected component of clusters receives a unique label. If clusters are not connected,
        they receive a unique label within their own component.

        Returns:
            np.ndarray: An array where the value at index i represents the cluster ID of the i-th sample.
        """
        # Initialize an array to hold the labels with a default label (e.g., -1 for unassigned)
        labels = np.full(len(self.samples), fill_value=-1, dtype=int)

        # Initialize a label counter
        label_counter = 0

        # Iterate over the clusters
        for cluster_id, cluster in self.clusters.items():
            # Skip if the cluster is None or already labeled
            if cluster is None:
                continue

            if np.any(labels[list(cluster.sample_indices)] >= 0):
                continue

            # Start a new connected component label
            current_label = label_counter
            label_counter += 1

            # Set up a queue to process all connected clusters
            queue = [cluster_id]

            while queue:
                current_cluster_id = queue.pop(0)
                current_cluster = self.clusters[current_cluster_id]

                # Skip if the cluster is None
                if current_cluster is None:
                    continue

                # Label the samples in the current cluster
                sample_indices_list = list(current_cluster.sample_indices)
                labels[sample_indices_list] = current_label

                # Add connected clusters to the queue if they haven't been processed yet
                for (
                    connected_cluster_id
                ) in current_cluster.connections.get_all_labels():
                    cc = self.clusters[connected_cluster_id]
                    if cc is not None:
                        if np.any(
                            labels[
                                list(self.clusters[connected_cluster_id].sample_indices)
                            ]
                            < 0
                        ):
                            queue.append(connected_cluster_id)

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
        cluster = OGMCluster(graph=self, nsr=self.nsr)
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

        if min_dist <= self.thr_f:
            self.clusters[min_idx].add_sample_by_index(fn_idx)
            if len(self.clusters[min_idx]) > self.nsr:
                self.clusters[min_idx].is_robust = True

            if len(self.clusters) > 1:
                self.recluster(*self.compute_distances(min_idx))
            else:
                return fn_idx
        else:
            new_cluster_idx = self.create_cluster(fn_idx)
            if (min_dist <= self.thr_wc) and (len(min_cluster) >= self.nsr):
                self.connect_clusters(new_cluster_idx, min_idx)

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

        cluster1.merge_cluster(cluster2)

        # Update all connections in the graph that pointed to i2 to point to i1
        # for cluster in self.clusters.values():
        #     if cluster and i2 in cluster.connections:
        #         distance = cluster.connections.pop(i2)
        #         cluster.add_connection(i1, distance)  # Update to the new connection

        # Finally, we mark the second cluster as None to indicate it's been fused
        self.clusters[i2] = None

        if len(cluster1) > self.nsr:
            cluster1.is_robust = True

    def connect_clusters(self, i1, i2):
        """Connect two clusters."""

        dist = euclidean(self.clusters[i1].centroid, self.clusters[i2].centroid)
        # Ensure clusters are not the same
        if i1 == i2:
            raise ValueError("A cluster cannot be connected to itself.")

        # Establish the connection
        self.clusters[i1].connections.add(dist, i2)
        self.clusters[i2].connections.add(dist, i1)

    # def get_connected_clusters(self, cluster_idx: int):
    #     """
    #     Return all clusters connected to the given cluster.

    #     Args:
    #         cluster_idx (int): The index of the cluster.

    #     Returns:
    #         List[Tuple[int, float]]: A list of tuples where each tuple contains the index of a connected cluster
    #                                  and the distance to that cluster.
    #     """
    #     cluster = self.clusters.get(cluster_idx)
    #     if not cluster:
    #         return []  # If the cluster doesn't exist, return an empty list

    #     # The items of the connections dictionary are tuples of (cluster_label, distance)
    #     return list(cluster.connections.get_sorted_data())

    def compute_distances(self, u_idx: int) -> None:
        """Trigger reclustering process

        Args:
            u_idx (int): The index of the cluster that was updated
        """
        u_clust = self.clusters[u_idx]

        # Filter out the u_clust and any None clusters
        valid_clusters = [
            (k, c.centroid)
            for k, c in self.clusters.items()
            if c is not None and k != u_idx
        ]

        # If there are no valid clusters after filtering, return
        if not valid_clusters:
            return u_idx, np.array([]), np.array([])

        # Unpack the keys and centroids into separate lists
        valid_keys, valid_centroids = zip(*valid_clusters)
        valid_keys = np.array(valid_keys)
        valid_centroids = np.array(valid_centroids)

        # Compute the distances from u_clust to all other clusters
        cdists = cdist(np.array([u_clust.centroid]), valid_centroids).flatten()

        return u_idx, valid_keys, cdists

    def recluster(self, u_idx: int, valid_keys: np.ndarray, cdists: np.ndarray):
        while valid_keys.size > 0:
            u_clust = self.clusters[u_idx]

            c_min = np.argmin(cdists)
            # Determine if the number of samples in the min_cluster is greater than the threshold
            min_idx = valid_keys[c_min]
            u_min_dist = cdists[c_min]
            min_clust = self.clusters[min_idx]

            # ns[uIdx] >= nsr
            if u_clust.is_robust:
                # dist[min_idx] <= thrf and ns[minIdx] < nsr
                if (u_min_dist <= self.thr_f) and not min_clust.is_robust:
                    # Fuse u_clust and min_clust
                    self.fuse_clusters(u_idx, min_idx)
                    u_idx, valid_keys, cdists = self.compute_distances(u_idx)

                # dist[minIdx] <= thrsc
                elif u_min_dist <= self.thr_sc:
                    # Connect u_clust and min_clust
                    u_clust.add_connection(u_min_dist, min_idx)
                    min_clust.add_connection(u_min_dist, u_idx)

                    # print(f'Adding connection to {u_clust}, total: {len(u_clust.connections)}')

                    # Remove the min_idx and its corresponding dist
                    valid_keys = np.delete(valid_keys, c_min)
                    cdists = np.delete(cdists, c_min)

                # dist[minIdx] <= thrwc and ns[minIdx] < nsr
                elif (u_min_dist <= self.thr_wc) and not min_clust.is_robust:
                    # Connect u_clust and min_clust
                    u_clust.add_connection(u_min_dist, min_idx)
                    min_clust.add_connection(u_min_dist, u_idx)

                    # print(f'Adding connection to {u_clust}, total: {len(u_clust.connections)}')

                    # Remove the min_idx and its corresponding dist
                    valid_keys = np.delete(valid_keys, c_min)
                    cdists = np.delete(cdists, c_min)

            # dist[minIdx] <= thrf
            elif u_min_dist <= self.thr_f:
                # Fuse u_clust and min_clust
                self.fuse_clusters(u_idx, min_idx)
                u_idx, valid_keys, cdists = self.compute_distances(u_idx)

            # dist[minidx] <= thrwc and ns[minidx] >= nsr
            elif (u_min_dist <= self.thr_wc) and min_clust.is_robust:
                # Connect u_clust and min_clust
                u_clust.add_connection(u_min_dist, min_idx)
                min_clust.add_connection(u_min_dist, u_idx)

                # print(f'Adding connection to {u_idx}, total: {len(u_clust.connections)}')

                # Remove the min_idx and its corresponding dist
                valid_keys = np.delete(valid_keys, c_min)
                cdists = np.delete(cdists, c_min)

            return


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

        # print(
        #     f"samples: {i+1}/{features.shape[0]}, clusters: {len(graph.clusters)}, "
        #     f"clusters above {min_samples} samples: {count}\t\r",
        #     end="",
        # )

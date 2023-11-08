from dfclust.ogmc import OGMCGraph, OGMCluster
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
import numpy as np
import unittest


class TestOGMC(unittest.TestCase):
    def setUp(self):
        """Set up a fresh OGMCGraph instance and an associated OGMCluster instance for each test."""
        self.graph = OGMCGraph()
        self.cluster = OGMCluster(self.graph)

    # Tests related to OGMCGraph
    def test_initialization(self):
        """Test that a new OGMCGraph instance is correctly initialized."""
        self.assertEqual(len(self.graph.samples), 0)
        self.assertEqual(len(self.graph.clusters), 0)

    def test_add_sample(self):
        """Test adding a single sample to the graph."""
        sample = normalize(np.array([1.0] * 512).reshape(1, -1))[0]
        self.graph.add_sample(sample)
        self.assertTrue(np.allclose(self.graph.samples[0], sample))

    def test_connect_clusters(self):
        """Test connecting two clusters in the graph."""
        cluster1 = OGMCluster(self.graph)
        cluster2 = OGMCluster(self.graph)
        self.graph.clusters[0] = cluster1
        self.graph.clusters[1] = cluster2
        distance = euclidean(cluster1.centroid, cluster2.centroid)
        cluster1.add_connection(1, distance)
        cluster2.add_connection(0, distance)
        self.assertIn(1, cluster1.connections)
        self.assertIn(0, cluster2.connections)

    def test_disconnect_clusters(self):
        """Test disconnecting two previously connected clusters in the graph."""
        cluster1 = OGMCluster(self.graph)
        cluster2 = OGMCluster(self.graph)
        self.graph.clusters[0] = cluster1
        self.graph.clusters[1] = cluster2
        distance = euclidean(cluster1.centroid, cluster2.centroid)
        cluster1.add_connection(1, distance)
        cluster2.add_connection(0, distance)
        cluster1.delete_connection(1)
        cluster2.delete_connection(0)
        self.assertNotIn(1, cluster1.connections)
        self.assertNotIn(0, cluster2.connections)

    def test_get_connected_clusters(self):
        """Test retrieving clusters connected to a given cluster."""
        cluster1 = OGMCluster(self.graph)
        cluster2 = OGMCluster(self.graph)
        self.graph.clusters[0] = cluster1
        self.graph.clusters[1] = cluster2
        distance = euclidean(cluster1.centroid, cluster2.centroid)
        cluster1.add_connection(1, distance)
        connected_clusters = cluster1.get_connected_clusters()
        self.assertIn(1, connected_clusters)

    def test_add_sample_cluster(self):
        """Test adding a single sample to a cluster."""
        sample = normalize(np.array([1.0] * 512).reshape(1, -1))[0]
        self.graph.add_sample(sample)
        self.cluster.add_sample_by_index(0)
        self.assertEqual(len(self.cluster.sample_indices), 1)
        self.assertIn(0, self.cluster.sample_indices)
        self.assertTrue(np.allclose(self.graph.samples, sample))

    def test_centroid(self):
        """Test centroid computation for a cluster."""
        sample1 = np.array([1.0] * 512)
        sample2 = np.array([2.0] * 512)
        self.graph.add_sample(sample1)
        self.graph.add_sample(sample2)
        self.cluster.add_sample_by_index(0)
        self.cluster.add_sample_by_index(1)
        expected_centroid = normalize((sample1 + sample2).reshape(1, -1))[0]
        self.assertTrue(np.allclose(self.cluster.centroid, expected_centroid))

    def test_multiple_distant_samples(self):
        """Test clustering of multiple distant samples."""
        # Set the random seed for consistency
        np.random.seed(42)

        samples = []

        # Generate 10 random arrays with distances > 0.7 between them
        while len(samples) < 10:
            random_array = np.random.rand(1, 512)
            normalized_array = normalize(random_array)

            # If there are no samples yet, just append
            if not samples:
                samples.append(normalized_array)
                continue

            # Check distances to existing samples using scipy's euclidean function
            is_far_enough = True
            for existing_sample in samples:
                if euclidean(normalized_array[0], existing_sample[0]) <= 0.7:
                    is_far_enough = False
                    break

            if is_far_enough:
                samples.append(normalized_array)

        # Add samples to the clusterer
        for sample in samples:
            self.graph.add_sample(sample)

        # Verify clusters were created
        self.assertEqual(len(self.graph.clusters), 10, "Expected 10 clusters!")

    def test_fusing_clusters(self):
        """Test the behavior when clusters are fused."""
        # First, add the distant samples to make some clusters
        np.random.seed(42)

        samples = []

        # Generate 10 random arrays with distances > 0.7 between them
        while len(samples) < 10:
            random_array = np.random.rand(1, 512)
            normalized_array = normalize(random_array)

            # If there are no samples yet, just append
            if not samples:
                samples.append(normalized_array)
                continue

            # Check distances to existing samples using scipy's euclidean function
            is_far_enough = True
            for existing_sample in samples:
                if euclidean(normalized_array[0], existing_sample[0]) <= 0.7:
                    is_far_enough = False
                    break

            if is_far_enough:
                samples.append(normalized_array)

        for sample in samples:
            self.graph.add_sample(sample)

        # Then add samples that are going to be merged into existing clusters
        for _ in range(10):
            self.graph.add_sample(np.random.rand(1, 512))

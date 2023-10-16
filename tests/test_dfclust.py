from dfclust.simple import SimpleDFClust
from sklearn.metrics import adjusted_rand_score
import numpy as np
import unittest


class TestSimpleDFClust(unittest.TestCase):

    def setUp(self):
        self.clusterer = SimpleDFClust(threshold=0.6)
        with np.load('data/test.npz') as f:
            self.features = f['features']
            self.labels = f['labels']

    
    def test_new_feature_new_label(self):
        label = self.clusterer.add_feature(self.features[0])

        # first label gets 0
        self.assertEqual(label, 0)

    def test_similar_features_same_label(self):
        label1 = self.clusterer.add_feature(self.features[1])
        

        label2 = self.clusterer.add_feature(self.features[2])

        self.assertEqual(label1, label2)
    
    def test_dissimilar_features_different_labels(self):
        label1 = self.clusterer.add_feature(self.features[16890])
        label2 = self.clusterer.add_feature(self.features[0])
        self.assertNotEqual(label1, label2)
    
    def test_get_all_features_and_labels(self):
        features_to_add = [np.random.rand(512) for _ in range(3)]
        for feature in features_to_add:
            self.clusterer.add_feature(feature)
        returned_features, returned_labels = self.clusterer.get_all_features_and_labels()
        np.testing.assert_array_equal(returned_features, np.array(features_to_add))
        self.assertEqual(len(returned_features), len(features_to_add))
        self.assertEqual(len(returned_labels), len(features_to_add))

    def test_ari_score(self):
        predicted_labels = [self.clusterer.add_feature(feature) for feature in self.features[:10000]]

        actual_labels = self.labels[:10000]

        ari = adjusted_rand_score(predicted_labels, actual_labels)

        print(f'ARI: {round(ari, 3)}')
        self.assertGreaterEqual(ari, 0.7)
from sklearn.metrics.pairwise import cosine_distances
import numpy as np


class SimpleDFClust:
    def __init__(self, threshold=0.2):
        self.features = np.empty((0, 512))
        self.labels = np.array([], dtype=int)
        self.next_label = 0
        self.threshold = threshold

    def add_feature(self, new_feature):
        """
        Add a new feature and return its label.
        """
        if self.features.shape[0] == 0:

            self.features = np.vstack([self.features, new_feature])
            self.labels = np.hstack([self.labels, self.next_label])
            self.next_label += 1
            return self.labels[-1]

        distances = cosine_distances([new_feature], self.features).flatten()

        if distances.min() < self.threshold:
            label = self.labels[np.argmin(distances)]
            self.features = np.vstack([self.features, new_feature])
            self.labels = np.hstack([self.labels, label])
            return label
        else:

            self.features = np.vstack([self.features, new_feature])
            self.labels = np.hstack([self.labels, self.next_label])
            self.next_label += 1
            return self.labels[-1]

    def get_all_features_and_labels(self):
        """
        Return all features and their labels.
        """
        return self.features, self.labels

    def get_all_features_and_labels(self):
        """
        Return all features and their labels.
        """
        return self.features, self.labels

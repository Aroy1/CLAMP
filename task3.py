import numpy as np
import pandas as pd
import sklearn.cluster
import yellowbrick.cluster


class KmeansClustering:
    def __init__(self,
                 train_features: pd.DataFrame,
                 test_features: pd.DataFrame,
                 random_state: int
                 ):
        self.train_features = train_features
        self.test_features = test_features
        self.random_state = random_state
        # TODO: Add any state variables you may need to make your functions work
        pass

    def kmeans_train(self) -> list:
        # TODO: train a kmeans model using the training data, determine the optimal value of k (between 1 and 10) with n_init set to 10 and return a list of cluster ids
        # corresponding to the cluster id of each row of the training data
        kmeans = sklearn.cluster.KMeans(random_state=self.random_state)

        visualizer = yellowbrick.cluster.KElbowVisualizer(kmeans, k=(1, 11))
        visualizer.fit(self.train_features)

        optimal_k = visualizer.elbow_value_

        optimal_kmeans = sklearn.cluster.KMeans(n_clusters=optimal_k, random_state=self.random_state)
        optimal_kmeans.fit(self.train_features)

        cluster_ids = optimal_kmeans.predict(self.train_features)
        return cluster_ids.tolist()

    def kmeans_test(self) -> list:
        # TODO: return a list of cluster ids corresponding to the cluster id of each row of the test data

        kmeans = sklearn.cluster.KMeans(random_state=self.random_state)

        visualizer = yellowbrick.cluster.KElbowVisualizer(kmeans, k=(1, 11))
        visualizer.fit(self.train_features)

        optimal_k = visualizer.elbow_value_

        optimal_kmeans = sklearn.cluster.KMeans(n_clusters=optimal_k, random_state=self.random_state)
        optimal_kmeans.fit(self.train_features)

        cluster_ids = optimal_kmeans.predict(self.test_features)
        return cluster_ids.tolist()

    def train_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # TODO: return the training dataset with a new feature called kmeans_cluster_id
        cluster_ids = self.kmeans_train()

        output_df = self.train_features.copy()
        output_df["kmeans_cluster_id"] = cluster_ids

        return output_df

    def test_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # TODO: return the test dataset with a new feature called kmeans_cluster_id
        cluster_ids = self.kmeans_test()

        output_df = self.test_features.copy()
        output_df["kmeans_cluster_id"] = cluster_ids

        return output_df

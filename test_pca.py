import unittest
import pandas as pd
import os
from task2 import PreprocessDataset


def double_height(dataframe: pd.DataFrame):
    return dataframe["height"] * 2


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.read_csv("sample.csv")
        self.target_col = "target"

        # PCA Doesnt play nice with strings so we are just dropping color for this example
        self.ans_train_features = pd.read_csv(os.path.join("task2", "train_feats_tts.csv")).set_index("index").drop(
            columns=["color"])
        self.ans_test_features = pd.read_csv(os.path.join("task2", "test_feats_tts.csv")).set_index("index").drop(
            columns=["color"])
        self.ans_train_targets = pd.read_csv(os.path.join("task2", "train_targets_tts.csv")).set_index("index")[
            "target"]
        self.ans_test_targets = pd.read_csv(os.path.join("task2", "test_targets_tts.csv")).set_index("index")["target"]

        self.preprocessDataset = PreprocessDataset(
            train_features=self.ans_train_features,
            test_features=self.ans_test_features,
            one_hot_encode_cols=["color", "version"],
            min_max_scale_cols=["cost"],
            n_components=2,
            feature_engineering_functions={"double_height": double_height})

        self.train_features_pca = self.preprocessDataset.pca_train()
        self.test_features_pca = self.preprocessDataset.pca_test()

        # self.train_features_pca.to_csv(os.path.join("task2","train_feats_pca.csv"))
        # self.test_features_pca.to_csv(os.path.join("task2","test_feats_pca.csv"))

        self.ans_train_features_pca = pd.read_csv(os.path.join("task2", "train_feats_pca.csv")).set_index("index")
        self.ans_test_features_pca = pd.read_csv(os.path.join("task2", "test_feats_pca.csv")).set_index("index")

    def test_train_features_pca(self):
        print('\n Printing ans_train_features_pca\n')
        print(self.ans_train_features_pca)
        print('\n Printing train_features_pca\n')
        print(self.train_features_pca)
        self.assertTrue(self.train_features_pca.round(4).equals(self.ans_train_features_pca.round(4)))

    def test_test_features_pca(self):
        print('\n Printing ans_test_features_pca\n')
        print(self.ans_test_features_pca)
        print('\n Printing test_features_pca\n')
        print(self.test_features_pca)
        print('\n')
        self.assertTrue(self.test_features_pca.round(4).equals(self.ans_test_features_pca.round(4)))

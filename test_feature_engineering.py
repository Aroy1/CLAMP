import unittest
import pandas as pd
import os
from task2 import PreprocessDataset


def double_height(dataframe: pd.DataFrame):
    return dataframe["height"] * 2


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.read_csv("sample.csv")
        self.target_col = "target"

        self.ans_train_features = pd.read_csv(os.path.join("task2", "train_feats_tts.csv")).set_index("index")
        self.ans_test_features = pd.read_csv(os.path.join("task2", "test_feats_tts.csv")).set_index("index")
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

        self.train_features_fe = self.preprocessDataset.feature_engineering_train()
        self.test_features_fe = self.preprocessDataset.feature_engineering_test()

        self.ans_train_features_fe = pd.read_csv(os.path.join("task2", "train_feats_fe.csv")).set_index("index")
        self.ans_test_features_fe = pd.read_csv(os.path.join("task2", "test_feats_fe.csv")).set_index("index")

    def test_train_features_fe(self):
        # print(self.ans_train_features_fe)
        # print(self.train_features_fe)
        self.assertTrue(self.train_features_fe.equals(self.ans_train_features_fe))

    def test_test_features_fe(self):
        # print(self.ans_test_features_fe)
        # print(self.test_features_fe)
        self.assertTrue(self.test_features_fe.equals(self.ans_test_features_fe))

import unittest
import pandas as pd
import os
from task2 import PreprocessDataset


def double_height(dataframe: pd.DataFrame):
    return dataframe["height"] * 2


class TestPreprocess(unittest.TestCase):
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

        self.train_features_preprocess, self.test_features_preprocess = self.preprocessDataset.preprocess()

        self.ans_train_features_preprocess = pd.read_csv(os.path.join("task2", "train_feats_preprocess.csv")).set_index(
            "index")
        self.ans_test_features_preprocess = pd.read_csv(os.path.join("task2", "test_feats_preprocess.csv")).set_index(
            "index")

    def test_train_features_preprocess(self):
        # print(self.ans_train_features_preprocess)
        # print(self.train_features_preprocess)
        self.assertTrue(self.train_features_preprocess.equals(self.ans_train_features_preprocess))

    def test_test_features_preprocess(self):
        # print(self.ans_test_features_preprocess)
        # print(self.test_features_preprocess)
        self.assertTrue(self.test_features_preprocess.equals(self.ans_test_features_preprocess))

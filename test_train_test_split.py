import unittest
import pandas as pd
import os
from task2 import train_test_split

class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.read_csv("sample.csv")
        self.target_col = "target"
        self.train_features,self.test_features,self.train_targets,self.test_targets = train_test_split(self.dataset,
                            self.target_col, 
                            test_size=.2,
                            stratify=True,
                            random_state=0)
        self.ans_train_features = pd.read_csv(os.path.join("task2","train_feats_tts.csv")).set_index("index")
        self.ans_test_features = pd.read_csv(os.path.join("task2","test_feats_tts.csv")).set_index("index")
        self.ans_train_targets = pd.read_csv(os.path.join("task2","train_targets_tts.csv")).set_index("index")["target"]
        self.ans_test_targets = pd.read_csv(os.path.join("task2","test_targets_tts.csv")).set_index("index")["target"]
        
    def test_train_features(self):
        #print(self.ans_train_features)
        #print(self.train_features)
        self.assertTrue(self.train_features.equals(self.ans_train_features))

    def test_test_features(self):
        #print(self.ans_test_features)
        #print(self.test_features)
        self.assertTrue(self.test_features.equals(self.ans_test_features))

    def test_train_targets(self):
        #print(self.ans_train_targets)
        #print(self.train_targets)
        self.assertTrue(self.train_targets.equals(self.ans_train_targets))

    def test_test_targets(self):
        #print(self.ans_test_targets)
        #print(self.test_targets)
        self.assertTrue(self.test_targets.equals(self.ans_test_targets))
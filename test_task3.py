# CS6035 A4 Task3 Unit Test
# I almost cried.
# r00t@gatech.edu
# Copyright (c) 2023 - Jason Royes
import unittest
import pandas as pd
import numpy as np
import task3
from sklearn.model_selection import train_test_split

def compare_dataset_rows(a:pd.DataFrame, b:pd.DataFrame, idx: np.int64) -> bool:
    columns = list(a.columns.values)
    truth = a.loc[idx][columns] == b.loc[idx][columns]
    return truth.all()

def rnd_idx(a:pd.DataFrame) -> np.int64:
    rnd_row_idx = np.random.choice(a.index)
    return rnd_row_idx

def rnd_rows_equal(a:pd.DataFrame, b:pd.DataFrame, num: int) -> bool:
    results = [ compare_dataset_rows(a, b, rnd_idx(a)) for i in range(0, num) ]
    true_results = list(filter(lambda r: r == True, results))
    return len(true_results) == len(results)

class TestTask3(unittest.TestCase):

    def setUp(self):
        dataset = pd.read_csv("../CLAMP_Train.csv")
        dataset = dataset.drop(columns=['e_res', 'e_res2'])
        rndval = 42
        self.train, self.test = train_test_split(dataset, test_size=.25, random_state=rndval)
        self.kc = task3.KmeansClustering(self.train, self.test, rndval)

    def test_01_kmeans_train(self):
        clusters = self.kc.kmeans_train()
        self.assertEqual(len(clusters), len(self.train))
        cluster_df = pd.DataFrame(data=clusters, columns=['cluster_id'])
        self.assertEqual(len(cluster_df.value_counts()), 2)

    def test_02_kmeans_test(self):
        self.test_01_kmeans_train()
        clusters = self.kc.kmeans_test()
        self.assertEqual(len(clusters), len(self.test))

    def test_03_kmeans_train_add(self):
        self.test_01_kmeans_train()
        new_ds = self.kc.train_add_kmeans_cluster_id_feature()
        self.assertEqual(len(new_ds), len(self.train))
        self.assertEqual(new_ds['kmeans_cluster_id'].count(), len(self.train))
        self.assertEqual(len(new_ds.columns), len(self.train.columns) + 1)
        self.assertTrue(rnd_rows_equal(self.train, new_ds, 100))

    def test_04_kmeans_test_add(self):
        self.test_01_kmeans_train()
        new_ds = self.kc.test_add_kmeans_cluster_id_feature()
        self.assertEqual(len(new_ds), len(self.test))
        self.assertEqual(new_ds['kmeans_cluster_id'].count(), len(self.test))
        self.assertEqual(len(new_ds.columns), len(self.test.columns) + 1)
        self.assertTrue(rnd_rows_equal(self.test, new_ds, 100))
    
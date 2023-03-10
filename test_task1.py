import unittest
import pandas as pd
from task1 import find_dataset_statistics

class TestTask1(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.read_csv("sample.csv")
        self.target_col = "target"
        self.n_records,self.n_columns,self.n_negative,self.n_positive,self.perc_positive = find_dataset_statistics(self.dataset,self.target_col)

    def test_nrecords(self):
        self.assertEqual(self.n_records,10)

    def test_n_columns(self):
        self.assertEqual(self.n_columns,5)

    def test_n_negative(self):
        self.assertEqual(self.n_negative,5)

    def test_n_positive(self):
        self.assertEqual(self.n_positive,5)
    
    def test_perc_positive(self):
        self.assertEqual(self.perc_positive,50)
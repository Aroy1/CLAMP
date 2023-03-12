from tempfile import NamedTemporaryFile
import os
import pandas as pd
from task5 import train_model_return_scores
from sklearn.metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def test_train_model_return_scores():
    # Generate some example data
    df = pd.read_csv('CLAMP_Train.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    test_df = test_df.drop(columns=['class'], axis=1, errors='ignore')

    # Save the data to temporary files
    with NamedTemporaryFile(mode='w', delete=False) as f:
        train_df.to_csv(f, index=False)
        train_df_path = f.name
    with NamedTemporaryFile(mode='w', delete=False) as f:
        test_df.to_csv(f, index=False)
        test_df_path = f.name

    # Run the function
    result = train_model_return_scores(train_df_path, test_df_path)

    # Check the result
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'index', 'malware_score'}
    assert result.shape[0] == test_df.shape[0]

    # Clean up the temporary files
    os.remove(train_df_path)
    os.remove(test_df_path)

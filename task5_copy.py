import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.metrics import *


def train_model_return_scores(train_df_path, test_df_path) -> pd.DataFrame:
    # TODO: Load and preprocess the train and test dfs
    # Train a sklearn model using training data at train_df_path 
    # Use any sklearn model and return the test index and model scores

    # Read Data In
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    # Create a True Y for testing from test_df
    # y_test = test_df['class']

    # Drop class(Target) if present
    # test_df = test_df.drop(columns=['class'], axis=1, errors='ignore')

    # Split Train to x_train and y_train
    x_train = train_df.drop(columns=['class'])
    y_train = train_df['class']

    # Encode all features Ignore unknown
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_train_encoded = encoder.fit_transform(x_train)
    test_encoded = encoder.transform(test_df)
    # Convert NDArray encode to DataFrame
    x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoder.get_feature_names_out(), index=train_df.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(), index=test_df.index)
    # Scale all features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_encoded_df)
    test_scaled = scaler.transform(test_encoded_df)
    # Convert NDArray scaled to DataFrame
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=encoder.get_feature_names_out(), index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=encoder.get_feature_names_out(), index=test_df.index)
    #  Drop any NaN columns
    x_train_nona = x_train_scaled_df.dropna(axis=1)
    test_nona = test_scaled_df.dropna(axis=1)
    # Generate Model
    model = LogisticRegression(penalty='l2', solver='newton-cholesky')
    # Fit Model to Training Data
    model.fit(x_train_nona, y_train)
    # Only needed for Metrics calculation
    # y_pred = model.predict(test_nona)

    # Perform prediction on test data
    y_prob = model.predict_proba(test_nona)[:, 1]

    # TODO: output dataframe should have 2 columns
    # index : this should be the row index of the test df
    # malware_score : this should be your model's output for the row in the test df
    test_scores = pd.DataFrame({'index': test_df.index, 'malware_score': y_prob})
    return test_scores

# Metrics calculation
    # acc = accuracy_score(y_test, y_pred)
    # rec = recall_score(y_test, y_pred)
    # prec = precision_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    # auc = roc_auc_score(y_test, y_prob)

import pandas as pd
import sklearn.decomposition
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.impute


def train_test_split(dataset: pd.DataFrame,
                     target_col: str,
                     test_size: float,
                     stratify: bool,
                     random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # TODO: Write the necessary code to split a dataframe into a Train and Test feature dataframe and a Train and Test 
    # target series

    features = dataset.drop(target_col, axis=1)
    target = dataset[target_col]

    stratify_target = target if stratify else None

    train_features, \
        test_features, \
        train_targets, \
        test_targets = sklearn.model_selection.train_test_split(features,
                                                                target,
                                                                test_size=test_size,
                                                                stratify=stratify_target,
                                                                random_state=random_state)

    return train_features, test_features, train_targets, test_targets


class PreprocessDataset:
    def __init__(self,
                 train_features: pd.DataFrame,
                 test_features: pd.DataFrame,
                 one_hot_encode_cols: list[str],
                 min_max_scale_cols: list[str],
                 n_components: int,
                 feature_engineering_functions: dict
                 ):
        # TODO: Add any state variables you may need to make your functions work
        self.train_features = train_features
        self.test_features = test_features
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions
        return

    def one_hot_encode_columns_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in 
        # the variable one_hot_encode_cols "one hot" encoded
        encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)

        train_encoded = encoder.fit_transform(self.train_features[self.one_hot_encode_cols])

        train_encoded_df = pd.DataFrame(train_encoded,
                                        columns=encoder.get_feature_names_out(self.one_hot_encode_cols),
                                        index=self.train_features.index)

        one_hot_encoded_dataset = pd.concat(
            [train_encoded_df, self.train_features.drop(self.one_hot_encode_cols, axis=1)], axis=1)

        return one_hot_encoded_dataset

    def one_hot_encode_columns_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in 
        # the variable one_hot_encode_cols "one hot" encoded 
        encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        encoder.fit(self.train_features[self.one_hot_encode_cols])
        test_encoded = encoder.transform(self.test_features[self.one_hot_encode_cols])

        test_encoded_df = pd.DataFrame(test_encoded,
                                       columns=encoder.get_feature_names_out(self.one_hot_encode_cols),
                                       index=self.test_features.index)

        one_hot_encoded_dataset = pd.concat(
            [test_encoded_df, self.test_features.drop(self.one_hot_encode_cols, axis=1)], axis=1)

        return one_hot_encoded_dataset

    def min_max_scaled_columns_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in 
        # the variable min_max_scale_cols scaled to the min and max of each column
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaled_features = scaler.fit_transform(self.train_features[self.min_max_scale_cols])
        scaled_features_df = pd.DataFrame(scaled_features, columns=self.min_max_scale_cols,
                                          index=self.train_features.index)

        min_max_scaled_dataset = pd.concat(
            [scaled_features_df, self.train_features.drop(self.min_max_scale_cols, axis=1)], axis=1)
        return min_max_scaled_dataset

    def min_max_scaled_columns_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in 
        # the variable min_max_scale_cols scaled to the min and max of each column
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(self.train_features[self.min_max_scale_cols])
        scaled_features = scaler.transform(self.test_features[self.min_max_scale_cols])
        scaled_features_df = pd.DataFrame(scaled_features, columns=self.min_max_scale_cols,
                                          index=self.test_features.index)

        min_max_scaled_dataset = pd.concat(
            [scaled_features_df, self.test_features.drop(self.min_max_scale_cols, axis=1)], axis=1)
        return min_max_scaled_dataset

    def pca_train(self) -> pd.DataFrame:
        # TODO: use PCA to reduce the train_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n
        train_feature_unnaned = self.train_features.dropna(axis=1)

        pca = sklearn.decomposition.PCA(n_components=self.n_components, random_state=0)
        principle_components = pca.fit_transform(train_feature_unnaned)

        pca_cols = [f"component_{i}" for i in range(1, self.n_components + 1)]

        pca_dataset = pd.DataFrame(data=principle_components, columns=pca_cols,
                                   index=range(train_feature_unnaned.shape[0]))
        return pca_dataset

    def pca_test(self) -> pd.DataFrame:
        # TODO: use PCA to reduce the test_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n 
        test_feature_unnaned = self.test_features.dropna(axis=1)
        train_feature_unnaned = self.train_features.dropna(axis=1)

        pca = sklearn.decomposition.PCA(n_components=self.n_components, random_state=0)
        pca.fit(train_feature_unnaned)
        principle_components = pca.transform(test_feature_unnaned)

        pca_cols = [f"component_{i}" for i in range(1, self.n_components + 1)]

        pca_dataset = pd.DataFrame(data=principle_components, columns=pca_cols,
                                   index=range(test_feature_unnaned.shape[0]))
        return pca_dataset

    def feature_engineering_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied 
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series
        feature_engineered_cols = {}
        for col, func in self.feature_engineering_functions.items():
            feature_engineered_cols[col] = func(self.train_features)

        feature_engineered_dataset = pd.concat([self.train_features,
                                                pd.DataFrame(feature_engineered_cols)], axis=1)
        return feature_engineered_dataset

    def feature_engineering_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied 
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series
        feature_engineered_cols = {}
        for col, func in self.feature_engineering_functions.items():
            feature_engineered_cols[col] = func(self.test_features)

        feature_engineered_dataset = pd.concat([self.test_features,
                                                pd.DataFrame(feature_engineered_cols)], axis=1)
        return feature_engineered_dataset

    def preprocess(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO: Use the functions you wrote above to create train/test splits of the features and target with scaled and encoded values 
        # for the columns specified in the init function
        self.test_features = self.one_hot_encode_columns_test()
        self.test_features = self.min_max_scaled_columns_test()
        self.test_features = self.feature_engineering_test()

        self.train_features = self.one_hot_encode_columns_train()
        self.train_features = self.min_max_scaled_columns_train()
        self.train_features = self.feature_engineering_train()

        train_features = self.train_features
        test_features = self.test_features
        return train_features, test_features

import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.feature_selection import RFE


class ModelMetrics:
    def __init__(self, model_type: str, train_metrics: dict, test_metrics: dict, feature_importance_df: pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"

    def add_train_metric(self, metric_name: str, metric_val: float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self, metric_name: str, metric_val: float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self):
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str


def calculate_naive_metrics(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target_col: str,
                            naive_assumption: int) -> ModelMetrics:
    # TODO: Write the necessary code to calculate accuracy, recall, precision and fscore given a train and test dataframe
    # and a train and test target series and naive assumption

    train_target_count = train_dataset[target_col].value_counts()
    test_target_count = test_dataset[target_col].value_counts()

    train_naive = np.repeat(naive_assumption, len(train_dataset))
    test_naive = np.repeat(naive_assumption, len(test_dataset))

    train_accuracy = accuracy_score(train_dataset[target_col], train_naive)
    test_accuracy = accuracy_score(test_dataset[target_col], test_naive)

    train_recall = recall_score(train_dataset[target_col], train_naive)
    test_recall = recall_score(test_dataset[target_col], test_naive)

    train_precision = precision_score(train_dataset[target_col], train_naive)
    test_precision = precision_score(test_dataset[target_col], test_naive)

    train_fscore = f1_score(train_dataset[target_col], train_naive)
    test_fscore = f1_score(test_dataset[target_col], test_naive)

    train_metrics = {
        "accuracy": round(train_accuracy, 4),
        "recall": round(train_recall, 4),
        "precision": round(train_precision, 4),
        "fscore": round(train_fscore, 4)
    }
    test_metrics = {
        "accuracy": round(test_accuracy, 4),
        "recall": round(test_recall, 4),
        "precision": round(test_precision, 4),
        "fscore": round(test_fscore, 4)
    }
    naive_metrics = ModelMetrics("Naive", train_metrics, test_metrics, None)
    return naive_metrics


def calculate_logistic_regression_metrics(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target_col: str,
                                          logreg_kwargs) -> tuple[ModelMetrics, LogisticRegression]:
    # TODO: Write the necessary code to train a logistic regression binary classifiaction model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test target series 
    # and keyword arguments for the logistic regrssion model
    model = LogisticRegression(**logreg_kwargs)

    x_train = train_dataset.drop(columns=[target_col])
    y_train = train_dataset[target_col]

    x_test = test_dataset.drop(columns=[target_col])
    y_test = test_dataset[target_col]

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    train_recall = recall_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred)
    train_fscore = f1_score(y_train, train_pred)

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, train_pred).ravel()
    train_fpr = fp_train / (fp_train + tn_train)
    train_fnr = fn_train / (fn_train + tp_train)

    train_roc_auc = roc_auc_score(y_train, train_prob)

    test_acc = accuracy_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    test_fscore = f1_score(y_test, test_pred)

    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test_pred).ravel()
    test_fpr = fp_test / (fp_test + tn_test)
    test_fnr = fn_test / (fn_test + tp_test)

    test_roc_auc = roc_auc_score(y_test, test_prob)

    train_metrics = {
        "accuracy": round(train_acc, 4),
        "recall": round(train_recall, 4),
        "precision": round(train_precision, 4),
        "fscore": round(train_fscore, 4),
        "fpr": round(train_fpr, 4),
        "fnr": round(train_fnr, 4),
        "roc_auc": round(train_roc_auc, 4)
    }
    test_metrics = {
        "accuracy": round(test_acc, 4),
        "recall": round(test_recall, 4),
        "precision": round(test_precision, 4),
        "fscore": round(test_fscore, 4),
        "fpr": round(test_fpr, 4),
        "fnr": round(test_fnr, 4),
        "roc_auc": round(test_roc_auc, 4)
    }
    # TODO: Use RFE to select the top 10 features 
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by ascending ranking then decending absolute value of Importance
    rfe = RFE(model, n_features_to_select=10)
    rfe = rfe.fit(x_train, y_train)

    ranking = rfe.ranking_

    importance = abs(rfe.estimator_.coef_[0])

    log_reg_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': importance, 'Ranking': ranking})
    log_reg_importance = log_reg_importance.sort_values(by=["Importance", "Ranking"],
                                                        ascending=[False, True]).reset_index(drop=True)
    log_reg_importance = log_reg_importance.drop('Ranking', axis=1)

    log_reg_metrics = ModelMetrics("Logistic Regression", train_metrics, test_metrics, log_reg_importance)

    return log_reg_metrics, model


def calculate_random_forest_metrics(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target_col: str,
                                    rf_kwargs) -> tuple[ModelMetrics, RandomForestClassifier]:
    # TODO: Write the necessary code to train a random forest binary classification model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test 
    # target series and keyword arguments for the random forest model
    model = RandomForestClassifier(**rf_kwargs)

    x_train = train_dataset.drop(columns=[target_col])
    y_train = train_dataset[target_col]

    x_test = test_dataset.drop(columns=[target_col])
    y_test = test_dataset[target_col]

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    train_recall = recall_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred)
    train_fscore = f1_score(y_train, train_pred)

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, train_pred).ravel()
    train_fpr = fp_train / (fp_train + tn_train)
    train_fnr = fn_train / (fn_train + tp_train)

    train_roc_auc = roc_auc_score(y_train, train_prob)

    test_acc = accuracy_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    test_fscore = f1_score(y_test, test_pred)

    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test_pred).ravel()
    test_fpr = fp_test / (fp_test + tn_test)
    test_fnr = fn_test / (fn_test + tp_test)

    test_roc_auc = roc_auc_score(y_test, test_prob)

    train_metrics = {
        "accuracy": round(train_acc, 4),
        "recall": round(train_recall, 4),
        "precision": round(train_precision, 4),
        "fscore": round(train_fscore, 4),
        "fpr": round(train_fpr, 4),
        "fnr": round(train_fnr, 4),
        "roc_auc": round(train_roc_auc, 4)
    }
    test_metrics = {
        "accuracy": round(test_acc, 4),
        "recall": round(test_recall, 4),
        "precision": round(test_precision, 4),
        "fscore": round(test_fscore, 4),
        "fpr": round(test_fpr, 4),
        "fnr": round(test_fnr, 4),
        "roc_auc": round(test_roc_auc, 4)
    }
    # TODO: Reminder DONT use RFE for rf_importance
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by decending absolute value of Importance
    rf_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': model.feature_importances_})
    rf_importance = rf_importance.sort_values(by=['Importance'], ascending=False).reset_index(drop=True)
    rf_metrics = ModelMetrics("Random Forest", train_metrics, test_metrics, rf_importance)

    return rf_metrics, model


def calculate_gradient_boosting_metrics(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target_col: str,
                                        gb_kwargs) -> tuple[ModelMetrics, GradientBoostingClassifier]:
    # TODO: Write the necessary code to train a gradient boosting binary classification model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test 
    # target series and keyword arguments for the gradient boosting model
    model = GradientBoostingClassifier(**gb_kwargs)

    x_train = train_dataset.drop(columns=[target_col])
    y_train = train_dataset[target_col]

    x_test = test_dataset.drop(columns=[target_col])
    y_test = test_dataset[target_col]

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    train_recall = recall_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred)
    train_fscore = f1_score(y_train, train_pred)

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, train_pred).ravel()
    train_fpr = fp_train / (fp_train + tn_train)
    train_fnr = fn_train / (fn_train + tp_train)

    train_roc_auc = roc_auc_score(y_train, train_prob)

    test_acc = accuracy_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    test_fscore = f1_score(y_test, test_pred)

    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test_pred).ravel()
    test_fpr = fp_test / (fp_test + tn_test)
    test_fnr = fn_test / (fn_test + tp_test)

    test_roc_auc = roc_auc_score(y_test, test_prob)

    train_metrics = {
        "accuracy": round(train_acc, 4),
        "recall": round(train_recall, 4),
        "precision": round(train_precision, 4),
        "fscore": round(train_fscore, 4),
        "fpr": round(train_fpr, 4),
        "fnr": round(train_fnr, 4),
        "roc_auc": round(train_roc_auc, 4)
    }
    test_metrics = {
        "accuracy": round(test_acc, 4),
        "recall": round(test_recall, 4),
        "precision": round(test_precision, 4),
        "fscore": round(test_fscore, 4),
        "fpr": round(test_fpr, 4),
        "fnr": round(test_fnr, 4),
        "roc_auc": round(test_roc_auc, 4)
    }
    # TODO: Reminder DONT use RFE for gb_importance
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by decending absolute value of Importance
    gb_importance = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    gb_metrics = ModelMetrics("Gradient Boosting", train_metrics, test_metrics, gb_importance)

    return gb_metrics, model

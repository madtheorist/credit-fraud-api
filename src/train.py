# run this script to generate rf_model_final.pkl and preprocessor.pkl file in the /models directory
# this trains a random forest model on the full fraudTrain.csv dataset
# there is a separate test set available on Kaggle

import os
import joblib
import pandas as pd
import time
from typing import Tuple
from pipeline import Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from imblearn.over_sampling import SMOTE
from utils import create_timestamp_columns



def preprocess(df: pd.DataFrame) -> Tuple[csr_matrix, pd.Series]:
    """
    Preprocesses the raw data by transforming features and extracting the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input data containing features and target labels.

    Returns
    -------
    X : scipy.sparse.csr_matrix
        The preprocessed feature matrix with transformed features, represented as a sparse matrix.
    y : pd.Series
        The target labels indicating fraud (1) or non-fraud (0).
    """
    start = time.time()
    print(f'Preprocessing training data with {len(df)} samples')

    # generate engineered timestamp columns
    df = create_timestamp_columns(df)
    X = df[Preprocessor.all_features]
    y = df[Preprocessor.target]

    # feature scaling and imputation
    preprocessor = Preprocessor(X)
    X = preprocessor.fit_transform()
    preprocessor_path = os.path.join("models", "preprocessor.pkl")
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved in {preprocessor_path}")

    # Oversample minority class using SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X, y = smote.fit_resample(X, y)

    print(f'Finished preprocessing data in {time.time() - start:.2f} seconds')
    return X, y


if __name__ == "__main__":

    filepath = os.path.join("data", "fraudTrain.csv")
    print(f"Importing raw data from {filepath}")
    df = pd.read_csv(filepath)
    X, y = preprocess(df)

    start = time.time()
    n_estimators, max_depth = 50, 5
    print(f"Training random forest classifier with {n_estimators=} and {max_depth=}")
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
    clf.fit(X, y)
    print(f'Finished training classifier in {time.time() - start:.2f} seconds')

    filename = "rf_model_final.pkl"
    joblib.dump(clf, f"models/{filename}")
    print(f"Model saved in models/{filename}")

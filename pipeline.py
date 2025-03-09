import numpy as np
import pandas as pd
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import csr_matrix


class Preprocessor:
    """
    Pipeline object to handle preprocessing tasks such as
    - Engineering of new timestamp features
    - Imputing missing categorical data and one-hot encoding
    - Scaling of numerical data
    """

    numerical_features: List[str] = ["amt", "hour", "time_since_last_minutes"]
    categorical_features: List[str] = ["category"]
    all_features = numerical_features + categorical_features
    target: str = "is_fraud"

    def __init__(
        self,
        X_train: pd.DataFrame,
    ):
        self.X = X_train[self.all_features]
        self._numerical_transformer = None
        self._categorical_transformer = None
        self._final_transformer = None

    @property
    def numerical_transformer(self) -> Pipeline:
        if not self._numerical_transformer:
            numerical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # Fill missing values with median
                    (
                        "scaler",
                        StandardScaler(),
                    ),  # Standardize features to zero mean and unit variance
                ]
            )
            self._numerical_transformer = numerical_transformer
        return self._numerical_transformer

    @property
    def categorical_transformer(self) -> Pipeline:
        if not self._categorical_transformer:
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Fill missing with mode
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),  # One-hot encode categorical features
                ]
            )
            self._categorical_transformer = categorical_transformer
        return self._categorical_transformer

    @property
    def final_transformer(self) -> ColumnTransformer:
        if not self._final_transformer:
            final_transformer = ColumnTransformer(
                transformers=[
                    ("num", self.numerical_transformer, self.numerical_features),
                    ("cat", self.categorical_transformer, self.categorical_features),
                ]
            )
            self._final_transformer = final_transformer
        return self._final_transformer

    def fit_transform(self) -> csr_matrix:
        """
        Apply the preprocessing steps to the numerical and categorical columns
        Returns scipy sparse matrix
        This is a more efficient way to store the data as there are a lot of 0s in the one-hot encoded columns
        """
        return self.final_transformer.fit_transform(self.X)

    def transform(self, X_test: np.ndarray) -> csr_matrix:
        """
        Apply the fitted transformer to the test set.
        This must be called after fit_transform has been called on the train set.
        """
        if not self._final_transformer:
            raise Exception("The preprocessor has not yet been fit to training data.")
        return self.final_transformer.transform(X_test)

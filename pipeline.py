import numpy as np
import pandas as pd
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils import create_timestamp_columns
from scipy.sparse import csr_matrix

class Preprocessor:
    """
    Pipeline object to handle preprocessing tasks such as
    - Engineering of new timestamp features
    - Imputing missing categorical data and one-hot encoding
    - Scaling of numerical data
    """

    def __init__(
            self, 
            data: pd.DataFrame,
            numerical_features: List[str] = ['amt', 'hour', 'time_since_last_minutes'],
            categorical_features: List[str] = ['category'],
            target: str = 'is_fraud'
    ):
        """
        target (str): the column indicating the fraud label
        add the engineered timestamp columns to 
        """
        self.data = create_timestamp_columns(data)
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.X = self.data[numerical_features + categorical_features]
        self.y = self.data[target]
        self._numerical_transformer = None
        self._categorical_transformer = None

    @property
    def numerical_transformer(self) -> Pipeline:
        if not self._numerical_transformer:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
                ('scaler', StandardScaler())  # Standardize features to zero mean and unit variance
            ])
            self._numerical_transformer = numerical_transformer
        return self._numerical_transformer
    
    @property
    def categorical_transformer(self) -> Pipeline:
        if not self._categorical_transformer:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with mode
                ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
            ])
            self._categorical_transformer = categorical_transformer
        return self._categorical_transformer

    def fit(self) -> csr_matrix:
        """
        Apply the preprocessing steps to the numerical and categorical columns
        Returns scipy sparse matrix
        This is a more efficient way to store the data as there are a lot of 0s in the one-hot encoded columns
        """
        preprocessor = ColumnTransformer(transformers=[
            ('num', self.numerical_transformer, self.numerical_features),
            ('cat', self.categorical_transformer, self.categorical_features)
        ])
        self.X_trans = preprocessor.fit_transform(self.X)
        return self.X_trans

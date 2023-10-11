# src/features/scaling_encoding.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder

def scale_and_encode(X_train, X_test, y_train, y_test):
    # Encode categorical columns separately for training and testing sets
    X_train_encoded = pd.get_dummies(X_train, columns=['Gender', 'Source of Traffic'], drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=['Gender', 'Source of Traffic'], drop_first=True)

    # Initialize KNN imputer
    imputer = KNNImputer(n_neighbors=5)

    # Impute missing values in training data
    X_train_imputed = imputer.fit_transform(X_train_encoded)
    X_test_imputed = imputer.transform(X_test_encoded)

    # Round the imputed values to integers 
    X_train_imputed = np.round(X_train_imputed)
    X_test_imputed = np.round(X_test_imputed)

    # Convert imputed data back to DataFrame
    X_train = pd.DataFrame(X_train_imputed, columns=X_train_encoded.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test_encoded.columns)

    # Removing post trip columns since they will not be available to our predictor model in real use case
    columns_to_remove = ['Cruise Distance', 'WiFi', 'Dining', 'Entertainment']
    X_train = X_train.drop(columns_to_remove, axis=1)
    X_test = X_test.drop(columns_to_remove, axis=1)

    # Instantiate the scaler
    scaler = RobustScaler()

    # Fit the scaler on your training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

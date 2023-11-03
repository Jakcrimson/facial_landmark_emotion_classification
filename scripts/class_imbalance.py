import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def resample_data(X, y, method='smote', sampling_strategy='auto', random_state=None):
    """
    Resample the dataset using either oversampling (SMOTE) or undersampling.

    Parameters:
    - X: Features
    - y: Target labels
    - method: 'smote' for oversampling (default) or 'undersample' for random undersampling
    - sampling_strategy: The sampling strategy for resampling. 'auto' will adjust to balance the classes.
    - random_state: Random state for reproducibility.

    Returns:
    - X_resampled: Resampled features
    - y_resampled: Resampled labels
    """
    if method == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'random_oversample':
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        raise ValueError("Invalid resampling method. Use 'smote' for oversampling or 'undersample' for undersampling.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled


def stratified_sampling(X, y, test_size=0.3):
    """This ensures that the class distribution in the training and testing sets matches the original dataset's distribution.
    This induces that that the initial dataset be balanced first.

    Args:
        X : Features
        y : Target labels
        test_size (float, optional): pct of data that should be the testing data. Defaults to 0.3.

    Returns:
        array-like : X_train, X_test, y_train, y_test : the two datasets (X_.. , y_..) used to train and test the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

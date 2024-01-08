import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def resample_data(
        X: list|np.ndarray, 
        y: list|np.ndarray, 
        method: str='smote', 
        sampling_strategy: str='auto', 
        random_state=None):
    """
Resample the dataset using either oversampling (SMOTE) or undersampling.

Parameters
----------
X: array_like
    Features
y: array_like
    Target labels
method: str
    'smote' for oversampling (default) or 'undersample' for random undersampling
sampling_strategy: str
    The sampling strategy for resampling. 'auto' will adjust to balance the classes.
random_state: any
    Random state for reproducibility.

Returns
-------
out: (array_like, array_like)
    X_resampled: resampled features
    y_resampled: resampled labels
    """
    match method:
        case "smote":
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        case "undersample":
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        case'random_oversample':
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        case _:
            raise ValueError("Invalid resampling method. Use 'smote' for oversampling or 'undersample' for undersampling.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled


def stratified_sampling(
        X: list|np.ndarray, 
        y: list|np.ndarray, 
        test_size: float=0.3, 
        random_state: int=42
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
This ensures that the class distribution in the training and testing sets matches the original dataset's distribution.
This induces that that the initial dataset be balanced first.

Parameters
----------
X: array_like
    Features
y: array_like
    Target labels
test_size: float
    (optional) pct of data that should be the testing data. Defaults to 0.3.
random_state: int
    (optional) Random state of the split (default = 42)
    
Returns
-------
out: (array_like, array_like, array_like, array_like) 
    X_train, X_test, y_train, y_test : the two datasets (X_.. , y_..) used to train and test the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test

"""Utility functions for generating and preparing datasets."""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification, make_regression, make_blobs, load_iris, load_wine
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_classification_data(n_samples=500, n_features=10, n_classes=3, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=n_classes,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def get_regression_data(n_samples=500, n_features=8, noise=20.0, random_state=42):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        noise=noise,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def get_clustering_data(n_samples=400, n_clusters=4, random_state=42):
    X, labels = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=1.2,
        random_state=random_state,
    )
    return X, labels


def get_iris_data(random_state=42):
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=0.2, random_state=random_state), iris.feature_names, iris.target_names


def get_wine_data(random_state=42):
    wine = load_wine()
    X, y = wine.data, wine.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=random_state), wine.feature_names, wine.target_names


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

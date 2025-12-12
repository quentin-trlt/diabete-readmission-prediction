"""
XGBoost Classifier with class weights for imbalanced data

This module implements an XGBoost model for predicting
30-day hospital readmission in diabetic patients.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


class XGBoostModel:
    """
    XGBoost Classifier with balanced class weights.

    This model provides gradient boosting for the diabetes readmission
    prediction task with handling of class imbalance.

    Attributes:
        model: xgboost.XGBClassifier instance
        is_fitted: bool indicating whether the model has been trained
    """

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Initialize the XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for boosting
            random_state: Random seed for reproducibility
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        self.is_fitted = False

    def train(self, X_train, y_train):
        """
        Train the XGBoost model.

        Args:
            X_train: Training features (numpy array or pandas DataFrame)
            y_train: Training labels (numpy array or pandas Series)

        Returns:
            self: Returns the instance for method chaining
        """
        # Calculate scale_pos_weight for class imbalance
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model.set_params(scale_pos_weight=scale_pos_weight)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make binary predictions.

        Args:
            X: Features to predict (numpy array or pandas DataFrame)

        Returns:
            numpy array of binary predictions (0 or 1)

        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features to predict (numpy array or pandas DataFrame)

        Returns:
            numpy array of shape (n_samples, 2) with probabilities for each class

        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features
            y_test: True test labels

        Returns:
            dict: Dictionary containing evaluation metrics
                - accuracy: Overall accuracy
                - precision: Precision for positive class
                - recall: Recall for positive class
                - f1_score: F1 score for positive class
                - roc_auc: ROC-AUC score

        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        return metrics

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on gain.

        Args:
            feature_names: List of feature names (optional)

        Returns:
            numpy array of importance scores or dict if feature_names provided

        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before extracting feature importance")

        importance = self.model.feature_importances_

        if feature_names is not None:
            return dict(zip(feature_names, importance))

        return importance

    def get_params(self):
        """
        Get model parameters.

        Returns:
            dict: Model parameters
        """
        return self.model.get_params()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: colic_classifier.py
Author: Claudia Leins
Description: Flexible classifier for equine colic risk factors
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


class ColicClassifier:
    def __init__(self, model_type='catboost', **model_params):
        """
        Flexible classifier for equine colic analysis
        
        Args:
            model_type: 'catboost', 'xgboost', or 'random_forest'
            **model_params: Additional parameters for the specific model
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, model_params)
        self.cat_features = ['horse_keeping', 'breed', 'horse_gender', 'age_group', 'feed_main']
        self.is_trained = False
        self.label_encoder = None
        self.feature_columns = None

    def _create_model(self, model_type, params):
        """Create the specified model with given parameters"""
        if model_type == 'catboost':
            from catboost import CatBoostClassifier
            default_params = {
                'iterations': 1000,
                'depth': 4,
                'l2_leaf_reg': 5,
                'learning_rate': 0.05,
                'early_stopping_rounds': 30,
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'random_seed': 42,
                'verbose': 100,
                'auto_class_weights': 'Balanced',
                'random_strength': 1,
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.8,
                'colsample_bylevel': 0.8,
                'min_data_in_leaf': 10,
            }
            default_params.update(params)
            return CatBoostClassifier(**default_params)

        elif model_type == 'xgboost':
            from xgboost import XGBClassifier
            default_params = {
                'n_estimators': 500,
                'max_depth': 4,
                'learning_rate': 0.05,
                'random_state': 42,
                'eval_metric': 'mlogloss',
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5
            }
            default_params.update(params)
            return XGBClassifier(**default_params)

        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            default_params = {
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'class_weight': 'balanced'
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _prepare_features(self, df):
        """Prepare features for training"""
        columns_to_keep = [
            'colic_keywords',
            'breed',
            'horse_gender',
            'age_group',
            'horse_keeping',
            'temp_max',
            'temp_min',
            'precipitation'
        ]
        df_cleaned = df[columns_to_keep].copy()

        # Neue Spalte feed_main
        df_cleaned["feed_main"] = df["feed_keywords"].str.split(",").str[0].str.strip()

        # Zielvariable vorbereiten
        df_cleaned["colic_main"] = df["colic_keywords"].str.split(",").str[0].str.strip()

        X = df_cleaned.drop(columns=["colic_keywords", "colic_main"])
        y = df_cleaned["colic_main"]

        # Sicherstellen, dass age_group kategorisch bleibt
        X['age_group'] = X['age_group'].astype('category')

        return X, y

    def _prepare_data_for_training(self, X, y):
        """Prepare data based on model type"""
        if self.model_type == 'catboost':
            return X, y, self.cat_features

        elif self.model_type in ['xgboost', 'random_forest']:
            X_encoded = pd.get_dummies(X, columns=self.cat_features)
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.feature_columns = X_encoded.columns.tolist()
            return X_encoded, y_encoded, None
        else:
            return X, y, None

    def train(self, df):
        """Train the model with the given data"""
        X, y = self._prepare_features(df)
        X_prepared, y_prepared, cat_features = self._prepare_data_for_training(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y_prepared, test_size=0.25, stratify=y_prepared, random_state=42
        )

        print(f"=== Training {self.model_type.upper()} Model ===")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        if self.model_type == 'catboost':
            self.model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                cat_features=cat_features,
                use_best_model=True,
                verbose=100
            )
            print(f"Beste Iteration: {self.model.get_best_iteration()}")

        elif self.model_type == 'xgboost':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=100
            )

        else:  # Random Forest
            self.model.fit(X_train, y_train)

        y_pred = self.predict(X_test)

        print("Vorhergesagte Klassen:")
        print(pd.Series(y_pred).value_counts())

        if self.label_encoder is not None:
            y_test_decoded = self.label_encoder.inverse_transform(y_test)
        else:
            y_test_decoded = y_test

        print(classification_report(y_test_decoded, y_pred, zero_division=0))

        self.is_trained = True
        return self

    def predict(self, X):
        """Make predictions"""
        if self.model_type in ['xgboost', 'random_forest'] and self.feature_columns:
            existing_cat_features = [col for col in self.cat_features if col in X.columns]
            X_encoded = pd.get_dummies(X, columns=existing_cat_features)
            for col in self.feature_columns:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_encoded = X_encoded[self.feature_columns]
            predictions = self.model.predict(X_encoded)
            if self.label_encoder is not None:
                predictions = self.label_encoder.inverse_transform(predictions)
        else:
            predictions = self.model.predict(X)

        return np.array(predictions).ravel()

    def feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if self.model_type == 'catboost':
            return self.model.get_feature_importance(prettified=True)

        elif self.model_type == 'xgboost':
            importance = self.model.feature_importances_
            if self.feature_columns:
                return pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
            else:
                return pd.DataFrame({
                    'Feature': [f'feature_{i}' for i in range(len(importance))],
                    'Importance': importance
                }).sort_values('Importance', ascending=False)

        elif self.model_type == 'random_forest':
            importance = self.model.feature_importances_
            if self.feature_columns:
                return pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
            else:
                return pd.DataFrame({
                    'Feature': [f'feature_{i}' for i in range(len(importance))],
                    'Importance': importance
                }).sort_values('Importance', ascending=False)

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
from sklearn.preprocessing import LabelEncoder, StandardScaler

class ColicClassifier:
    def __init__(self, model_type='catboost', include_num_features=True, **model_params):
        self.model_type = model_type
        self.include_num_features = include_num_features
        self.model = self._create_model(model_type, model_params)
        self.cat_features = ['horse_keeping', 'breed', 'horse_gender', 'age_group', 'feed_main']
        self.num_features = ['temp_max', 'temp_min', 'precipitation'] if include_num_features else []
        self.is_trained = False
        self.label_encoder = None
        self.feature_columns = None
        self.scaler = None

    def _create_model(self, model_type, params):
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
        df = df.copy()
        df['feed_main'] = df.get('feed_keywords', pd.Series(['unknown']*len(df))).str.split(',').str[0].str.strip()
        df['age_group'] = df['age_group'].astype('category')
        
        # Nur bestimmte Kolikarten behalten für bessere accuracy zum Testen der synthetischen Daten
        relevant_types = ['displacement', 'sand', 'impaction']
        df = df[df['colic_keywords'].str.split(',').str[0].isin(relevant_types)].copy()
        
        y = df['colic_keywords'].str.split(',').str[0].astype('category')

        X_cat = pd.get_dummies(df[self.cat_features], drop_first=False)

        if self.include_num_features:
            X_num = df[self.num_features].fillna(0)
            if self.model_type != 'catboost':
                # StandardScaler für NN/XG/RF
                self.scaler = StandardScaler()
                X_num_scaled = self.scaler.fit_transform(X_num)
            else:
                X_num_scaled = X_num.values
            X = np.hstack([X_num_scaled, X_cat.values])
        else:
            X = X_cat.values

        return X, y, X_cat.columns.tolist() + (self.num_features if self.include_num_features else [])

    def _prepare_data_for_training(self, X, y):
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        return X, y_encoded

    def train(self, df, test_size=0.25):
        X, y, feature_cols = self._prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, self._prepare_data_for_training(X, y)[1],
            test_size=test_size, stratify=self._prepare_data_for_training(X, y)[1],
            random_state=42
        )

        self.feature_columns = feature_cols

        print(f"=== Training {self.model_type.upper()} Model ===")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        if self.model_type == 'catboost':
            self.model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                cat_features=[i for i, col in enumerate(feature_cols) if col in self.cat_features],
                use_best_model=True,
                verbose=100
            )
            print(f"Beste Iteration: {self.model.get_best_iteration()}")
        else:
            self.model.fit(X_train, y_train)

        # --- Vorhersagen ---
        y_pred = self.model.predict(X_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        y_test_decoded = self.label_encoder.inverse_transform(y_test)

        print("Vorhergesagte Klassen:")
        print(pd.Series(y_pred_decoded).value_counts())
        print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

        self.is_trained = True
        return self

    def predict(self, df):
        X, _, feature_cols = self._prepare_features(df)
        if self.model_type != 'catboost' and self.scaler:
            # Nur für NN/XG/RF
            X[:, :len(self.num_features)] = self.scaler.transform(df[self.num_features].fillna(0))
        preds = self.model.predict(X)
        if self.label_encoder:
            preds = self.label_encoder.inverse_transform(preds)
        return preds

    def feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if self.model_type == 'catboost':
            return self.model.get_feature_importance(prettified=True)
        else:
            importance = self.model.feature_importances_
            return pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: colic_classifier.py
Author: Claudia Leins
Description: CatBoostClassifier
"""

# models/colic_classifier.py
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class ColicClassifier:
    def __init__(self):
        # CatBoost-Modell mit optimierten Parametern gegen Overfitting
        self.model = CatBoostClassifier(
            iterations=1000,  # Mehr Iterationen für Early Stopping
            depth=4,          # Leicht erhöhte Tiefe für mehr Komplexität
            l2_leaf_reg=5,    # Angepasste Regularization :cite[2]
            learning_rate=0.05,  # Höhere Lernrate für besseres Lernen
            early_stopping_rounds=30,  # Mehr Geduld für Early Stopping :cite[6]
            loss_function="MultiClass",
            eval_metric="MultiClass",  # Besser für Multiklassen-Probleme
            random_seed=42,
            verbose=100,
            # NEUE Parameter gegen Overfitting:
            auto_class_weights='Balanced',  # Automatischer Klassenausgleich :cite[9]
            random_strength=1,              # Zufälligkeit für Split-Bewertung :cite[9]
            bootstrap_type='Bernoulli',  # Changed to a type that supports 'subsample'
            subsample=0.8,                  # Sampling für Diversität :cite[9]
            colsample_bylevel=0.8,          # Feature-Sampling :cite[2]
            min_data_in_leaf=10,            # Mindestanzahl pro Blatt :cite[2]
        )

        # Definieren, welche Features kategorial sind
        self.cat_features = ['horse_keeping',
                             'breed',
                             'horse_gender',
                             'age_group',
                             'feed_main'
                             ]

    def train(self, df):
        # df vorbereiten, nur relevante Spalten untersuchen
        columns_to_keep = ['colic_keywords',
                           'breed',
                           'horse_gender',
                           'age_group',
                           'horse_keeping',
                           'temp_max',
                           'temp_min',
                           'precipitation'
                           ]
        df_cleaned = df[columns_to_keep].copy()
        
        # df_cleaned["age_group"] = df["age_group"]
        df_cleaned["feed_main"] = df["feed_keywords"].str.split(",").str[0].str.strip()

        
        # Zielvariable vorbereiten
        df_cleaned["colic_keywords"] = df["colic_keywords"].str.split(", ")
        df_cleaned["colic_main"] = df_cleaned["colic_keywords"].str[0]

        X = df_cleaned.drop(columns=["colic_keywords", "colic_main"])
        y = df_cleaned["colic_main"]

        # Sicherstellen dass age_group kategorisch bleibt
        X['age_group'] = X['age_group'].astype('category')
         
        # Train-Test-Split mit Stratification für Klassenbalance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42  # Mehr Testdaten
        )
        
        # Training mit use_best_model für Early Stopping
        self.model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            cat_features=self.cat_features,
            use_best_model=True,  # Wichtig: Verwendet beste Iteration :cite[6]
            verbose=100
        )
        
        # Beste Iteration analysieren
        print(f"Beste Iteration: {self.model.get_best_iteration()}")

        # Ergebnisse mit korrekter y_pred Verarbeitung
        y_pred = self.model.predict(X_test)
        y_pred_flat = y_pred.flatten()  # 2D zu 1D konvertieren
        
        print("Vorhergesagte Klassen:")
        print(pd.Series(y_pred_flat).value_counts())
        print(classification_report(y_test, y_pred_flat, zero_division=0))

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.get_feature_importance(prettified=True)


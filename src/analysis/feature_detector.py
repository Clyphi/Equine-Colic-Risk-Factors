#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: feature_detector
Author: Claudia Leins
Description: Feature detector for colic analysis.

This module contains the class `ColicFeatureDetector`, which uses
Chi² statistics and logistic regression to check whether text-based features
are significantly related to colic-relevant labels.

Functions:
- Convert texts into bag-of-words
- Extract significant features using the Chi² test
- Train logistic regression
- Make predictions for new texts

Feature Detector für Kolik-Analysen.

Dieses Modul enthält die Klasse `ColicFeatureDetector`, die mithilfe von
Chi²-Statistik und logistischer Regression prüft, ob textbasierte Features
signifikant mit Kolik-relevanten Labels zusammenhängen.

Funktionen:
- Texte in Bag-of-Words umwandeln
- Signifikante Features per Chi²-Test extrahieren
- Logistische Regression trainieren
- Vorhersagen für neue Texte treffen

"""
# src/analysis/feature_detector.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression

class ColicFeatureDetector:
    def __init__(self, n_features=20):
        self.vectorizer = CountVectorizer(max_features=n_features, stop_words="english")
        self.model = LogisticRegression(max_iter=1000)
        self.feature_names = None
        self.scores = None

    def fit(self, texts, labels):
        """
        texts : Liste von Strings (Reddit-Posts)
        labels: Binär (1 = Kolik-relevant, 0 = nicht)
        """
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        chi2_scores, _ = chi2(X, labels)
        self.scores = dict(zip(self.feature_names, chi2_scores))
        self.model.fit(X, labels)
        return self

    def get_top_features(self, n=10):
        """
        Gibt die Top-n Features nach Chi²-Scores zurück
        """
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def predict(self, texts):
        """
        Vorhersage, ob Text Kolik-relevant ist
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

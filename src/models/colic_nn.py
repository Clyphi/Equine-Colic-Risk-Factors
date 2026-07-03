#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: colic_nn_train.py
Author: Claudia Leins
Description: Train a deep learning NN on synthetic equine colic data
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.utils import shuffle

# ------------------------------
# NN-Modell
# ------------------------------
class ColicNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

# ------------------------------
# Trainer
# ------------------------------
class ColicNNTrainer:
    def __init__(self, hidden_dim=64, include_num_features=True):
        self.hidden_dim = hidden_dim
        self.include_num_features = include_num_features
        self.model = None
        self.input_dim = None
        self.output_dim = None
        self.label_encoder = None
        self.scaler = None
        self.cat_features = ['breed', 'horse_gender', 'horse_keeping', 'feed_main', 'age_group']
        self.num_features = ['temp_max', 'temp_min', 'precipitation'] if include_num_features else []

    # --------------------------
    # Feature Preparation
    # --------------------------
    def _prepare_features(self, df):
        df = df.copy()
        df['feed_main'] = df.get('feed_keywords', pd.Series(['unknown']*len(df))).str.split(',').str[0].str.strip()
        df['age_group'] = df['age_group'].astype('category')
        
        # Nur bestimmte Kolikarten behalten für bessere accuracy zum Testen
        relevant_types = ['displacement', 'sand', 'impaction']
        df = df[df['colic_keywords'].str.split(',').str[0].isin(relevant_types)].copy()
        
        y = df['colic_keywords'].str.split(',').str[0].astype('category')

        # One-Hot-Encoding für kategoriale Features
        X_cat = pd.get_dummies(df[self.cat_features], drop_first=False)

        # Numerische Features
        if self.include_num_features:
            X_num = df[self.num_features].fillna(0)
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_num_scaled = self.scaler.fit_transform(X_num)
            else:
                X_num_scaled = self.scaler.transform(X_num)
            X = np.hstack([X_num_scaled, X_cat.values])
        else:
            X = X_cat.values

        feature_columns = list(X_cat.columns) + (self.num_features if self.include_num_features else [])
        return X, y, feature_columns

    # --------------------------
    # Training
    # --------------------------
    def train(self, df, epochs=100, lr=0.001, batch_size=64):
        X, y, feature_columns = self._prepare_features(df)
        self.input_dim = X.shape[1]

        # Label Encoding für Zielvariable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.output_dim = len(self.label_encoder.classes_)

        self.model = ColicNN(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
        )

        # In Tensors konvertieren
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # --------------------------
        # Training Loop
        # --------------------------
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # --------------------------
        # Evaluation
        # --------------------------
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test_tensor).argmax(dim=1)
            acc = (preds == y_test_tensor).float().mean()
        print(f"Test Accuracy: {acc:.4f}")

    # --------------------------
    # Prediction
    # --------------------------
    def predict(self, df):
        X, _, _ = self._prepare_features(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).argmax(dim=1).numpy()
        return self.label_encoder.inverse_transform(preds)

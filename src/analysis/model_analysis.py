#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: model_analysis
Author: Claudia Leins
Description: Workflow for feature detection and visualisation.

This script shows how the `ColicFeatureDetector` and the 
`FeatureVisualizer` class can be used together to
identify and graphically display colic-related text features.

Procedure:
1. Load data (cleaned Reddit posts)
2. Feature detection with Chi² and LogReg
3. Extract top N features
4. Save or display results as a plot

Workflow für Feature-Erkennung und Visualisierung.

Dieses Skript zeigt, wie der `ColicFeatureDetector` und die 
`FeatureVisualizer`-Klasse zusammen genutzt werden können, um
Kolik-relevante Textfeatures zu identifizieren und grafisch darzustellen.

Ablauf:
1. Daten laden (bereinigte Reddit-Posts)
2. Feature-Detection mit Chi² und LogReg
3. Top-N-Features extrahieren
4. Ergebnisse als Plot speichern oder anzeigen

"""
import pandas as pd
from pathlib import Path
import sys
import os

# Projektpfade konfigurieren
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_detector import ColicFeatureDetector
from src.visualization.plot_features import FeatureVisualizer

# Pfade
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "analyzed_posts.csv"
    
# Daten laden
df = pd.read_csv(CSV_PATH)

# Texte und Labels
texts = df["text"].astype(str)
labels = df["is_colic"].astype(int)

# Analyse
detector = ColicFeatureDetector(n_features=500)
detector.fit(texts, labels)
top_features = detector.get_top_features(15)

# Visualisierung
FeatureVisualizer.plot_feature_scores(top_features, top_n=15, output_path="outputs/plots/top_features.png")

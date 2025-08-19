#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: 

File: plot_features
Author: Claudia Leins
Description: Visualisation of colic features.

This module contains the class `FeatureVisualizer`, which graphically displays the results from
the feature detector. The most important features (e.g.
according to Chi² scores) are plotted in a bar chart.

Functions:
- Display top N features in a horizontal bar chart
- Enable saving of the graphic in /outputs/plots

Visualisierung von Kolik-Features.

Dieses Modul enthält die Klasse `FeatureVisualizer`, die die Ergebnisse aus
dem Feature Detector grafisch darstellt. Die wichtigsten Features (z. B.
nach Chi²-Scores) werden in einem Balkendiagramm geplottet.

Funktionen:
- Top-N-Features in horizontalem Balkendiagramm anzeigen
- Speicherung der Grafik in /outputs/plots ermöglichen

"""
# src/visualization/plot_features.py
import matplotlib.pyplot as plt

class FeatureVisualizer:
    @staticmethod
    def plot_feature_scores(feature_scores, top_n=10, output_path=None):
        """
        feature_scores: Liste von (feature, score)
        """
        top_features = feature_scores[:top_n]
        features, scores = zip(*top_features)

        plt.figure(figsize=(10, 6))
        plt.barh(features, scores, color="skyblue")
        plt.xlabel("Chi²-Score")
        plt.title("Top Features für Kolik-Relevanz")
        plt.gca().invert_yaxis()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            print("Plot wurde in Datei gesichert.")
        else:
            plt.show()

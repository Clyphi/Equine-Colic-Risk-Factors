#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: model_analysis
Author: Claudia Leins
Description: Sentiment analysis
"""

import pandas as pd
from bag_of_words import bag_of_words  # Import der Bag-of-Words-Funktion

def sentiment_analyse(filepath, model, vokabular):
    """
    Führt eine Sentiment-Analyse auf Textdaten aus einer CSV-Datei durch.

    Parameter:
    - filepath (str): Pfad zur CSV-Datei mit Textdaten
    - model: Vortrainiertes Modell mit einer predict()-Methode
    - vokabular: Vokabular für die Bag-of-Words-Repräsentation

    Rückgabe:
    - pd.DataFrame: Original-DataFrame mit zusätzlicher Spalte 'predicted_sentiment'
    """
    # CSV-Datei einlesen
    df = pd.read_csv(filepath)

    # Sicherstellen, dass die Spalte 'text' vorhanden ist
    if 'text' not in df.columns:
        raise ValueError("Spalte 'text' fehlt.")

    # Entfernen von Zeilen mit fehlendem Text
    df = df.dropna(subset=['text'])

    # Entfernen von sehr kurzen Texten (≤ 5 Zeichen)
    df = df[df['text'].str.len() > 5]

    # Sicherstellen, dass alle Texte als String behandelt werden
    df['text'] = df['text'].astype(str)

    # Umwandlung jedes Texts in eine Feature-Vektor mit Bag-of-Words
    features = [bag_of_words(text, vokabular) for text in df['text']]

    # Vorhersage der Sentiments mithilfe des Modells
    df['predicted_sentiment'] = model.predict(features)

    # Rückgabe des DataFrames mit Sentiment-Ergebnissen
    return df

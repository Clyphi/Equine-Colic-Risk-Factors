#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: analyze_reddit
Author: Claudia Leins
Description: Sentiment analysis of Reddit posts for colic-related content
"""

import sys
import pandas as pd
from pathlib import Path
import nltk

# NLTK Ressourcen sicherstellen
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Projektpfade konfigurieren
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # 3x parent, um aus src/analysis/ zum Root zu gelangen
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.filter_keywords import is_colic_related
from nltk.sentiment import SentimentIntensityAnalyzer

# Pfade zu Ein- und Ausgabedateien
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "reddit_colic_posts.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analyzed_posts.csv"


def load_data():
    """Lädt Reddit-Daten und kombiniert Titel und Text zu einer Spalte."""
    df = pd.read_csv(CSV_PATH)
    df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    return df


def analyze_posts(df):
    """
    Führt Kolik-Erkennung und Sentiment-Analyse durch.
    
    Args:
        df: DataFrame mit Reddit-Posts
        
    Returns:
        DataFrame mit analysierten Kolik-relevanten Posts und Sentiment-Scores
    """
    # Kolik-Erkennung
    df['is_colic'] = df['full_text'].apply(is_colic_related)
    colic_posts = df[df['is_colic']].copy()
    
    # Sentiment-Analyse
    sia = SentimentIntensityAnalyzer()
    colic_posts['sentiment'] = colic_posts['full_text'].apply(
        lambda x: sia.polarity_scores(x)['compound']
    )
    return colic_posts


if __name__ == "__main__":
    print("⏳ Lade Daten...")
    data = load_data()
    
    print("🔍 Analysiere Posts...")
    analyzed = analyze_posts(data)
    
    analyzed.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Ergebnisse gespeichert unter: {OUTPUT_PATH}")
    print(f"📊 Gefundene Kolik-Posts: {len(analyzed)}/{len(data)}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: sentiment.py
Author: Claudia Leins
Description: Sentiment analysis
"""
import pandas as pd
from src.preprocessing.filter_keywords import is_colic_related
from src.preprocessing.clean_text import clean_text  # Deine Bereinigungsfunktion

def analyze_colic_posts(input_csv: str, output_csv: str):
    """Hauptfunktion: Filtert Kolik-Posts und analysiert Sentiment."""
    df = pd.read_csv(input_csv)
    
    # 1. Bereinigung
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 2. Kolik-Posts filtern
    df['is_colic'] = df['clean_text'].apply(is_colic_related)
    colic_posts = df[df['is_colic']].copy()
    
    # 3. Sentiment-Analyse (hier mit VADER)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    colic_posts['sentiment'] = colic_posts['clean_text'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    
    colic_posts.to_csv(output_csv, index=False)
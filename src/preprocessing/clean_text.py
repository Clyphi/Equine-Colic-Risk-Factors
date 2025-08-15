#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: clean_text
Author: Claudia Leins
Description: Sentiment analysis
"""
"""Bereinigt Rohtexte aus Reddit-Posts für die Sentiment-Analyse."""
import re
import pandas as pd
from typing import List

def remove_urls(text: str) -> str:
    """Entfernt URLs aus Text."""
    return re.sub(r'http\S+|www\S+|https\S+', '', text)

def clean_csv(input_path: str, output_path: str) -> None:
    """Hauptfunktion: Lädt CSV, bereinigt Texte, speichert Ergebnis."""
    df = pd.read_csv(input_path)
    df['clean_text'] = df['text'].apply(remove_urls)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/cleaned.csv")
    args = parser.parse_args()
    
    clean_csv(args.input, args.output)
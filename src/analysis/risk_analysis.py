#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: risk_analysis
Author: Claudia Leins
Module: risk_analysis
Description: RiskFactorAnalyzer 
"""
from pathlib import Path
import sys
import pandas as pd

# Projekt-Root setzen
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ..database.create_db import db, Task, app

class RiskFactorAnalyzer:
    def load_data_from_db(self):
        """Lädt alle Posts aus der DB als DataFrame."""
        with app.app_context():
            tasks = Task.query.all()
            data = [{
                "title": t.title,
                "full_text": t.full_text,
                "date": t.date,
                "horse_age": t.horse_age,
                "horse_gender": t.horse_gender,
                "colic_keywords": t.colic_keywords,
                "weather_keywords": t.weather_keywords,
                "feed_keywords": t.feed_keywords,
                "weather_count": t.weather_count,
                "feed_count": t.feed_count
            } for t in tasks]
        df = pd.DataFrame(data)
        return df

    def analyze_risk_factors(self, df=None):
        """Analysiert Risikofaktoren anhand der gespeicherten Keyword-Spalten."""
        if df is None:
            df = self.load_data_from_db()
        
        df = df.copy()
        # Als Kolik-Post gilt, wenn colic_keywords nicht leer ist
        df['is_colic'] = df['colic_keywords'].astype(bool)
        colic_posts = df[df['is_colic']]

        risk_results = {
            "total_posts": len(df),
            "total_colic_posts": len(colic_posts),
            "weather_related": colic_posts['weather_count'].sum(),
            "feed_related": colic_posts['feed_count'].sum()
        }
        return risk_results, df

    def calculate_risk_statistics(self, risk_results):
        """Berechnet die Prozentanteile für die Risikofaktoren."""
        total_colic = risk_results['total_colic_posts']
        stats = {
            'weather_related': {
                'count': risk_results['weather_related'],
                'percentage': (risk_results['weather_related'] / total_colic) * 100 if total_colic else 0,
                'description': 'Wetter-bezogene Risikofaktoren'
            },
            'feed_related': {
                'count': risk_results['feed_related'],
                'percentage': (risk_results['feed_related'] / total_colic) * 100 if total_colic else 0,
                'description': 'Fütterungs-bezogene Risikofaktoren'
            }
        }
        return stats

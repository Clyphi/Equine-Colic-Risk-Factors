#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: create_synthetic_data.py
Author: Claudia Leins
Description: Create synthetic data for equine colic analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

class SyntheticDataGenerator:
    """Generate synthetic equine colic data for analysis"""
    
    def __init__(self, num_rows=500, start_date="2020-01-01", end_date="2023-12-31"):
        self.num_rows = num_rows
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Mögliche Werte für die Generierung
        self.genders = ['mare', 'stallion', 'gelding']
        self.colic_types = ['spasmodic', 'impaction', 'gas', 'sand', 'displacement', 'torsion']
        self.locations = ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 
                         'Stuttgart', 'Dresden', 'Hanover', 'Leipzig', 'Nuremberg']
        self.horse_keeping =['stable','pasture','paddock']
        self.breed = ['Lipizzaner',
                      'Holsteiner',
                      'Württemberger',
                      'Arabisches Vollblut',
                      'Englische Vollblüter',
                      'American Quarter Horse',
                      'Paint Horse',
                      'Friese',
                      'Andalusier',
                      'Appaloosa',
                      'Hannoveraner',
                      'Oldenburger',
                      'Westfale',
                      'Trakehner',
                      'Shetlandpony',
                      'Haflinger',
                      'Deutsches Reitpony',
                      'Welsh Pony',
                      'Islandpferd']
        
        # Wahrscheinlichkeiten für verschiedene Koliktypen
        self.colic_probabilities = {
            'spasmodic': 0.25,
            'impaction': 0.20,
            'gas': 0.15,
            'sand': 0.10,
            'displacement': 0.20,
            'torsion': 0.10
        }
        
        # Wahrscheinlichkeiten für verschiedene Futterarten
        self.feed_probabilities = {
            'hay': 0.3,
            'haylage': 0.20,
            'silage': 0.15,
            'fresh grass': 0.10,
            'straw': 0.20,
            'corn': 0.10,
            'pellets': 0.20
        }
    
        
    def generate_data(self):
        """Generate synthetic data with colic cases"""
        data = []
        
        for i in range(self.num_rows):
            # Datum generieren (mehr Fälle in kälteren Monaten)
            days_diff = (self.end_date - self.start_date).days
            random_days = random.randint(0, days_diff)
            date = self.start_date + timedelta(days=random_days)
            
            # Höhere Wahrscheinlichkeit für Kolik in Wintermonaten
            month = date.month
            colic_chance = 0.8 if month in [11, 12, 1, 2] else 0.6  # 80% im Winter, 60% im Sommer
            
            has_colic = random.random() < colic_chance
            
            # Pferde-Alter und Geschlecht
            breed = random.choice(self.breed)
            age = random.randint(1, 30)
            gender = random.choice(self.genders)
            location = random.choice(self.locations)
            horse_keeping = random.choice(self.horse_keeping)
            # 1-2 Futterarten pro Fall
            num_types = random.choices([1, 2], weights=[0.7, 0.3])[0]
            feed_keywords = random.choices(
                list(self.feed_probabilities.keys()),
                weights=list(self.feed_probabilities.values()),
                k=num_types
            )
            # Duplikate entfernen und sortieren
            feed_keywords = sorted(set(feed_keywords))
                
            # Kolik-Keywords generieren
            if has_colic:
                # 1-2 Koliktypen pro Fall
                num_types = random.choices([1, 2], weights=[0.7, 0.3])[0]
                colic_keywords = random.choices(
                    list(self.colic_probabilities.keys()),
                    weights=list(self.colic_probabilities.values()),
                    k=num_types
                )
                # Duplikate entfernen und sortieren
                colic_keywords = sorted(set(colic_keywords))
            else:
                colic_keywords = []
            
            data.append({
                "title": "",
                "full_text": "",
                'date': date.strftime('%Y-%m-%d'),
                'location': location,
                'breed': breed,
                'horse_age': age,
                'horse_gender': gender,
                'colic_keywords': ', '.join(colic_keywords),
                'weather_keywords': None,
                'feed_keywords': ', '.join(feed_keywords),
                'weather_count': 0,
                'feed_count': len(feed_keywords) if feed_keywords else 0,
                'latitude': None,
                'longitude': None,
                'weather_tmax': None,
                'weather_tmin': None,
                'weather_precip': None,
                'horse_keeping': horse_keeping
            })
        
        return pd.DataFrame(data)
    
        def save_to_csv(self, df, file_path):
            """Save generated data to CSV file"""
            # Projekt-Root setzen
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT))

            # Pfade zu Keyword-Dateien
            SYNTHETIC_COLIC_DATA_PATH = PROJECT_ROOT / "data/processed/synthetic_colic_data.csv"

            # In CSV speichern
            df.to_csv(SYNTHETIC_COLIC_DATA_PATH, index=False)
            print(f"Synthetic data generated and saved to {SYNTHETIC_COLIC_DATA_PATH}")
            print(f"Generated {len(df)} records")
            print(f"Colic cases: {len(df[df['colic_keywords'] != ''])} ({len(df[df['colic_keywords'] != ''])/len(df)*100:.1f}%)")
            
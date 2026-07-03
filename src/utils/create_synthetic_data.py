#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: create_synthetic_data.py
Author: Claudia Leins
Description: Generate synthetic equine colic data with balanced classes and stronger causal signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import sys

# Projekt-Root setzen
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.age_group_generator import AgeGroupGenerator


class SyntheticDataGenerator:
    """Generate synthetic equine colic data with stronger causal signals"""
    
    def __init__(self, num_rows=2000, start_date="2020-01-01", end_date="2023-12-31"):
        self.num_rows = num_rows
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Grunddaten
        self.genders = ['mare', 'stallion', 'gelding']
        self.colic_types = ['spasmodic', 'impaction', 'gas', 'sand', 'displacement', 'torsion']
        self.locations = ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 
                          'Stuttgart', 'Dresden', 'Hanover', 'Leipzig', 'Nuremberg']
        self.horse_keeping = ['stable', 'pasture', 'paddock']
        self.breeds = [
            'Lipizzaner', 'Holsteiner', 'Württemberger', 'Arabisches Vollblut',
            'Englische Vollblüter', 'American Quarter Horse', 'Paint Horse',
            'Friese', 'Andalusier', 'Appaloosa', 'Hannoveraner', 'Oldenburger',
            'Westfale', 'Trakehner', 'Shetlandpony', 'Haflinger',
            'Deutsches Reitpony', 'Welsh Pony', 'Islandpferd'
        ]
        self.feeds = ['hay', 'haylage', 'silage', 'fresh grass', 'straw', 'corn', 'pellets']

        # Wahrscheinlichkeiten (für Restzufall)
        self.colic_probabilities = {
            'spasmodic': 0.2,
            'impaction': 0.2,
            'gas': 0.2,
            'sand': 0.15,
            'displacement': 0.15,
            'torsion': 0.1
        }

    # ----------------------------------------------------------
    # 🧠 Realistische Koliktyp-Zuordnung mit klaren Regeln
    # ----------------------------------------------------------
    def _get_colic_type(self, age, feed_main, horse_keeping, breed):
        """Kausal getriebene Koliktyp-Auswahl"""
        # Altersregeln
        if age > 20 and feed_main == 'straw':
            return 'impaction'
        elif age < 5 and feed_main == 'fresh grass':
            return 'gas'
        # Haltung & Futter
        if horse_keeping == 'stable' and feed_main == 'silage':
            return 'displacement'
        elif horse_keeping == 'paddock' and feed_main in ['hay', 'straw']:
            return 'sand'
        # Rassenabhängig
        if 'Haflinger' in breed and feed_main in ['hay', 'straw']:
            return 'sand'
        if 'Arabisches' in breed and feed_main == 'fresh grass':
            return 'gas'
        # Zufall für Rest
        return random.choices(
            list(self.colic_probabilities.keys()),
            weights=list(self.colic_probabilities.values()),
            k=1
        )[0]

    # ----------------------------------------------------------
    # 🧩 Daten generieren
    # ----------------------------------------------------------
    def generate_data(self):
        """Generate synthetic dataset with balanced colic types"""
        data = []
        n_per_type = self.num_rows // len(self.colic_types)

        for colic_type in self.colic_types:
            for _ in range(n_per_type):
                # Datum (mehr im Winter)
                days_diff = (self.end_date - self.start_date).days
                date = self.start_date + timedelta(days=random.randint(0, days_diff))
                month = date.month

                horse_keeping = random.choices(self.horse_keeping, k=1)[0]
                breed = random.choice(self.breeds)
                age = random.randint(1, 30)
                gender = random.choice(self.genders)
                location = random.choice(self.locations)
                feed_main = random.choice(self.feeds)

                # Koliktype deterministisch setzen
                colic = self._get_colic_type(age, feed_main, horse_keeping, breed)
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'location': location,
                    'breed': breed,
                    'horse_age': age,
                    'horse_gender': gender,
                    'colic_keywords': colic,
                    'feed_keywords': feed_main,
                    'horse_keeping': horse_keeping,
                    'is_synthetic': True
                })

        # Zufällige Restzeilen, falls num_rows nicht exakt teilbar
        while len(data) < self.num_rows:
            days_diff = (self.end_date - self.start_date).days
            date = self.start_date + timedelta(days=random.randint(0, days_diff))
            horse_keeping = random.choice(self.horse_keeping)
            breed = random.choice(self.breeds)
            age = random.randint(1, 30)
            gender = random.choice(self.genders)
            location = random.choice(self.locations)
            feed_main = random.choice(self.feeds)
            colic = self._get_colic_type(age, feed_main, horse_keeping, breed)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'location': location,
                'breed': breed,
                'horse_age': age,
                'horse_gender': gender,
                'colic_keywords': colic,
                'feed_keywords': feed_main,
                'horse_keeping': horse_keeping,
                'is_synthetic': True
            })

        # DataFrame & Altersgruppen
        df = pd.DataFrame(data)
        age_group_gen = AgeGroupGenerator()
        df = age_group_gen.add_age_group(df)
        return df

    # ----------------------------------------------------------
    # 💾 CSV speichern
    # ----------------------------------------------------------
    def save_to_csv(self, df, file_path=None):
        if file_path is None:
            file_path = PROJECT_ROOT / "data/processed/synthetic_colic_data.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ Synthetic data saved to {file_path}")
        print(f"📊 {len(df)} records generated")
        print(f"🐎 Colic cases per type:\n{df['colic_keywords'].value_counts()}")

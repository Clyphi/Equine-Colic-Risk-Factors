#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: synthetic_data_pipeline
Author: Claudia Leins
Description: Workflow:  - Create synthetic data with create_synthetic_data.py 
                        - Load cleaned data into SQLite DB
"""
from pathlib import Path
import sys

# Projekt-Root setzen
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.create_synthetic_data import SyntheticDataGenerator
from src.database.load_to_db import DatabaseLoader
from src.preprocessing.weather_manager import WeatherManager

if __name__ == '__main__':
    
    sdg = SyntheticDataGenerator()
    df_synthetic = sdg.generate_data()
    # print(df_synthetic)
        
    # Synthetische Daten in DB laden
    loader = DatabaseLoader(df_synthetic)
    loader.load_synthetic_data()    
    
    # Wetterdaten nachtragen
    print("Starte WeatherManager...")
    wm = WeatherManager()
    print("Rufe bulk_update_missing auf...")
    wm.bulk_update_missing()
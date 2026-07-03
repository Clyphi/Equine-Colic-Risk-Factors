#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: DataCleaner
Author: Claudia Leins
Module: data_cleaner
Description: Enthält die DataCleaner-Klasse zum Bereinigen von Reddit-Daten
"""
from pathlib import Path
import sys
import re
import pandas as pd
from datetime import datetime

# Projekt-Root setzen
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.keyword_manager import KeywordManager

class DataCleaner:
    def __init__(self, df, colic_keywords_path, weather_keywords_path, feed_keywords_path):
        self.df = df
        self.weather_manager = KeywordManager(weather_keywords_path)
        self.feed_manager = KeywordManager(feed_keywords_path)
        self.colic_manager = KeywordManager(colic_keywords_path) 
        
        # Mapping für Zahlwörter
        self.number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12
        }
        
        # Regex-Muster vorbereiten
        self.patterns = [
            re.compile(r"\b(\d{1,2})\s*(?:years?|yrs?)\s*old\b", re.IGNORECASE),
            re.compile(r"\b(\d{1,2})-?\s*year-?old\b", re.IGNORECASE),
            re.compile(r"\b(\d{1,2})\s*(?:yrs?|years?)\b", re.IGNORECASE),
            re.compile(r"\bborn\s+in\s+((?:19|20)\d{2})\b", re.IGNORECASE)
        ]
        self.word_pattern = re.compile(
            r"\b(" + "|".join(self.number_words.keys()) + r")-?year-?old\b", re.IGNORECASE
        )
        
        # Regex für Geschlecht
        self.gender_map = {
            "mare": ["mare", "female horse", "filly"],
            "stallion": ["stallion", "male horse", "colt"],
            "gelding": ["gelding", "gelded horse", "castrated male"]
        }
        self.gender_patterns = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in self.gender_map.items()}

    def find_keywords(self, text, keyword_list):
        """Findet alle Keywords im Text (case-insensitive, ganze Wörter, Satzzeichen ignoriert)."""
        if not text or pd.isna(text):
            return ""
        
        text_lower = re.sub(r'[^\w\s]', ' ', str(text).lower())  # Satzzeichen entfernen
        found = []
        for keyword in keyword_list:
            # Suche nur ganze Wörter
            if re.search(rf'\b{re.escape(keyword.lower())}\b', text_lower):
                found.append(keyword.lower())
        return ','.join(found)
    
    def extract_horse_age(self, text):
        text = str(text).lower()
        for pattern in self.patterns[:3]:
            m = pattern.search(text)
            if m:
                return int(m.group(1))
        m = self.patterns[3].search(text)
        if m:
            birth_year = int(m.group(1))
            return datetime.now().year - birth_year
        m = self.word_pattern.search(text)
        if m:
            word = m.group(1).lower()
            return self.number_words.get(word)
        return None
    
    def extract_horse_gender(self, text: str) -> str:
        for gender, pattern in self.gender_patterns.items():
            if pattern.search(text):
                return gender
        return "unknown"

    def data_to_table(self):
        """Erstellt eine bereinigte Tabelle für die Datenbank"""
        self.table = []
        self.df["colic_keywords"] = self.df.apply(
                lambda row: ", ".join(self.colic_manager.extract_keywords(str(row["title"]) + " " + str(row["text"]))),
                axis=1
            )
        self.colic_manager.analyze_and_update(self.df, column="colic_keywords", top_n=30)
        self.weather_keywords = self.weather_manager.load_keywords()
        self.feed_keywords = self.feed_manager.load_keywords()
        self.colic_keywords = self.colic_manager.load_keywords()
        
        for _, row in self.df.iterrows():
            text = str(row.get('text', ''))
            title = str(row.get('title', ''))
            
            weather_results = self.find_keywords(text, self.weather_keywords)
            feed_results = self.find_keywords(text, self.feed_keywords)
            colic_results = self.find_keywords(text, self.colic_keywords)
            
            horse_age = self.extract_horse_age(text)
            horse_gender = self.extract_horse_gender(text)
            
            self.table.append({
                'title': title,
                'full_text': text,
                'date': row.get('date'),
                'horse_age': horse_age,
                'horse_gender': horse_gender,
                'colic_keywords': colic_results,
                'weather_keywords': weather_results,
                'feed_keywords': feed_results,
                'weather_count': len(weather_results.split(',')) if weather_results else 0,
                'feed_count': len(feed_results.split(',')) if feed_results else 0
                'is_synthetic': False
            })
    
        print(f"✅ {len(self.table)} Zeilen bereinigt und bereit für DB-Import")
        return self.table

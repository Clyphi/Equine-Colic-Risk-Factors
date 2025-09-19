#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: keyword_manager
Author: Claudia Leins
Module: keyword_manager
Description: Enthält die KeywordManager-Klasse
"""

import os
import pandas as pd
from typing import List
from collections import Counter

class KeywordManager:
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        
    def load_keywords(self) -> List[str]:
        """Lädt Keywords aus TXT-Datei (ignoriert Kommentare mit #)."""
        if self.filepath is None or not os.path.exists(self.filepath):
            return []
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f 
                    if line.strip() and not line.startswith("#")]
    
    def is_colic_related(self, text):
        """Prüft, ob Text kolik-relevant ist basierend auf Keywords."""
        if not text or pd.isna(text):
            return False
            
        text_lower = str(text).lower()
        keywords = self.load_keywords()
        
        return any(keyword in text_lower for keyword in keywords)

    def extract_keywords(self, text) -> List[str]:
        """Gibt alle im Text vorkommenden Keywords zurück."""
        if not text or pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        keywords = self.load_keywords()
        return [kw for kw in keywords if kw in text_lower]
    
    def analyze_and_update(self, df: pd.DataFrame, column: str = "colic_keywords", top_n: int = 30):
        """
        Analysiert die Keywords in einer DataFrame-Spalte und fügt neue, fehlende Keywords
        automatisch zur Keyword-Datei hinzu.
        """
        if self.filepath is None:
            raise ValueError("Pfad zur Keyword-Datei ist nicht gesetzt!")

        # Bestehende Keywords laden
        existing_keywords = set(self.load_keywords())

        # Alle Tokens aus der Spalte sammeln
        all_tokens = []
        for val in df[column].dropna():
            tokens = [t.strip().lower() for t in str(val).split(",") if t.strip()]
            all_tokens.extend(tokens)

        counter = Counter(all_tokens)

        print(f"🔎 Insgesamt {len(counter)} unterschiedliche Keywords gefunden\n")
        print(f"📊 Top {top_n} Keywords:")
        for kw, count in counter.most_common(top_n):
            mark = "✅" if kw in existing_keywords else "❌"
            print(f"{mark} {kw}: {count}")

        # Fehlende Keywords identifizieren
        missing = [kw for kw in counter if kw not in existing_keywords]
        if missing:
            print(f"\n⚠️ {len(missing)} neue Keywords gefunden, werden zur Liste hinzugefügt:")
            print(", ".join(sorted(missing)))
            updated_keywords = existing_keywords.union(missing)

            # Datei aktualisieren
            with open(self.filepath, "w", encoding="utf-8") as f:
                for kw in sorted(updated_keywords):
                    f.write(kw + "\n")
            print(f"✅ Keyword-Datei aktualisiert: {self.filepath}")
        else:
            print("\n✅ Keine neuen Keywords – alles schon in der Liste enthalten!")
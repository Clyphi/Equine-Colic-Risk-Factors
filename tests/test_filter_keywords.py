#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: test_filter_keywords
Author: Claudia Leins
Description: Sentiment analysis
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: test_filter_keywords
Author: Claudia Leins
Description: Sentiment analysis
"""
""" import sys
from pathlib import Path
import pytest

# Füge das Projekt-Root zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.filter_keywords import is_colic_related
from src.utils.file_loader import load_keywords

def test_keyword_loading():
    """'Testet, ob Keywords korrekt geladen werden.'"""
    keywords = load_keywords()
    assert isinstance(keywords, list), "Keywords sollten eine Liste sein"
    assert len(keywords) >= 5, "Mindestens 5 Keywords erwartet"
    assert "colic" in keywords, "'colic' sollte in Keywords sein"
    print("✅ test_keyword_loading bestanden")

@pytest.mark.parametrize("text,expected", [
    ("My horse has colic", True),
    ("Pferd mit Kolik", True),
    ("Emergency vet!", True),
    ("Healthy horse", False),
    ("", False),
    (None, False),
])
def test_is_colic_related(text, expected):
    result = is_colic_related(text)
    assert result == expected, f"Für Text '{text}' wurde {result} erhalten, aber {expected} erwartet"
    print(f"✅ test_is_colic_related bestanden für: '{text}'")

if __name__ == "__main__":
    # Führe Tests explizit aus und zeige Ergebnisse
    print("\n=== Manuelle Testausführung ===")
    try:
        test_keyword_loading()
        test_is_colic_related("My horse has colic", True)
        test_is_colic_related("Healthy horse", False)
        print("=== Alle Tests erfolgreich! ===")
    except AssertionError as e:
        print(f"❌ Test fehlgeschlagen: {e}") """
import sys        
import pandas as pd
from pathlib import Path

# Füge das Projekt-Root zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.filter_keywords import is_colic_related
from src.utils.file_loader import load_keywords




# Pfad zur CSV-Datei
CSV_PATH = Path(__file__).parent.parent / "data" / "raw" / "reddit_colic_posts.csv"

def load_and_prepare_data():
    """Lädt die Reddit-Daten und bereitet sie vor."""
    df = pd.read_csv(CSV_PATH)
    
    # Grundlegende Bereinigung
    df['text'] = df['text'].astype(str)
    df['title'] = df['title'].astype(str)
    
    # Kombiniere Titel und Text für bessere Analyse
    df['full_text'] = df['title'] + " " + df['text']
    
    return df

if __name__ == "__main__":
    # Daten laden und anzeigen
    reddit_data = load_and_prepare_data()
    print("\nBeispiel-Daten aus der CSV:")
    print(reddit_data[['full_text']].head(3))
    
    # Teste die Kolik-Erkennung
    sample_text = reddit_data.iloc[0]['full_text']
    print(f"\nErster Eintrag ist Kolik-relevant?: {is_colic_related(sample_text)}")        
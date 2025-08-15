#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: file_loader
Author: Claudia Leins
Description: Sentiment analysis
"""
from typing import List
import os

def load_keywords(filepath: str = None) -> List[str]:
    """Lädt Kolik-Begriffe aus TXT-Datei (ignoriert Kommentare mit #)."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "../../data/keywords/colic_keywords.txt")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith("#")]
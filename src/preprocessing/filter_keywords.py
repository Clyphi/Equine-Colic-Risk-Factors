#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: filter_keywords
Author: Claudia Leins
Description: Sentiment analysis
"""

from src.utils.file_loader import load_keywords
from typing import Optional

def is_colic_related(text: Optional[str]) -> bool:
    """Prüft, ob Text Kolik- oder Wetterbegriffe enthält."""
    if not text:
        return False
    
    keywords = load_keywords()
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


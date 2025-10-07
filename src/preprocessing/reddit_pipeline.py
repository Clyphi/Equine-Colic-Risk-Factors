#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: reddit_pipeline
Author: Claudia Leins
Description: Workflow:  - Fetch reddit posts
                        - Clean data
                        - Load cleaned data into SQLite DB
"""
import pandas as pd
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Projekt-Root setzen
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_cleaner import DataCleaner
from src.database.load_to_db import DatabaseLoader
from src.collection.reddit_colic_posts import RedditFetcher
from src.utils.keyword_manager import KeywordManager

# Rohdaten sammeln
# Keys aus .env
client_id = os.getenv("REDDIT_ID")
client_secret = os.getenv("REDDIT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

fetcher = RedditFetcher(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)
raw_data = fetcher.fetch_posts()

# Pfade zu Keyword-Dateien
colic_keywords_path = PROJECT_ROOT / "data/keywords/colic_keywords.txt"
weather_keywords_path = PROJECT_ROOT / "data/keywords/weather_keywords.txt"
feed_keywords_path = PROJECT_ROOT / "data/keywords/horse_feed_keywords.txt"

# DataCleaner instanzieren und Daten bereinigen
cleaner = DataCleaner(
    df=raw_data,
    colic_keywords_path=colic_keywords_path,
    weather_keywords_path=weather_keywords_path,
    feed_keywords_path=feed_keywords_path
)

cleaned_data = pd.DataFrame(cleaner.data_to_table())

# Bereinigte Daten in DB laden
loader = DatabaseLoader(cleaned_data)
loader.load()


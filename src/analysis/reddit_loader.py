#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: reddit_loader.py
Module: reddit_loader
Author: Claudia Leins
Description: Class to load Reddit data from the SQLite database into a pandas DataFrame
"""

import pandas as pd
from ..database.create_db import db, Task, app


class RedditDataLoader:
    """Loads data from the SQLite database into a DataFrame."""

    def __init__(self):
        pass

    def load_data_from_db(self) -> pd.DataFrame:
        """Fetches all posts from the database and returns them as a DataFrame."""
        with app.app_context():
            tasks = Task.query.all()
            data = [{
                "id": t.id,
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

        return pd.DataFrame(data)

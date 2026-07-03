#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: reddit_loader.py
Module: data_loader
Author: Claudia Leins
Description: Class to load Reddit data from the SQLite database into a pandas DataFrame
"""
import pandas as pd
from abc import ABC, abstractmethod
from .create_db import app, db, Task

class BaseDataLoader(ABC):
    def __init__(self, model, synthetic_only=False, real_only=False):
        self.model = model
        self.synthetic_only = synthetic_only
        self.real_only = real_only

    def _query_all(self):
        with app.app_context():
            query = self.model.query
            if self.synthetic_only:
                query = query.filter_by(is_synthetic=True)
            elif self.real_only:
                query = query.filter_by(is_synthetic=False)
            return query.all()

    @abstractmethod
    def load_data_from_db(self) -> pd.DataFrame:
        pass


class RedditProcessedDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__(Task, real_only=True)

    def load_data_from_db(self) -> pd.DataFrame:
        tasks = self._query_all()
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


class SyntheticDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__(Task, synthetic_only=True)

    def load_data_from_db(self) -> pd.DataFrame:
        cases = self._query_all()
        data = [{
            "id": c.id,
            "breed": c.breed,
            "horse_age": c.horse_age,
            "horse_gender": c.horse_gender,
            "feed_keywords": c.feed_keywords,
            "temp_max": c.weather_tmax,
            "temp_min": c.weather_tmin,
            "precipitation": c.weather_precip,
            "location": c.location,
            "colic_keywords": c.colic_keywords,
            "feed_count": c.feed_count,
            "horse_keeping": c.horse_keeping
        } for c in cases]
        return pd.DataFrame(data)


class ClinicalDataLoader(BaseDataLoader):
    def __init__(self):
       pass

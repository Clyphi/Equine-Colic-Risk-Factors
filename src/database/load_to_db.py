#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: load_to_db
Author: Claudia Leins (angepasst)
Description: Load cleaned data into SQLite DB as a class
"""

import os
import re
import json
import csv
import unicodedata
import pandas as pd
from ..database.create_db import db, Task, app


DEFAULT_COLIC_TERMS = [
    "kolik", "colic", "bauchschmerzen", "darmverschluss", "ileus",
    "verstopfung", "blähung", "blähungen", "aufgebläht", "aufgeblasen",
    "krampf", "wälzen", "rollen", "rollt", "unruhig", "schwitzen",
    "kolikverdacht", "kolikverdächtig", "appetitlos", "nicht fressen",
    "schmerz", "kolikschmerz", "kolikzeichen", "kolikepisode"
]


class DatabaseLoader:
    def __init__(self, df: pd.DataFrame, colic_terms: list | None = None,
                 colic_terms_file: str | None = None):
        """
        df: DataFrame mit mindestens den Spalten 'colic_keywords', 'title', 'full_text'
        colic_terms: optionale Liste von zusätzlichen Colic-Begriffen
        colic_terms_file: optionaler Pfad zu einer Datei (.txt, .csv, .json) mit Keywords
        """
        self.df = df

        if colic_terms_file:
            self.colic_terms = self._load_terms_from_file(colic_terms_file)
        else:
            terms = colic_terms if colic_terms is not None else DEFAULT_COLIC_TERMS
            self.colic_terms = set(self._normalize(t) for t in terms if t)

    # ----------------- Hilfsfunktionen -----------------
    def _normalize(self, s: str) -> str:
        """Normalisiert Text: lower, strip, ohne Diakritika, komprimierte Leerzeichen."""
        if not isinstance(s, str):
            return ""
        s = s.lower().strip()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", s)

    def _split_field(self, field) -> list:
        """Zerlegt kommaseparierten Text oder Liste in normalisierte Tokens."""
        if pd.isna(field) or field is None:
            return []
        if isinstance(field, (list, tuple, set)):
            tokens = list(field)
        else:
            tokens = re.split(r"[,;|/]+", str(field))
        return [self._normalize(t) for t in tokens if t.strip()]

    def _extract_keywords(self, *fields) -> list:
        """
        Extrahiert alle Keywords, die in den gegebenen Textfeldern vorkommen.
        """
        found = set()
        for field in fields:
            if not isinstance(field, str):
                continue
            text = self._normalize(field)
            for kw in self.colic_terms:
                if kw in text:
                    found.add(kw)
        return sorted(found)

    def _load_terms_from_file(self, path: str) -> set:
        """Lädt Keywords aus txt, csv oder json Datei."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Colic terms file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        terms = []
        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                terms = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        elif ext == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        terms.append(row[0])
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    terms = data
                elif isinstance(data, dict):
                    for k in ("terms", "keywords", "colic_terms"):
                        if k in data and isinstance(data[k], list):
                            terms = data[k]
                            break
        else:
            raise ValueError(f"Unsupported keyword file format: {ext}")
        return set(self._normalize(t) for t in terms if t)

    # ----------------- Hauptfunktion -----------------
    def load(self):
        """
        Speichert nur Zeilen mit Colic-Bezug.
        In der DB stehen die tatsächlich gefundenen Keywords als kommagetrennte Zeichenkette.
        """
        inserted = 0
        with app.app_context():
            for _, row in self.df.iterrows():
                colic_field = row.get("colic_keywords", "")
                title = row.get("title", "")
                full_text = row.get("full_text", "")

                # Alle Quellen prüfen: vorhandene colic_keywords-Spalte, Title, Fulltext
                tokens = self._split_field(colic_field)
                found_keywords = set(tokens) | set(self._extract_keywords(title, full_text))

                if not found_keywords:
                    continue  # keine relevanten Keywords → nicht speichern

                task = Task(
                    title=title or "",
                    full_text=full_text or "",
                    date=pd.to_datetime(row.get("date")) if pd.notna(row.get("date")) else None,
                    location = row.get("location"),
                    horse_age=row.get("horse_age"),
                    horse_gender=row.get("horse_gender"),
                    colic_keywords=", ".join(sorted(found_keywords)),
                    weather_keywords=row.get("weather_keywords"),
                    feed_keywords=row.get("feed_keywords"),
                    weather_count=row.get("weather_count", 0),
                    feed_count=row.get("feed_count", 0)
                )
                db.session.add(task)
                inserted += 1

            db.session.commit()

        print(f"✅ {inserted} Zeilen mit Colic-Bezug in die Datenbank importiert")

    def load_synthetic_data(self):
        """
        Speichert nur Zeilen mit Colic-Bezug für synthetische Daten.
        """
        inserted = 0
        with app.app_context():
            for _, row in self.df.iterrows():
                colic_field = row.get("colic_keywords", "")
                tokens = self._split_field(colic_field)
                found_keywords = set(tokens) 

                if not found_keywords:
                    continue

                task = Task(
                    title=row.get("title", ""),
                    full_text="",  # Da full_text in synthetischen Daten nicht vorhanden ist
                    date=pd.to_datetime(row.get("date")) if pd.notna(row.get("date")) else None,
                    location=row.get("location", ""),
                    horse_age=row.get("horse_age", ""),
                    horse_gender=row.get("horse_gender", ""),
                    colic_keywords=", ".join(sorted(found_keywords)),
                    weather_keywords=row.get("weather_keywords", ""),
                    feed_keywords=row.get("feed_keywords", ""),
                    weather_count=row.get("weather_count", 0),
                    feed_count=row.get("feed_count", 0),
                    latitude=row.get("latitude", None),
                    longitude=row.get("longitude", None),
                    weather_tmax=row.get("weather_tmax", None),
                    weather_tmin=row.get("weather_tmin", None),
                    weather_precip=row.get("weather_precip", None)
                )
                db.session.add(task)
                inserted += 1

            db.session.commit()
        print(f"✅ {inserted} Zeilen mit Colic-Bezug in die Datenbank importiert")
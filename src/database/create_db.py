#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: create_db
Author: Claudia Leins
Description: Creates the SQLite database for the project
"""
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Verzeichnis des aktuellen Skripts
# Verzeichnis des aktuellen Skripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Eine Ebene zurück zum Projektverzeichnis
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# Datenbankverzeichnis unter Projektname/data/database
db_dir = os.path.join(BASE_DIR, "data", "database")
os.makedirs(db_dir, exist_ok=True)

# Pfad zur SQLite-Datenbank
db_path = os.path.join(db_dir, "equine_colic.db")
print("DB-Pfad:", db_path)

# Flask-App
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Datenbank initialisieren
db = SQLAlchemy(app)

# Model für die Analyse
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    full_text = db.Column(db.Text, nullable=True)
    date = db.Column(db.Date, nullable=True)
    location = db.Column(db.Text, nullable=True)
    breed = db.Column(db.String(120), nullable=True)
    horse_age = db.Column(db.Integer, nullable=True)
    horse_gender = db.Column(db.String(10), nullable=True)
    colic_keywords = db.Column(db.Text, nullable=True)
    weather_keywords = db.Column(db.Text, nullable=True)
    feed_keywords = db.Column(db.Text, nullable=True)
    weather_count = db.Column(db.Integer, default=0)
    feed_count = db.Column(db.Integer, default=0)
    # 🌍 neue Spalten für Geodaten und Wetter
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    weather_tmax = db.Column(db.Float)
    weather_tmin = db.Column(db.Float)
    weather_precip = db.Column(db.Float)
    horse_keeping = db.Column(db.String(10), nullable=True) # stable, pasture, paddock
    is_synthetic = db.Column(db.Boolean, default=False)

    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "full_text": self.full_text,
            "date": self.date.isoformat() if self.date else None,
            "location": self.location,
            "breed": self.breed,
            "horse_age": self.horse_age,
            "horse_gender": self.horse_gender,
            "colic_keywords": self.colic_keywords,
            "weather_keywords": self.weather_keywords,
            "feed_keywords": self.feed_keywords,
            "weather_count": self.weather_count,
            "feed_count": self.feed_count,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "weather_tmax": self.weather_tmax,
            "weather_tmin": self.weather_tmin,
            "weather_precip": self.weather_precip,
            "horse_keeping": self.horse_keeping,
            "is_synthetic": self.is_synthetic
        }

    def __repr__(self):
        return f"<Task {self.id}: {self.title}>"

# Datenbank erstellen
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"✅ Datenbank erstellt unter {db_path}")

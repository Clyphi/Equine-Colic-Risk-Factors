#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: weather_manager
Author: Claudia Leins
Description: Holt historische Wetterdaten von Open-Meteo und speichert sie in DB.
"""

import requests
from datetime import datetime
from src.database.create_db import db, Task, app  
from sqlalchemy import or_
import math
import time
import logging

# Logger einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherManager:
    def __init__(self):
        self.api_url = "https://archive-api.open-meteo.com/v1/archive"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

    def get_coordinates(self, city_name):
        """Wandelt Städtenamen in Koordinaten um."""
        try:
            params = {
                "name": city_name,
                "count": 1,
                "language": "de",
                "format": "json"
            }
            
            logger.info(f"Geocoding-Anfrage für: {city_name}")
            r = requests.get(self.geocoding_url, params=params, timeout=10)
            
            if r.status_code != 200:
                logger.warning(f"Geocoding-Fehler: Status Code {r.status_code}")
                return None, None
                
            data = r.json()
            
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                latitude = result["latitude"]
                longitude = result["longitude"]
                logger.info(f"Koordinaten für {city_name}: ({latitude}, {longitude})")
                return latitude, longitude
            else:
                logger.warning(f"Keine Koordinaten gefunden für: {city_name}")
                return None, None
                
        except Exception as e:
            logger.error(f"Geocoding-Fehler für {city_name}: {e}")
            return None, None

    def get_weather(self, lat, lon, date: datetime):
        """Holt Wetterdaten für ein bestimmtes Datum/Koordinaten."""
        try:
            if lat is None or lon is None:
                logger.warning("Ungültige Koordinaten")
                return None
                
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date.strftime("%Y-%m-%d"),
                "end_date": date.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "Europe/Berlin"
            }
            
            logger.info(f"Wetter-API-Anfrage für {date.strftime('%Y-%m-%d')} bei ({lat}, {lon})")
            
            r = requests.get(self.api_url, params=params, timeout=10)
            
            if r.status_code != 200:
                logger.warning(f"Wetter-API-Fehler: Status Code {r.status_code}")
                return None
            
            if not r.content:
                logger.warning("Leere API-Antwort erhalten")
                return None
                
            data = r.json()
            
            if "daily" in data:
                return {
                    "tmax": data["daily"]["temperature_2m_max"][0],
                    "tmin": data["daily"]["temperature_2m_min"][0],
                    "precip": data["daily"]["precipitation_sum"][0]
                }
            else:
                logger.warning(f"Keine Wetterdaten in API-Antwort")
                return None
                
        except Exception as e:
            logger.error(f"Fehler bei Wetter-API-Anfrage: {e}")
            return None

    def bulk_update_missing(self):
        """Füllt Wetterdaten für Tasks nach, die noch keine haben."""
        with app.app_context():
            tasks = Task.query.filter(
                or_(
                    Task.weather_tmax == None,
                    Task.weather_tmax == 0.0
                )
            ).all()
            
            successful_updates = 0
            
            for i, task in enumerate(tasks):
                # Überspringe Tasks ohne Datum oder Ort
                if task.date is None or task.location is None:
                    logger.info(f"  → Überspringe Task {task.id}: Fehlendes Datum oder Ort")
                    continue
                
                logger.info(f"Verarbeite Task {task.id} ({i+1}/{len(tasks)}): {task.location}")
                
                # Hole Koordinaten für den Ort
                latitude, longitude = self.get_coordinates(task.location)
                
                if latitude is None or longitude is None:
                    logger.warning(f"  ❌ Keine Koordinaten für Ort '{task.location}' gefunden")
                    continue
                
                # Hole Wetterdaten mit den Koordinaten
                weather = self.get_weather(latitude, longitude, task.date)
                
                if weather:
                    task.weather_tmax = weather["tmax"]
                    task.weather_tmin = weather["tmin"]
                    task.weather_precip = weather["precip"]
                    task.latitude = latitude  # Speichere Koordinaten für zukünftige Use
                    task.longitude = longitude
                    successful_updates += 1
                    logger.info(f"  ✅ Wetterdaten für {task.location} aktualisiert")
                else:
                    logger.warning(f"  ❌ Keine Wetterdaten für {task.location} erhalten")
                
                # Kurze Pause zwischen Anfragen
                time.sleep(1)
            
            try:
                db.session.commit()
                logger.info(f"✅ Wetterdaten für {successful_updates} von {len(tasks)} Tasks erfolgreich ergänzt")
            except Exception as e:
                db.session.rollback()
                logger.error(f"❌ Datenbank-Fehler beim Commit: {e}")
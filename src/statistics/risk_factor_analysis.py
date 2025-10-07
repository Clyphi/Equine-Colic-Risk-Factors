#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: risk_analysis
Author: Claudia Leins
Module: risk_analysis
Description: RiskFactorAnalyzer mit Unterscheidung nach Datenquelle
"""

from pathlib import Path
import sys
import pandas as pd
from src.database.data_loader import RedditProcessedDataLoader, SyntheticDataLoader, ClinicalDataLoader

# Projekt-Root setzen
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ..database.create_db import db, Task, app


class RiskFactorAnalyzer:
    def __init__(self, loader_class):
        """Analyzer kann mit jedem Loader betrieben werden"""
        self.loader = loader_class()
        self.loader_type = loader_class.__name__  # z. B. "RedditProcessedDataLoader"

    def load_data_from_db(self):
        return self.loader.load_data_from_db()

    def analyze_risk_factors(self, df=None):
        """Analysiert Risikofaktoren, je nach Datenquelle unterschiedlich."""
        if df is None:
            df = self.load_data_from_db()
        df = df.copy()

        if self.loader_type == "RedditProcessedDataLoader":
            return self._analyze_reddit(df)
        elif self.loader_type == "SyntheticDataLoader":
            return self._analyze_synthetic(df)
        else:
            raise ValueError(f"Keine Analyse-Methode für {self.loader_type} definiert")

    # --- interne Analyse-Methoden ---
    def _analyze_reddit(self, df):
        """Analyse für Reddit-Daten"""
        df['is_colic'] = df['colic_keywords'].astype(bool)
        colic_posts = df[df['is_colic']]

        risk_results = {
            "total_posts": len(df),
            "total_colic_posts": len(colic_posts),
            "weather_related": colic_posts['weather_count'].sum(),
            "feed_related": colic_posts['feed_count'].sum()
        }
        return risk_results, df

    def _analyze_synthetic(self, df):
        """Analyse für synthetische Daten"""
        df['is_colic'] = df['colic_keywords'].astype(bool)
        colic_cases = df[df['is_colic']]

        risk_results = {
            "total_cases": len(df),
            "colic_cases": len(colic_cases),
            "age_most_common_count": colic_cases['horse_age'].value_counts().max(),
            "age_most_common": colic_cases['horse_age'].mode()[0] if not colic_cases['horse_age'].mode().empty else None,
            "feed_counts": colic_cases['feed_keywords'].value_counts().max(),
            "most_common_feed" : colic_cases['feed_keywords'].value_counts().max(),
            "most_common_breed" : colic_cases['breed'].value_counts().idxmax(),
            "temp_max_mean" : colic_cases['temp_max'].mean(),
            "temp_min_mean" : colic_cases['temp_min'].mean(),
            "precipitation_mean" : colic_cases['precipitation'].mean(),
        }
        return risk_results, df

    
    def calculate_risk_statistics(self, risk_results):
        if self.loader_type == "RedditProcessedDataLoader":
            return self.calculate_risk_statistics_for_reddit_data(risk_results)
        elif self.loader_type == "SyntheticDataLoader":
            return self.calculate_risk_statistics_for_synthetic_data(risk_results)
        else:
            raise ValueError(f"Keine Statistik-Methode für {self.loader_type} definiert")


    
    def calculate_risk_statistics_for_reddit_data(self, risk_results):
        total_colic = risk_results['total_colic_posts']
        stats = {
            'weather_related': {
                'count': risk_results['weather_related'],
                'percentage': (risk_results['weather_related'] / total_colic) * 100 if total_colic else 0,
                'description': 'Wetterbedingte Risikofaktoren'
            },
            'feed_related': {
                'count': risk_results['feed_related'],
                'percentage': (risk_results['feed_related'] / total_colic) * 100 if total_colic else 0,
                'description': 'Fütterungsbedingte Risikofaktoren'
            }
        }
        return stats
    
    def calculate_extended_risk_metrics(self, synthetic_data):
        """Berechnet erweiterte Risikometriken für die synthetischen Daten"""
        
        risk_results = {
            'colic_cases': len(synthetic_data),
            'age_most_common': synthetic_data['horse_age'].mode()[0] if not synthetic_data['horse_age'].empty else 0,
            'age_most_common_count': synthetic_data['horse_age'].value_counts().iloc[0] if not synthetic_data['horse_age'].empty else 0,
        }
        
        
        # Geschlechterverteilung
        risk_results['gender_counts'] = synthetic_data['horse_gender'].value_counts().to_dict()
        
        # Haltungsbedingungen
        risk_results['housing_counts'] = synthetic_data['horse_keeping'].value_counts().to_dict()
        
        # Rassebedingungen
        risk_results['breed_counts'] = synthetic_data['breed'].value_counts().to_dict()
        
        # Altersgruppen
        synthetic_data['age_group'] = pd.cut(synthetic_data['horse_age'], 
                                        bins=[0, 5, 10, 15, 20, 30], 
                                        labels=['0-5', '6-10', '11-15', '16-20', '21+'])
        risk_results['age_groups'] = synthetic_data['age_group'].value_counts().to_dict()
        
        # Kombinierte Risikofaktoren (Beispiel)
        synthetic_data['risk_combination'] = synthetic_data['horse_gender'] + "_" + synthetic_data['horse_keeping']
        risk_results['combined_risks'] = synthetic_data['risk_combination'].value_counts().head(10).to_dict()
        
        return risk_results

    def calculate_risk_statistics_for_synthetic_data(self, risk_results):
        total_colic = risk_results['colic_cases']
        
        # Basis-Statistiken 
        stats = {
            'age_most_common': {
                'measures of central tendency': risk_results['age_most_common'],
                'percentage': (risk_results['age_most_common_count'] / total_colic) * 100 if total_colic else 0,
                'description': 'Altersbedingte Risikofaktoren',
                'count': risk_results['age_most_common_count']
            },
            'temp_max_most_common': {
                'measures of central tendency': risk_results['temp_max_mean'],
                'percentage': (risk_results['temp_max_mean'] / total_colic) * 100 if total_colic else 0,
                'description': 'Risikofaktoren bei hohen Temperaturen',
                'count': risk_results.get('temp_max_count', 0)
            },
            'temp_min_most_common': {
                'measures of central tendency': risk_results['temp_min_mean'],
                'percentage': (risk_results['temp_min_mean'] / total_colic) * 100 if total_colic else 0,
                'description': 'Risikofaktoren bei niedrigen Temperaturen',
                'count': risk_results.get('temp_min_count', 0)
            },
            'precipitation_most_common': {
                'measures of central tendency': risk_results['precipitation_mean'],
                'percentage': (risk_results['precipitation_mean'] / total_colic) * 100 if total_colic else 0,
                'description': 'Risikofaktoren bei Niederschlag',
                'count': risk_results.get('precipitation_count', 0)
            },
            'feed_counts': {
                'measures of central tendency': risk_results['most_common_feed'],
                'percentage': (risk_results['feed_counts'] / total_colic) * 100 if total_colic else 0,
                'description': 'Fütterungsbedingte Risikofaktoren',
                'count': risk_results['feed_counts']
            },
        }
        
        
        # Rasse-spezifische Risiken
        if 'breed_counts' in risk_results:
            breed_stats = {}
            for breed, count in risk_results['breed_counts'].items():
                breed_stats[breed] = {
                    'count': count,
                    'percentage': (count / total_colic) * 100,
                    'description': f'Risikofaktoren bei {breed}'
                }
            stats['breed_analysis'] = breed_stats
            
        # Geschlechter-spezifische Risiken
        if 'gender_counts' in risk_results:
            gender_stats = {}
            for gender, count in risk_results['gender_counts'].items():
                gender_stats[gender] = {
                    'count': count,
                    'percentage': (count / total_colic) * 100,
                    'description': f'Risikofaktoren bei {gender}'
                }
            stats['gender_analysis'] = gender_stats
        
        # Haltungsbedingungen
        if 'housing_counts' in risk_results:
            housing_stats = {}
            for housing, count in risk_results['housing_counts'].items():
                housing_stats[housing] = {
                    'count': count,
                    'percentage': (count / total_colic) * 100,
                    'description': f'Risikofaktoren bei {housing} Haltung'
                }
            stats['housing_analysis'] = housing_stats
        
        # Kombinierte Risikofaktoren
        if 'combined_risks' in risk_results:
            combined_stats = {}
            for combination, count in risk_results['combined_risks'].items():
                combined_stats[combination] = {
                    'count': count,
                    'percentage': (count / total_colic) * 100,
                    'description': f'Kombinierte Risikofaktoren: {combination}'
                }
            stats['combined_risk_analysis'] = combined_stats
        
        # Altersgruppen-Analyse
        if 'age_groups' in risk_results:
            age_group_stats = {}
            for age_group, count in risk_results['age_groups'].items():
                age_group_stats[age_group] = {
                    'count': count,
                    'percentage': (count / total_colic) * 100,
                    'description': f'Risikofaktoren in Altersgruppe {age_group}'
                }
            stats['age_group_analysis'] = age_group_stats
        
        # Saisonale Muster
        if 'seasonal_counts' in risk_results:
            seasonal_stats = {}
            for season, count in risk_results['seasonal_counts'].items():
                seasonal_stats[season] = {
                    'count': count,
                    'percentage': (count / total_colic) * 100,
                    'description': f'Risikofaktoren in {season}'
                }
            stats['seasonal_analysis'] = seasonal_stats
        
        # Zusammenfassende Metriken
        stats['summary_metrics'] = {
            'total_colic_cases': total_colic,
            'most_common_risk_factor': max(
                [(key, val['count']) for key, val in stats.items() if 'count' in val],
                key=lambda x: x[1],
                default=('N/A', 0)
            )[0] if total_colic else 'N/A',
            'risk_diversity': len([key for key in stats if 'count' in stats[key]]),
            'high_risk_threshold': total_colic * 0.1  # 10% als High-Risk-Schwelle
        }
        
        return stats

    def print_risk_summary(self, stats):
        print("=" * 50)
        print("RISIKOFAKTOREN ANALYSE - ZUSAMMENFASSUNG")
        print("=" * 50)
        print(f"Gesamtanzahl Kolikfälle: {stats['summary_metrics']['total_colic_cases']}")
        print(f"Häufigster Risikofaktor: {stats['summary_metrics']['most_common_risk_factor']}")
        
        if 'gender_analysis' in stats:
            most_common_gender = max(stats['gender_analysis'].items(), 
                                key=lambda x: x[1]['percentage'])
            print(f"Häufigstes Geschlecht: {most_common_gender[0]} ({most_common_gender[1]['percentage']:.1f}%)")
        
        print("\nTop Risikofaktoren:")
        for key in ['age_most_common', 'feed_counts']:
            if key in stats:
                print(f"- {stats[key]['description']}: {stats[key]['percentage']:.1f}%")
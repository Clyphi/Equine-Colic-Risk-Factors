#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: visualizer
Author: Claudia Leins
Module: visualization
Description: Modular Visualizer for Risk Factors, Keywords, Demographics, and ML Results
"""

import matplotlib.pyplot as plt

class FeatureVisualizer:
    # =======================
    # Risikofaktoren-Plots
    # =======================
    @staticmethod
    def plot_risk_factors(risk_stats, output_path=None, title="Risikofaktoren in Kolik-Posts"):
        if not risk_stats:
            print("Keine Risikofaktoren zum Plotten vorhanden.")
            return
        
        descriptions = [data['description'] for data in risk_stats.values()]
        counts = [data['count'] for data in risk_stats.values()]
        percentages = [data['percentage'] for data in risk_stats.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(descriptions, counts, color=['lightblue', 'lightcoral'])
        
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom')

        plt.xlabel('Risikofaktor-Kategorie')
        plt.ylabel('Anzahl der Posts')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot gespeichert unter: {output_path}")
        else:
            plt.show()

    # =======================
    # Keyword-Distribution
    # =======================
    @staticmethod
    def plot_keyword_distribution(df, column, top_n=10, output_path=None, title=None):
        if df.empty or column not in df.columns:
            print(f"Spalte {column} existiert nicht oder DataFrame ist leer.")
            return

        all_keywords = df[column].dropna().str.split(',').explode()
        keyword_counts = all_keywords.value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(keyword_counts.index, keyword_counts.values, color='lightgreen')

        for bar, count in zip(bars, keyword_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., count + 0.5,
                    str(count), ha='center', va='bottom')

        plt.xlabel('Keyword')
        plt.ylabel('Anzahl der Posts')
        plt.title(title or f'Häufigste Keywords in {column}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Keyword-Distribution-Plot gespeichert unter: {output_path}")
        else:
            plt.show()

    # =======================
    # Altersverteilung
    # =======================
    @staticmethod
    def plot_age_distribution(df, age_column='horse_age', bins=10, output_path=None, title="Altersverteilung der Pferde"):
        if df.empty or age_column not in df.columns:
            print(f"Spalte {age_column} existiert nicht oder DataFrame ist leer.")
            return

        ages = df[age_column].dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ages, bins=bins, color='orange', edgecolor='black')
        plt.xlabel('Alter')
        plt.ylabel('Anzahl der Pferde')
        plt.title(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Altersverteilung-Plot gespeichert unter: {output_path}")
        else:
            plt.show()

    # =======================
    # Geschlechterverteilung
    # =======================
    @staticmethod
    def plot_gender_distribution(df, gender_column='horse_gender', output_path=None, title="Geschlechterverteilung der Pferde"):
        if df.empty or gender_column not in df.columns:
            print(f"Spalte {gender_column} existiert nicht oder DataFrame ist leer.")
            return

        gender_counts = df[gender_column].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(gender_counts.index, gender_counts.values, color=['pink', 'lightblue', 'lightgreen'])

        for bar, count in zip(bars, gender_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., count + 0.5,
                    str(count), ha='center', va='bottom')

        plt.xlabel('Geschlecht')
        plt.ylabel('Anzahl der Pferde')
        plt.title(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Geschlechterverteilung-Plot gespeichert unter: {output_path}")
        else:
            plt.show()

    # =======================
    # ML-Ergebnisse (Platzhalter)
    # =======================
    @staticmethod
    def plot_ml_results(results, output_path=None, title="ML Ergebnisse"):
        """
        Platzhalter für zukünftige ML-Visualisierungen.
        Z.B. Feature Importances, Confusion Matrix, ROC Curve
        """
        print("ML-Plot-Funktion noch nicht implementiert.")
        # Später z.B.:
        # - Barplot der Feature Importances
        # - Confusion Matrix als Heatmap
        # - ROC / Precision-Recall Kurven

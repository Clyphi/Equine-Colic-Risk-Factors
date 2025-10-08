#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: analyze_syntethic_data
Author: Claudia Leins
Description: Perform risk factor analysis of synthetic data about equine colic,
             using preprocessed data from the SQLite database.
"""

from pathlib import Path
import sys
import pandas as pd

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import internal modules
from src.statistics.risk_factor_analysis import RiskFactorAnalyzer
from src.visualization.visualizer import FeatureVisualizer
from src.database.data_loader import SyntheticDataLoader
from src.models.colic_classifier import ColicClassifier
from src.models.colic_nn import ColicNN


if __name__ == "__main__":
    # --- Load Reddit processed data from database ---
    print("📥 Loading data from database...")
    loader = SyntheticDataLoader()
    df = loader.load_data_from_db()
    
    
    if df.empty:
        print("⚠️ No data found in database. Please run reddit_pipeline.py first.")
        sys.exit(1)

    # --- Risk factor analysis ---
    print("🔍 Analyzing risk factors...")
    risk_analyzer = RiskFactorAnalyzer(loader_class=SyntheticDataLoader)
    
    # ERWEITERTE ANALYSE: Zuerst erweiterte Metriken berechnen
    print("📊 Calculating extended risk metrics...")
    extended_risk_results = risk_analyzer.calculate_extended_risk_metrics(df)
    
    # Bestehende Analyse durchführen
    risk_results, analyzed_df = risk_analyzer._analyze_synthetic(df)
    
    # Ergebnisse kombinieren
    combined_results = {**risk_results, **extended_risk_results}
    
    # Erweiterte Statistik berechnen
    risk_stats = risk_analyzer.calculate_risk_statistics_for_synthetic_data(combined_results)

    # --- Save analyzed results ---
    output_path = PROJECT_ROOT / "data" / "processed" / "analyzed_posts.csv"
    analyzed_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")

    # --- Print statistics ---
    print("\n📊 RISK FACTOR STATISTICS:")
    print(f"Total cases: {risk_results['total_cases']}")
    print(f"Colic-related cases: {risk_results['colic_cases']}")
    
    # INSTANZ VON FEATUREVISUALIZER ERSTELLEN
    print("📈 Creating visualizations...")
    visualizer = FeatureVisualizer()
    
    # Erweiterte Statistik-Ausgabe
    visualizer.print_risk_summary(risk_stats)
    
    # Detaillierte Ausgabe
    for factor, data in risk_stats.items():
        if isinstance(data, dict) and 'percentage' in data:
            print(f"{data['description']}: {data['measures of central tendency']} cases ({data['percentage']:.1f}%)")

    # --- Visualization ---
    output_dir = PROJECT_ROOT / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bestehende Visualisierungen
    visualizer.plot_age_distribution(
            df,
            age_column='horse_age',
            bins=10,
            output_path=output_dir / "risk_factors_horse_age_distribution_synth_data.png",
            title="Frequency of Risk Factors in Colic-Related Synthetic Data"
        )
    
    visualizer.plot_gender_distribution(
        df,
        gender_column='horse_gender',
        output_path=output_dir / "risk_factors_horse_gender_distribution_synth_data.png",
        title="Gender distribution of horses"
    )
    
    # NEUE ERWEITERTE VISUALISIERUNGEN
    print("📈 Generating extended visualizations...")
    visualizer.visualize_risk_statistics(risk_stats, output_path=output_dir / "colic_risk_factors_synth_data.png",
    title="Colic Risk Factors"
    )
    
    # Zusätzliche spezifische Visualisierungen
    if 'gender_analysis' in risk_stats:
        visualizer.plot_extended_gender_analysis(
            risk_stats['gender_analysis'],
            output_path=output_dir / "extended_gender_analysis.png",
            title="Extended Gender Analysis in Colic Cases"
        )
    
    if 'housing_analysis' in risk_stats:
        visualizer.plot_housing_analysis(
            risk_stats['housing_analysis'],
            output_path=output_dir / "housing_analysis.png",
            title="Housing Conditions in Colic Cases"
        )
    
    if 'age_group_analysis' in risk_stats:
        visualizer.plot_age_group_analysis(
            risk_stats['age_group_analysis'],
            output_path=output_dir / "age_group_analysis.png",
            title="Age Group Distribution in Colic Cases"
        )

    # Modell initialisieren und trainieren  
    model = ColicClassifier()
    model.train(df)

    # Feature-Importance anzeigen
    print(model.feature_importance())
    print("🎉 Extended risk factor analysis complete!")
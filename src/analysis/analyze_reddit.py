#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: analyze_reddit
Author: Claudia Leins
Description: Perform risk factor analysis of Reddit posts about equine colic,
             using preprocessed data from the SQLite database.
"""

from pathlib import Path
import sys
import pandas as pd

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import internal modules
from src.analysis.risk_analysis import RiskFactorAnalyzer
from src.visualization.visualizer import FeatureVisualizer
from src.analysis.reddit_loader import RedditDataLoader

if __name__ == "__main__":
    # --- Load data from database ---
    print("📥 Loading data from database...")
    loader = RedditDataLoader()
    df = loader.load_data_from_db()

    if df.empty:
        print("⚠️ No data found in database. Please run reddit_pipeline.py first.")
        sys.exit(1)

    # --- Risk factor analysis ---
    print("🔍 Analyzing risk factors...")
    risk_analyzer = RiskFactorAnalyzer()
    risk_results, analyzed_df = risk_analyzer.analyze_risk_factors(df)
    risk_stats = risk_analyzer.calculate_risk_statistics(risk_results)

    # --- Save analyzed results ---
    output_path = PROJECT_ROOT / "data" / "processed" / "analyzed_posts.csv"
    analyzed_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")

    # --- Print statistics ---
    print("\n📊 RISK FACTOR STATISTICS:")
    print(f"Total posts: {risk_results['total_posts']}")
    print(f"Colic-related posts: {risk_results['total_colic_posts']}")
    for factor, data in risk_stats.items():
        print(f"{data['description']}: {data['count']} posts ({data['percentage']:.1f}%)")

    # --- Visualization ---
    output_dir = PROJECT_ROOT / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    FeatureVisualizer.plot_risk_factors(
        risk_stats,
        output_path=output_dir / "risk_factors_distribution.png",
        title="Frequency of Risk Factors in Colic-Related Posts"
    )

    print("🎉 Risk factor analysis complete!")

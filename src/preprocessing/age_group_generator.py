#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: age_group_generator.py
Author: Claudia Leins
Description: Create age groups for horses based on age (usable for synthetic and real data)
"""

import pandas as pd
import numpy as np

class AgeGroupGenerator:
    """Assign age groups based on horse age (usable for synthetic and real data)"""

    def __init__(self):
        # Standardgrenzen – können später angepasst oder extern konfiguriert werden
        self.age_bins = [0, 3, 10, 18, np.inf]
        self.age_labels = ["foal", "young", "adult", "senior"]

    def add_age_group(self, df: pd.DataFrame, age_col: str = "horse_age") -> pd.DataFrame:
        """Add an 'age_group' column to the DataFrame"""
        if age_col not in df.columns:
            raise ValueError(f"Column '{age_col}' not found in DataFrame")

        df = df.copy()
        df["age_group"] = pd.cut(
            df[age_col],
            bins=self.age_bins,
            labels=self.age_labels,
            include_lowest=True,
            right=False
        )
        return df

    def get_group_statistics(self, df: pd.DataFrame):
        """Optional: Überblick über Altersverteilung und Koliktypen"""
        if "colic_keywords" in df.columns and "age_group" in df.columns:
            stats = (
                df.groupby("age_group")["colic_keywords"]
                .apply(lambda x: (x != "").sum())
                .reset_index(name="colic_cases")
            )
            stats["total"] = df.groupby("age_group").size().values
            stats["colic_rate_%"] = (stats["colic_cases"] / stats["total"] * 100).round(1)
            return stats
        else:
            return df["age_group"].value_counts(normalize=True).round(2) * 100


if __name__ == "__main__":
    # Beispiel: Test mit synthetischen Daten
    data = {
        "horse_age": [1, 4, 7, 12, 19, 25, 30],
        "colic_keywords": ["spasmodic", "", "impaction", "", "sand", "", "displacement"]
    }
    df = pd.DataFrame(data)

    generator = AgeGroupGenerator()
    df_with_groups = generator.add_age_group(df)
    print(df_with_groups)
    print("\nVerteilung:")
    print(generator.get_group_statistics(df_with_groups))

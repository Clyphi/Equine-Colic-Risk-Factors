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
            plt.close(fig) 
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
            plt.close(fig) 
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
       # plt.title(title)
       # plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 
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
       # plt.title(title)
       # plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 
            print(f"Geschlechterverteilung-Plot gespeichert unter: {output_path}")
        else:
            plt.show()


    # =======================
    # Risikofaktoren
    # Synthetische Daten
    # =======================
    def visualize_risk_statistics(self, stats, output_path=None, title=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Hauptrisikofaktoren (Balkendiagramm)
        main_factors = ['age_most_common', 'feed_counts']
        if 'temp_max_most_common' in stats:
            main_factors.append('temp_max_most_common')
        
        factors_data = [(key, stats[key]['percentage']) for key in main_factors if key in stats]
        factors_names = [stats[key]['description'] for key in main_factors if key in stats]
        factors_values = [stats[key]['percentage'] for key in main_factors if key in stats]
        
        axes[0].bar(factors_names, factors_values, color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[0].set_title('Häufigste Risikofaktoren')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Geschlechterverteilung
        if 'gender_analysis' in stats:
            genders = list(stats['gender_analysis'].keys())
            gender_percentages = [stats['gender_analysis'][g]['percentage'] for g in genders]
            axes[1].pie(gender_percentages, labels=genders, autopct='%1.1f%%')
            axes[1].set_title('Geschlechterverteilung bei Koliken')
        
        # 3. Haltungsbedingungen
        if 'housing_analysis' in stats:
            housings = list(stats['housing_analysis'].keys())
            housing_counts = [stats['housing_analysis'][h]['count'] for h in housings]
            axes[2].bar(housings, housing_counts, color='lightblue')
            axes[2].set_title('Haltungsbedingungen bei Koliken')
            axes[2].tick_params(axis='x', rotation=45)
        
        # 4. Altersgruppen
        if 'age_group_analysis' in stats:
            age_groups = list(stats['age_group_analysis'].keys())
            age_percentages = [stats['age_group_analysis'][ag]['percentage'] for ag in age_groups]
            axes[3].bar(age_groups, age_percentages, color='orange')
            axes[3].set_title('Altersgruppen bei Koliken')
            axes[3].tick_params(axis='x', rotation=45)
        
        # 5. Pferderassen
        if 'breed_analysis' in stats:
            breeds = list(stats['breed_analysis'].keys())
            breed_counts = [stats['breed_analysis'][s]['count'] for s in breeds]
            axes[4].plot(breeds, breed_counts, marker='o', color='green')
            axes[4].set_title('Verteilung der Pferderassen bei Koliken')
            axes[4].tick_params(axis='x', rotation=45, labelsize=6)
        
        # 6. Risiko-Heatmap (Beispiel für kombinierte Faktoren)
        if 'combined_risk_analysis' in stats and len(stats['combined_risk_analysis']) > 0:
            combinations = list(stats['combined_risk_analysis'].keys())[:5]  # Top 5
            combination_percentages = [stats['combined_risk_analysis'][c]['percentage'] for c in combinations]
            axes[5].barh(combinations, combination_percentages, color='purple')
            axes[5].set_title('Top kombinierte Risikofaktoren')
        
        plt.title(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 
            print(f"Kolik-Risikofaktoren-Plot gespeichert unter: {output_path}")
        else:
            plt.show()
            
        # Zusätzliche Zusammenfassung ausgeben
        self.print_risk_summary(stats)

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
                
                
    @staticmethod
    def plot_extended_gender_analysis(gender_stats, output_path, title):
        """Plot erweiterte Geschlechteranalyse"""
        import matplotlib.pyplot as plt
        
        genders = list(gender_stats.keys())
        percentages = [gender_stats[g]['percentage'] for g in genders]
        counts = [gender_stats[g]['count'] for g in genders]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Tortendiagramm
        ax1.pie(percentages, labels=genders, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'{title} - Verteilung')
        
        # Balkendiagramm
        bars = ax2.bar(genders, counts, color=['blue', 'pink', 'gray'])
        ax2.set_title(f'{title} - Absolute Häufigkeiten')
        ax2.set_ylabel('Anzahl der Fälle')
        
        # Werte auf den Balken anzeigen
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Extended gender analysis saved to: {output_path}")
    
    @staticmethod
    def plot_housing_analysis(housing_stats, output_path, title):
        """Plot Haltungsbedingungen Analyse"""
        import matplotlib.pyplot as plt
       # import seaborn as sns
        
        housings = list(housing_stats.keys())
        percentages = [housing_stats[h]['percentage'] for h in housings]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(housings, percentages, color='lightgreen')
        plt.title(title)
        plt.ylabel('Prozentanteil (%)')
        plt.xticks(rotation=45)
        
        # Werte auf den Balken anzeigen
        for bar, percentage, housing in zip(bars, percentages, housings):
            count = housing_stats[housing]['count']
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{percentage:.1f}%\n(n={count})', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Housing analysis saved to: {output_path}")
    
    @staticmethod
    def plot_age_group_analysis(age_group_stats, output_path, title):
        """Plot Altersgruppen Analyse"""
        import matplotlib.pyplot as plt
        
        age_groups = list(age_group_stats.keys())
        counts = [age_group_stats[ag]['count'] for ag in age_groups]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(age_groups, counts, color='orange', alpha=0.7)
        plt.title(title)
        plt.ylabel('Anzahl der Fälle')
        plt.xlabel('Altersgruppen')
        
        # Werte auf den Balken anzeigen
        for bar, count, age_group in zip(bars, counts, age_groups):
            percentage = age_group_stats[age_group]['percentage']
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Age group analysis saved to: {output_path}")
                    
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

# 🐴 Equine Colic Risk Factors (Version 2)

Dieses Projekt untersucht Risikofaktoren für Koliken bei Pferden anhand
von Reddit-Daten aus dem Subreddit **r/Horses**.\
Die Daten werden gesammelt, bereinigt, in einer SQLite-Datenbank
gespeichert und anschließend statistisch und mit Machine Learning
analysiert.

------------------------------------------------------------------------

## 📂 Projektstruktur

``` plaintext
Equine-Colic-Risk-Factors/
│
├── data/
│   ├── raw/                  # unbearbeitete Daten (Scraping)
│   ├── processed/            # bereinigte Daten für DB-Import
│   └── outputs/              # Plots, Analyseergebnisse, ML-Modelle
│
├── src/
│   ├── collection/           # Datenbeschaffung
│   ├── preprocessing/        # Datenbereinigung, Feature-Engineering
│   ├── database/             # DB-Schema & Import
│   ├── analysis/             # Statistik & erste Auswertungen
|   ├── utils   /             # Statistik & erste Auswertungen
│   └── ml/                   # Hilfsprogramme 
│       └── ml_experiments.py # Erste ML-Experimente
│
├── .env                      # Reddit API Keys (nicht ins Repo)
├── requirements.txt          # Python Abhängigkeiten
└── README.md                 # Diese Datei (Version 2)
```

------------------------------------------------------------------------

## ⚙️ Installation
### 1. Python 3.11 installieren (falls noch nicht vorhanden)

Mit Homebrew + pyenv (empfohlen):

```bash
brew install pyenv
pyenv install 3.11.8
pyenv global 3.11.8

🔹 Projektsetup
``` bash
# Repository klonen
git clone https://github.com/Clyphi/Equine-Colic-Risk-Factors.git
cd Equine-Colic-Risk-Factors

# Virtuelle Umgebung erstellen und aktivieren
python3 -m venv .venv
source .venv/bin/activate

# pip aktualisieren
pip install --upgrade pip

# Requirements installieren 
pip install -r requirements.txt

# Testen
python -c "import pandas, numpy, torch, matplotlib, seaborn; print('Alle Kernpakete OK!')"
```

------------------------------------------------------------------------

## 🚀 Nutzung

# Option 1: (Scraping von Reddit)

1.  **Daten sammeln**

    ``` bash
    python src/collection/reddit_colic_posts.py
    ```

2.  **Datenbank erstellen**

    ``` bash
    python src/database/create_db.py

    

3.    **Daten bereinigen und in die Datenbank importieren**

    ``` bash
    python src/database/reddit_pipeline.py
    ```

4.  **Analysen durchführen**

    ``` bash
    python src/analysis/analyze_reddit.py
    ```

5.  **Machine Learning (zukünftig)**

    ``` bash
    python src/ml/ml_experiments.py
    ```

# Option 2: (Synthetische Daten erstellen)

1.  **Datenbank erstellen**

    ``` bash
    python src/database/create_db.py

2.    **Synthetische Daten erstellen, reale Wetterdaten hinzufügen und in die Datenbank importieren**

    ``` bash
    python src/database/synthetic_data_pipeline.py
    ```

3.  **Analysen durchführen**

    ``` bash
    python src/analysis/analyze_data_from_db.py
    ```

4.  **Machine Learning (zukünftig)**

    ``` bash
    python src/ml/ml_experiments.py
    ```
------------------------------------------------------------------------

## 📊 Analysen

-   Deskriptive Statistik: Häufigkeit & Varianz von Risikofaktoren\
-   Zusammenhangsanalyse: Statistische Tests zwischen Faktoren und
    Kolik-Hinweisen\
-   Visualisierungen: Zeitreihen, Histogramme, Korrelationen

------------------------------------------------------------------------

## 🔮 Geplante Erweiterungen

### 🧠 Machine Learning Modelle (aktualisiert)

- **`src/models/colic_classifier.py`**  
  Enthält klassische Machine-Learning-Modelle (Random Forest, XGBoost, CatBoost) zur Klassifikation von Koliktypen basierend auf synthetischen Risikofaktoren.  

- **`src/models/colic_nn.py`**  
  Feedforward-Neural-Network zur Untersuchung nichtlinearer Zusammenhänge zwischen Futter, Haltung, Alter und Koliktyp.  

- **Zukünftig geplant:**  
  - Vergleich der Modelle mit erweiterten Feature-Sets (z. B. Wetter, Trainingsintensität)  
  - Hyperparameter-Optimierung (z. B. mit Optuna)  
  - Evaluierung mit Cross-Validation  

---

### 📊 Modellbewertung und Erklärbarkeit

- Erweiterung der Modellreports (Precision, Recall, F1-Score pro Koliktyp)  
- Visualisierung der Feature-Importances (z. B. mit SHAP oder Permutation Importance)  
- Robustheitsanalysen auf synthetischen **und**, sofern verfügbar, realen Daten  

---

### 🌦️ Zeitreihenanalyse

- Untersuchung saisonaler Muster (Wintermonate, Weidezeit, Temperaturverläufe)  
- Korrelation von Wetterdaten (Temperatur, Niederschlag) mit Koliktypen  

---

### 🕸️ Netzwerkanalyse

- Ermittlung von Beziehungen zwischen Risikofaktoren (z. B. Alter ↔ Haltung ↔ Futter)  
- Visualisierung als Graph (z. B. mit NetworkX oder Gephi-kompatiblem Export)  

---

### 🚀 Deployment

- Entwicklung eines interaktiven Dashboards (z. B. mit **Streamlit** oder **Dash**)  
  zur Visualisierung von Risikofaktoren, Modellvorhersagen und Feature-Importances  
- Optional: API-Endpunkte für Modellabfragen (`/predict`)  

---

------------------------------------------------------------------------

# Daten & Limitierungen

## Herkunft der Daten
Für diese Analyse wurden primär synthetische Datensätze verwendet, die mit `src/preprocessing/src/preprocessing/synthetic_data_pipeline.py` erzeugt wurden. Ziel war es, plausible Kausalbeziehungen zwischen Alter, Fütterung, Haltungsform und Koliktyp zu simulieren, damit Modelle und Visualisierungen entwickelt werden können.

Die Generierung folgt regelbasierten Heuristiken (Alter → Impaction, Silage+Stall → Displacement, Fresh Grass → Gas, usw.). Die Regeln sind im Generator kommentiert und bewusst konservativ gehalten.

## Wichtige Einschränkungen
- **Synthetische Daten sind kein Ersatz für Echtdaten.** Modelle, die ausschließlich an synthetischen Fällen trainiert wurden, können reale Muster nicht zuverlässig vorhersagen.
- **Bias und Vereinfachung:** Die Regeln im Generator vereinfachen komplexe klinische Zusammenhänge. Das bedeutet: gewisse Feature-Kombinationen sind überrepräsentiert oder fehlen vollständig.
- **Klassenungleichgewicht:** Einige Koliktypen (z. B. `torsion`) treten selten auf — das führt zu niedrigen Recall/F1 für diese Klassen in allen getesteten Modellen.
- **Mögliche Überanpassung:** Modelle wie CatBoost oder NN können vermeintlich gute Scores auf synthetischen Daten erzielen, ohne in realen Szenarien nützliche Vorhersagen zu liefern.

## Reproduzierbarkeit
- Seed und Generator-Parameter sind in `create_synthetic_data.py` einstellbar. Bitte immer denselben Seed verwenden, wenn Ergebnisse reproduziert werden sollen.
- Synthetische Daten sind mit dem Feld `is_synthetic=True` gekennzeichnet; beim Training auf Mixed-Datasets sollte diese Spalte berücksichtigt werden (z. B. für Domain-Split oder als Feature).

## Beobachtete Ergebnisse (Kurzfassung)
- Mehrere Modelle (CatBoost, XGBoost, RandomForest, Feed-Forward NN) wurden evaluiert.
- Mit ~2k synthetischen Fällen ergibt sich eine Test-Accuracy von ~0.37–0.40; gewichtete Feature-Importances zeigen `feed_main`, `horse_keeping` und `age_group` als stärkste Prädiktoren.
- Fazit: Mit den aktuellen synthetischen Daten sind die Klassen nicht hinreichend trennbar — für robuste Vorhersagen werden entweder realere, reichhaltigere Daten oder gezielt designte synthetische Datensätze (mehr Variation und weniger Rauschen) benötigt.
------------------------------------------------------------------------

## 👩‍💻 Autor

Claudia Leins\
Projekt *Equine Colic Risk Factors*


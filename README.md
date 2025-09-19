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

``` bash
git clone https://github.com/Clyphi/Equine-Colic-Risk-Factors.git
cd Equine-Colic-Risk-Factors
pip install -r requirements.txt
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

-   **Machine Learning Modelle**:
    -   `src/ml/ml_experiments.py` für erste Tests (z. B. Logistic
        Regression, Random Forest).\
    -   Später Aufteilung in Trainings-, Validierungs- und Testsets.\
-   **Zeitreihenanalyse**: Untersuchung saisonaler Muster (z. B. Wetter,
    Futterwechsel).\
-   **Netzwerkanalyse**: Beziehungen zwischen Risikofaktoren.\
-   **Deployment**: Interaktives Dashboard (Flask/Streamlit).

------------------------------------------------------------------------

## 👩‍💻 Autor

Claudia Leins\
Projekt *Equine Colic Risk Factors*


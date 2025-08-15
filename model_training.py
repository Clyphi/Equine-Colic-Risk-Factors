#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: model_training
Author: Claudia Leins
Description: Sentiment analysis
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import nltk


# def sentiment_train(filepath):
# NLTK-Daten herunterladen (nur beim ersten Mal)
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Daten laden (ersetzt deinen CSV-Lade-Teil)
filepath = "/Users/claudialeins/github/Equine-Colic-Risk-Factors/reddit_colic_posts.csv"
df = pd.read_csv(filepath)  # "reddit_colic_posts.csv"

# 2. Erweiterte Textbereinigung
def hybrid_preprocessor(text):
    # Deine Original-Bereinigung
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Zusätzliche Optimierungen
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Nur Buchstaben/Leerzeichen
    
    # Lemmatisierung + Stopword-Entfernung (aber behalte veterinärrelevante Begriffe)
    custom_stopwords = set(stopwords.words('english')) - {'colic', 'horse', 'pain', 'vet'}
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in custom_stopwords]
    
    return ' '.join(words)

# Anwendung auf die Texte
df['clean_text'] = df['text'].apply(hybrid_preprocessor)

# 3. TF-IDF mit Bigrams (ersetzt dein bag_of_words)
tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)  # Unigrams + Bigrams
)
features = tfidf.fit_transform(df['clean_text'])

# 4. Modelltraining (angepasst an deinen Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, 
    df['sentiment'], 
    test_size=0.2
)

model = RandomForestClassifier(
    n_estimators=150,
    class_weight='balanced',  # Wichtig bei unausgeglichenen Klassen
    max_depth=10
)
model.fit(X_train, y_train)
# Rückgabe von Modell, Vokabular und MQF
   # return model, vokabular, mqf
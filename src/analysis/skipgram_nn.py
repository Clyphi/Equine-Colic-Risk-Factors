#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: skipgram_nn
Author: Claudia Leins
Description: Implementation eines SkipGram-Neuronalen Netzes zur Generierung von Wort-Embeddings. 
Das Skript umfasst Datenvorbereitung, SkipGram-Paar-Erzeugung, Modelltraining und Beispielausgaben 
für Schlüsselwörter aus Reddit-Diskussionen über Kolik und Wetter.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import pandas as pd
from pathlib import Path
import os

# Projektpfade konfigurieren
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # 3x parent, um aus src/analysis/ zum Root zu gelangen
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.file_loader import load_keywords

# ==============================
# 1. Data Loader Klasse
# ==============================
class RedditDataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        """Lädt Reddit-Daten und kombiniert Titel und Text zu einer Spalte."""
        df = pd.read_csv(self.csv_path)
        df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
        return df

# ==============================
# 2. Skipgram Generator Klasse
# ==============================
class SkipGramGenerator:
    def __init__(self, keywords, window_size=15):
        self.keywords = {kw.lower() for kw in keywords if isinstance(kw, str)}
        self.window_size = window_size

    def generate(self, words):
        skipgrams = []
        n = len(words)
        for i in range(n):
            for j in range(i + 1, min(i + self.window_size + 1, n)):
                if words[i].lower() in self.keywords or words[j].lower() in self.keywords:
                    skipgrams.append((words[i], words[j]))
                    skipgrams.append((words[j], words[i]))
        return skipgrams

# ==============================
# 3. Keyword Manager Klasse
# ==============================
class KeywordManager:
    def __init__(self, colic_path, weather_path):
        self.colic_path = colic_path
        self.weather_path = weather_path

    def load(self):
        keywords = load_keywords(self.colic_path)
        keywords.extend(load_keywords(self.weather_path))
        return keywords

# ==============================
# 4. Dataset-Klasse
# ==============================
class SkipGramDataset(Dataset):
    def __init__(self, pairs, word_to_idx, vocab_size):
        self.pairs = pairs
        self.word_to_idx = word_to_idx
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_word, target_word = self.pairs[idx]
        input_idx = self.word_to_idx[input_word]
        target_idx = self.word_to_idx[target_word]
        return torch.tensor(input_idx), torch.tensor(target_idx)

# ==============================
# 5. Modell-Definition
# ==============================
class Word2VecNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Word2VecNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        return self.network(x)

# ==============================
# 6. Hauptprogramm
# ==============================
if __name__ == "__main__":
    # Pfade zu Ein- und Ausgabedateien
    CSV_PATH = PROJECT_ROOT / "data" / "raw" / "reddit_colic_posts.csv"
    COLIC_PATH = PROJECT_ROOT / "data" / "keywords" / "colic_keywords.txt"
    WEATHER_PATH = PROJECT_ROOT / "data" / "keywords" / "weather_keywords.txt"

    # Keywords laden
    km = KeywordManager(COLIC_PATH, WEATHER_PATH)
    keywords = km.load()
    print("Geladene Keywords:", keywords)

    # Daten laden
    loader = RedditDataLoader(CSV_PATH)
    df = loader.load()

    # Vokabular
    words_list = df['text'].str.lower().str.split(expand=True).stack().tolist()
    vocab = sorted(set(words_list))
    vocab_size = len(vocab)

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Skipgrams
    sg = SkipGramGenerator(keywords, window_size=15)
    df['skipgrams'] = df['text'].apply(lambda x: sg.generate(x.lower().split()) if isinstance(x, str) else [])
    all_skipgrams = [pair for sublist in df['skipgrams'] for pair in sublist]

    # ==============================
    # Training
    # ==============================
    hidden_size = 50
    learning_rate = 0.1
    num_epochs = 50

    dataset = SkipGramDataset(all_skipgrams, word_to_idx, vocab_size)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Word2VecNN(vocab_size, hidden_size).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for input_idx, target_idx in train_loader:
            # One-hot Encode Input
            input_one_hot = torch.zeros(input_idx.size(0), vocab_size)
            input_one_hot[torch.arange(input_idx.size(0)), input_idx] = 1
            input_one_hot = input_one_hot.to(device)
            target_idx = target_idx.to(device)

            outputs = model(input_one_hot)
            loss = loss_function(outputs, target_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    # ==============================
    # Beispielausgabe
    # ==============================
    with torch.no_grad():
        test_word = "rain"
        test_idx = word_to_idx.get(test_word)
        if test_idx is None:
            print(f"'{test_word}' ist nicht im Vokabular!")
        else:    
            one_hot = torch.zeros(1, vocab_size)
            one_hot[0, test_idx] = 1
            prediction = model(one_hot.to(device))
            predicted_idx = torch.argmax(prediction, dim=1).item()
            print(f"Eingabewort: '{test_word}' -> Vorhergesagtes Kontextwort: '{idx_to_word[predicted_idx]}'")

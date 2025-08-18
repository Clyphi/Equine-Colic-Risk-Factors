#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: skipgram_nn
Author: Claudia Leins
Description: Implementation eines SkipGram-Neuronalen Netzes zur Generierung von Wort-Embeddings. 
Das Skript umfasst Datenvorbereitung, SkipGram-Paar-Erzeugung, Modelltraining und Beispielausgaben 
für Schlüsselwörter aus Reddit-Diskussionen über Kolik und Wetter.
Second version: new model Word2VecSkipGram, negative Sampling, opitimizer from SGD to Adam, embedding_dim = 100"
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.file_loader import load_keywords

# ==============================
# 1. Data Loader Klasse
# ==============================
class RedditDataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

    def load_data(self):
        """Lädt Reddit-Daten und kombiniert Titel und Text zu einer Spalte."""
        self.df = pd.read_csv(self.csv_path)
        self.df['full_text'] = self.df['title'].fillna('') + " " + self.df['text'].fillna('')
        return self.df

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
# 3. Keyword Loader Klasse
# ==============================
class KeywordManager:
    def __init__(self, colic_path, weather_path):
        self.colic_path = colic_path
        self.weather_path = weather_path
        self.keywords = []

    def load_keywords(self):
        self.keywords = load_keywords(self.colic_path)
        self.keywords.extend(load_keywords(self.weather_path))
        return self.keywords

# ==============================
# Dataset-Klasse
# ==============================
class SkipGramDataset(Dataset):
    def __init__(self, pairs, word_to_idx):
        self.pairs = pairs
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_word, target_word = self.pairs[idx]
        input_idx = self.word_to_idx[input_word]
        target_idx = self.word_to_idx[target_word]
        return torch.tensor(input_idx), torch.tensor(target_idx)

# ==============================
# 4. Modell mit Embeddings + Negative Sampling Loss
# ==============================
class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecSkipGram, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_words, target_words, negative_words):
        # Eingabe-Vektoren
        input_vectors = self.in_embeddings(input_words)  # (batch, dim)
        target_vectors = self.out_embeddings(target_words)  # (batch, dim)
        neg_vectors = self.out_embeddings(negative_words)  # (batch, neg_samples, dim)

        # Positiver Score
        pos_score = torch.sum(input_vectors * target_vectors, dim=1)  # (batch,)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        # Negativer Score
        neg_score = torch.bmm(neg_vectors, input_vectors.unsqueeze(2)).squeeze()  # (batch, neg_samples)
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(1)  # (batch,)

        # Gesamtverlust = -(positive + negative)
        loss = -(pos_loss + neg_loss).mean()
        return loss

# ==============================
# 5. Training
# ==============================
if __name__ == "__main__":
    # Pfade
    CSV_PATH = PROJECT_ROOT / "data" / "raw" / "reddit_colic_posts.csv"
    COLIC_KEYWORDS_PATH = PROJECT_ROOT / "data" / "keywords" / "colic_keywords.txt"
    WEATHER_KEYWORDS_PATH = PROJECT_ROOT / "data" / "keywords" / "weather_keywords.txt"

    # Keywords laden
    km = KeywordManager(COLIC_KEYWORDS_PATH, WEATHER_KEYWORDS_PATH)
    keywords = km.load_keywords()

    # Daten laden
    data_loader = RedditDataLoader(CSV_PATH)
    df = data_loader.load_data()

    # Vokabular
    words_list = df['text'].str.lower().str.split(expand=True).stack().tolist()
    vocab = sorted(set(words_list))
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Skipgrams generieren
    sg_gen = SkipGramGenerator(keywords, window_size=15)
    df['skipgrams'] = df['text'].apply(lambda x: sg_gen.generate(x.lower().split()) if isinstance(x, str) else [])
    all_skipgrams = [pair for sublist in df['skipgrams'] for pair in sublist]

    print(f"Anzahl Skipgram-Pairs: {len(all_skipgrams)}")

    # Dataset & Loader
    dataset = SkipGramDataset(all_skipgrams, word_to_idx)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Modell
    embedding_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Word2VecSkipGram(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    neg_samples = 5

    for epoch in range(num_epochs):
        total_loss = 0
        for input_idx, target_idx in train_loader:
            input_idx, target_idx = input_idx.to(device), target_idx.to(device)

            # Negative Samples zufällig ziehen
            neg_idx = torch.randint(0, vocab_size, (input_idx.size(0), neg_samples)).to(device)

            loss = model(input_idx, target_idx, neg_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    # ==============================
    # 6. Beispielausgabe
    # ==============================
    with torch.no_grad():
        test_word = "rain"
        test_idx = word_to_idx.get(test_word)
        if test_idx is None:
            print(f"'{test_word}' ist nicht im Vokabular!")
        else:
            vector = model.in_embeddings(torch.tensor([test_idx]).to(device))
            print(f"Embedding für '{test_word}': {vector.cpu().numpy()}")



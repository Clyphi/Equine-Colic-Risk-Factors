# Author: Claudia Leins
# Erstellungsdatum: 2025-08-12
# Beschreibung: Skip-gram-Neuronales-Netz
# -----------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================
# 1. Text und Vokabular
# ==============================
text = "Die Katze jagt die Maus."
words_list = [word.lower().strip(".,!?") for word in text.split()]
vocab = sorted(set(words_list))
vocab_size = len(vocab)

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# ==============================
# 2. Skip-gram-Paare generieren
# ==============================
context_size = 2

def skipgram_pairs(words_list, context_size):
    skipgram_pairs_list = []
    for index, target_word in enumerate(words_list):
        start = max(0, index - context_size)
        end = min(len(words_list), index + context_size + 1)
        for j in range(start, end):
            if j != index:
                skipgram_pairs_list.append((target_word, words_list[j]))
    return skipgram_pairs_list

pairs = skipgram_pairs(words_list, context_size)

# ==============================
# 3. Dataset-Klasse
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
# 4. Modell-Definition
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
# 5. Training
# ==============================
# Hyperparameter
hidden_size = 8
learning_rate = 0.01
num_epochs = 50

dataset = SkipGramDataset(pairs, word_to_idx, vocab_size)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

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
# 6. Beispielausgabe
# ==============================
with torch.no_grad():
    test_word = "katze"
    test_idx = word_to_idx[test_word]
    one_hot = torch.zeros(1, vocab_size)
    one_hot[0, test_idx] = 1
    prediction = model(one_hot.to(device))
    predicted_idx = torch.argmax(prediction, dim=1).item()
    print(f"Eingabewort: '{test_word}' -> Vorhergesagtes Kontextwort: '{idx_to_word[predicted_idx]}'")



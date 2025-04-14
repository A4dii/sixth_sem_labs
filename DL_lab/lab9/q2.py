#Q2
import glob
import os
import string
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def readLines(filename):
    with open(filename, encoding='utf-8') as f:
        return [line.strip() for line in f.read().strip().split('\n') if line.strip()]

data_path = "/content/drive/MyDrive/data/names"
category_lines = {}
all_categories = []

for filename in glob.glob(os.path.join(data_path, '*.txt')):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
if n_categories == 0:
    raise RuntimeError("No language files found in data/names. Check your data path.")

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters, device=device)
    index = letterToIndex(letter)
    if index != -1:
        tensor[0][index] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters, device=device)
    for li, letter in enumerate(line):
        index = letterToIndex(letter)
        if index != -1:
            tensor[li][0][index] = 1
    return tensor

def randomTrainingExample():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long, device=device)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq):
        # input_seq: [seq_len, 1, input_size]
        hidden = self.initHidden()
        cell = self.initHidden()
        output, (hidden, cell) = self.lstm(input_seq, (hidden, cell))
        output = self.fc(hidden[-1])  # Take output from the last LSTM layer
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

n_hidden = 128
learning_rate = 0.005

rnn = LSTMClassifier(n_letters, n_hidden, n_categories).to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

n_iters = 100000
print_every = 5000
all_losses = []
current_loss = 0
start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    hidden = rnn.initHidden()
    rnn.zero_grad()

    output = rnn(line_tensor)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    current_loss += loss.item()

    if iter % print_every == 0:
        guess, _ = categoryFromOutput(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f'{iter} {timeSince(start)} Loss: {loss.item():.4f}  Name: {line} / Predicted: {guess} {correct}')
        all_losses.append(current_loss / print_every)
        current_loss = 0

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    output = rnn(line_tensor)
    return output

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.cpu().numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.title("Confusion Matrix")
plt.show()

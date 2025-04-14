import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample text data (you can replace this with any text)
data = "hello pytorch, this is a simple example of next character prediction using RNN."

# Build the vocabulary
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("Vocabulary:", chars)

# Create mapping from char to index and vice versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
hidden_size = 128
num_layers = 1
learning_rate = 0.003
seq_length = 10  # length of input sequence
num_epochs = 300

# Prepare the dataset: Convert all characters to indices
data_idx = [char_to_idx[ch] for ch in data]

# Create input and target sequences:
# Input: a sequence of characters; Target: next character for each position
def create_sequences(data_idx, seq_length):
    inputs = []
    targets = []
    for i in range(len(data_idx) - seq_length):
        inputs.append(data_idx[i:i+seq_length])
        targets.append(data_idx[i+1:i+seq_length+1])
    return np.array(inputs), np.array(targets)

inputs, targets = create_sequences(data_idx, seq_length)
inputs = torch.LongTensor(inputs)
targets = torch.LongTensor(targets)

# Define the RNN model for character prediction
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer converts character indices to embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # RNN layer (can also try nn.LSTM or nn.GRU)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer that maps hidden state to vocab distribution
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, hidden_size]
        out, hidden = self.rnn(embedded, hidden)  # out: [batch_size, seq_length, hidden_size]
        out = self.fc(out)  # [batch_size, seq_length, vocab_size]
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Initialize model, loss function and optimizer
model = CharRNN(vocab_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
num_batches = inputs.size(0)
for epoch in range(num_epochs):
    epoch_loss = 0
    hidden = model.init_hidden(batch_size=inputs.size(0))
    # Zero gradients
    optimizer.zero_grad()
    # Forward pass: outputs shape is [batch_size, seq_length, vocab_size]
    outputs, hidden = model(inputs, hidden)

    # Reshape outputs and targets for computing loss
    outputs = outputs.reshape(-1, vocab_size)  # [(batch_size*seq_length), vocab_size]
    targets_reshaped = targets.reshape(-1)       # [(batch_size*seq_length)]

    loss = criterion(outputs, targets_reshaped)
    loss.backward()
    optimizer.step()

    epoch_loss = loss.item()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Function to generate text using the trained model
def generate_text(model, start_text, predict_len=100):
    model.eval()
    generated = start_text
    input_seq = torch.LongTensor([char_to_idx[ch] for ch in start_text]).unsqueeze(0)
    hidden = model.init_hidden(batch_size=1)

    # Warm up with the start text
    with torch.no_grad():
        for i in range(len(start_text)-1):
            _, hidden = model(input_seq[:, i:i+1], hidden)

    last_char = input_seq[:, -1]
    for _ in range(predict_len):
        output, hidden = model(last_char.unsqueeze(1), hidden)
        # Get probabilities by applying softmax
        prob = nn.functional.softmax(output.squeeze(), dim=0).data.cpu().numpy()
        # Sample from the distribution
        char_idx = np.random.choice(range(vocab_size), p=prob)
        generated += idx_to_char[char_idx]
        last_char = torch.LongTensor([char_idx])

    return generated

# Generate text
start_text = "hello "
generated_text = generate_text(model, start_text, predict_len=200)
print("\nGenerated Text:\n", generated_text)

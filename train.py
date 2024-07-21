import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import ByteLevelBPETokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import os

folder_name = "tokenizer"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

df = pd.read_csv('dataset.csv')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_inputs = train_df['input'].tolist()
train_targets = train_df['target'].tolist()
val_inputs = val_df['input'].tolist()
val_targets = val_df['target'].tolist()

unique_targets = list(set(train_targets + val_targets))
target_to_idx = {t: idx for idx, t in enumerate(unique_targets)}
idx_to_target = {idx: t for t, idx in target_to_idx.items()}

train_targets = [target_to_idx[t] for t in train_targets]
val_targets = [target_to_idx[t] for t in val_targets]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(train_inputs, vocab_size=30522, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

with open('target_to_idx.json', 'w') as f:
    json.dump(target_to_idx, f)

with open('idx_to_target.json', 'w') as f:
    json.dump(idx_to_target, f)

tokenizer.save_model("tokenizer")

def tokenize_data(data, targets):
    tokenized_inputs = [tokenizer.encode(input_text).ids for input_text in data]
    return tokenized_inputs, targets

train_tokenized, train_targets = tokenize_data(train_inputs, train_targets)
val_tokenized, val_targets = tokenize_data(val_inputs, val_targets)

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx], dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.token_to_id('<pad>'))
    return inputs_padded, torch.tensor(targets, dtype=torch.long)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, forward_expansion, max_len, num_classes):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=num_heads, 
                dim_feedforward=forward_expansion
            )
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out)

        out = out.mean(dim=1)
        out = self.fc_out(out)
        return self.softmax(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = tokenizer.get_vocab_size()
EMBED_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FORWARD_EXPANSION = 512
MAX_LEN = 512
NUM_CLASSES = len(unique_targets)

model = TransformerModel(
    vocab_size=VOCAB_SIZE, 
    embed_size=EMBED_SIZE, 
    num_heads=NUM_HEADS, 
    num_layers=NUM_LAYERS, 
    forward_expansion=FORWARD_EXPANSION, 
    max_len=MAX_LEN, 
    num_classes=NUM_CLASSES
).to(device)

train_dataset = CustomDataset(train_tokenized, train_targets)
val_dataset = CustomDataset(val_tokenized, val_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}, Val Loss: {val_loss}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_transformer_model.pth')

model.load_state_dict(torch.load('best_transformer_model.pth'))

model.eval()
val_loss = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

val_loss /= len(val_loader)
print(f'Final Validation Loss: {val_loss}')










'''
i have a dataset.csv with binary values like this example were each line contain one target and one input:

target,input
1000101111110000000111111000010000000100011010001101000010111111010011100111100000110110010000001111111001010100011110001110100111001011000011000100111011111110100001011100001101000101000001010100101100000101011110000010011111111000011010101001001001100000,00000000101110111101010011111110001111101001010110001101111010011000110001001011000001101101101101100101010011001111011001001110000111101110111101100010101010001100110111101111101000101011010101000000
1100110010110111110111101001101101011110111100010001111111010011110010110111001001101110111100001000000010100001110000101101100001011110001111111010001001100010000100111111100101001100110111101100111000100100001000000110001101111010011110010100000100011111,00000000000001111011110011000010010100010101010010011101110100001111011000011000011101001000010111011011000100101000000111000010000100000000000110001100011010101010010101000101001100010100010011111111
...

using pytorch create a transformer from scratch to train on the dataset to predict the target based on the input.
train for 10 epochs and save the model with best performance.
remember not all values have the same lenght.
'''
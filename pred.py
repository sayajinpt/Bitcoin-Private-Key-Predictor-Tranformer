import torch
from tokenizers import ByteLevelBPETokenizer
import torch.nn as nn
import json

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

with open('target_to_idx.json', 'r') as f:
    target_to_idx = json.load(f)

with open('idx_to_target.json', 'r') as f:
    idx_to_target = json.load(f)

tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename="tokenizer/vocab.json", 
    merges_filename="tokenizer/merges.txt"
)

VOCAB_SIZE = tokenizer.get_vocab_size()
EMBED_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FORWARD_EXPANSION = 512
MAX_LEN = 512
NUM_CLASSES = len(target_to_idx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(
    vocab_size=VOCAB_SIZE, 
    embed_size=EMBED_SIZE, 
    num_heads=NUM_HEADS, 
    num_layers=NUM_LAYERS, 
    forward_expansion=FORWARD_EXPANSION, 
    max_len=MAX_LEN, 
    num_classes=NUM_CLASSES
).to(device)
model.load_state_dict(torch.load('best_transformer_model.pth', map_location=device))
model.eval()

def preprocess_input(text):
    encoded = tokenizer.encode(text).ids
    return torch.tensor(encoded).unsqueeze(0).to(device)

def predict(text):
    input_tensor = preprocess_input(text)
    with torch.no_grad():
        output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return idx_to_target[str(prediction)]

text = input("Enter text for prediction: ")
prediction = predict(text)
print(f"Predicted target: {prediction}")

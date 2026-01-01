import random
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 8
EMB_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 100
TRAIN_SIZE = 5000
TEST_SIZE = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PAD = "_"
SOS = "<"
EOS = ">"

VOCAB = string.ascii_lowercase
ALL_CHARS = PAD + SOS + EOS + VOCAB

char2idx = {c: i for i, c in enumerate(ALL_CHARS)}
idx2char = {i: c for c, i in char2idx.items()}
VOCAB_SIZE = len(ALL_CHARS)

PAD_IDX = char2idx[PAD]
SOS_IDX = char2idx[SOS]
EOS_IDX = char2idx[EOS]


def generate_sample():
    original = "".join(random.choice(VOCAB) for _ in range(SEQ_LEN))
    remove_idx = random.randint(0, SEQ_LEN - 1)
    corrupted = original[:remove_idx] + original[remove_idx + 1 :]
    return corrupted, original


class SeqDataset(Dataset):
    def __init__(self, size):
        self.data = [generate_sample() for _ in range(size)]

    def encode(self, text):
        return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, out = self.data[idx]
        x = self.encode(inp)
        y = self.encode(out)

        y_in = torch.cat([torch.tensor([SOS_IDX]), y])
        y_out = torch.cat([y, torch.tensor([EOS_IDX])])

        return x, y_in, y_out


def collate_fn(batch):
    xs, yin, yout = zip(*batch)

    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    yin = nn.utils.rnn.pad_sequence(yin, batch_first=True, padding_value=PAD_IDX)
    yout = nn.utils.rnn.pad_sequence(yout, batch_first=True, padding_value=PAD_IDX)

    return xs, yin, yout


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN_DIM, batch_first=True)

    def forward(self, x):
        x = self.emb(x)
        outputs, (h, c) = self.lstm(x)
        return outputs, h, c


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_outputs):

        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)

        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(EMB_DIM + HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.attn = Attention()
        self.fc = nn.Linear(HIDDEN_DIM * 2, VOCAB_SIZE)

    def forward(self, y, h, c, encoder_outputs):
        embeddings = self.emb(y)
        outputs = []

        for t in range(embeddings.size(1)):
            emb_t = embeddings[:, t : t + 1, :]
            context = self.attn(h[-1], encoder_outputs)
            context = context.unsqueeze(1)

            lstm_input = torch.cat([emb_t, context], dim=2)
            out, (h, c) = self.lstm(lstm_input, (h, c))

            out = out.squeeze(1)
            combined = torch.cat([out, context.squeeze(1)], dim=1)
            outputs.append(self.fc(combined).unsqueeze(1))

        return torch.cat(outputs, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, y_in):
        enc_out, h, c = self.encoder(x)
        return self.decoder(y_in, h, c, enc_out)


train_ds = SeqDataset(TRAIN_SIZE)
test_ds = SeqDataset(TEST_SIZE)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

model = Seq2Seq().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y_in, y_out in train_loader:
        x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)

        optimizer.zero_grad()
        logits = model(x, y_in)

        loss = criterion(
            logits.view(-1, VOCAB_SIZE),
            y_out.view(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")


def decode(t):
    return "".join(idx2char[int(i)] for i in t if i not in [PAD_IDX, SOS_IDX, EOS_IDX])


print("\n=== EWALUACJA (ATTENTION) ===\n")

model.eval()
with torch.no_grad():
    for i in range(5):
        x, y_in, y_out = test_ds[i]
        x = x.unsqueeze(0).to(device)

        enc_out, h, c = model.encoder(x)

        inp = torch.tensor([[SOS_IDX]], device=device)
        pred = []

        for _ in range(SEQ_LEN + 1):
            logits = model.decoder(inp, h, c, enc_out)
            token = logits[:, -1].argmax(dim=1).item()
            if token == EOS_IDX:
                break
            pred.append(token)
            inp = torch.cat([inp, torch.tensor([[token]], device=device)], dim=1)

        print("Wejście (zakłócone):", decode(x.squeeze()))
        print("Oryginał:          ", decode(y_out))
        print("Predykcja:         ", decode(pred))
        print("-" * 40)

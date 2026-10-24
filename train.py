import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocess import load_data, Vocabulary, SkipGramDataset
from model import SkipGram

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def train(
    epochs=80,
    batch_size=64,
    embedding_dim=50,
    window_size=2,
    lr=0.02,
    save_path="vietnamese_word2vec.pt"
):
    # Chuẩn bị dữ liệu
    corpus = load_data()
    vocab = Vocabulary(min_count=1)
    vocab.build_vocab(corpus)

    dataset = SkipGramDataset(corpus, vocab, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Vocab: {len(vocab)} | Samples: {len(dataset)} | Dim: {embedding_dim}")

    model = SkipGram(vocab_size=len(vocab), embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for targets, contexts in dataloader:
            targets = targets.to(device)
            contexts = contexts.to(device)

            optimizer.zero_grad()
            logits = model(targets)
            loss = criterion(logits, contexts)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)

        epoch_loss = total_loss / len(dataset)
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")

    # Lưu trọng số và từ điển
    torch.save({
        "embedding_matrix": model.get_embedding_weights(),
        "word2idx": vocab.word2idx,
        "idx2word": vocab.idx2word,
        "embedding_dim": embedding_dim,
        "loss_history": loss_history
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()

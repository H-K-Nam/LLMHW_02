import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Ma trận nhúng từ đầu vào
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Ma trận nhúng ngữ cảnh đầu ra
        self.out_embeddings = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        bound = 0.5 / self.embedding_dim
        nn.init.uniform_(self.in_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.out_embeddings.weight, -bound, bound)

    def forward(self, target_ids):
        # target_ids: (batch_size,)
        embeds = self.in_embeddings(target_ids)       # (batch_size, embedding_dim)
        logits = self.out_embeddings(embeds)          # (batch_size, vocab_size)
        return logits

    def get_embedding_weights(self):
        with torch.no_grad():
            return self.in_embeddings.weight.data.cpu().numpy()

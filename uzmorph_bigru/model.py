import torch
import torch.nn as nn

class MorphBiGRU(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=128):
        super(MorphBiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Using GRU instead of LSTM
        self.gru = nn.GRU(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
    def forward(self, sentence):
        embeds = self.char_embeddings(sentence)
        gru_out, _ = self.gru(embeds)
        tag_space = self.hidden2tag(gru_out)
        return tag_space

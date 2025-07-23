import torch
import torch.nn as nn
import torch.nn.functional as F

class TestVITS(nn.Module):
    def __init__(self, vocab_size=40000, emotion_dim=1, hidden_dim=256, waveform_len=48000):
        super(TestVITS, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.emotion_proj = nn.Linear(emotion_dim, hidden_dim * 2)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, waveform_len),  # 直接预测波形
        )

    def forward(self, text, emotion):
        # text: [B, T_text]
        # emotion: [B, 1]
        embedded = self.embedding(text)                          # [B, T_text, hidden]
        _, (h_n, _) = self.text_encoder(embedded)                # [2, B, hidden]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)                  # [B, hidden*2]

        e = self.emotion_proj(emotion)                           # [B, hidden*2]
        combined = h + e                                         # [B, hidden*2]

        waveform = self.decoder(combined)                        # [B, waveform_len]
        return waveform

def build_vits_model(vocab_size=40000, emotion_dim=1, hidden_dim=256, waveform_len=48000):
    return TestVITS(
        vocab_size=vocab_size,
        emotion_dim=emotion_dim,
        hidden_dim=hidden_dim,
        waveform_len=waveform_len
    )

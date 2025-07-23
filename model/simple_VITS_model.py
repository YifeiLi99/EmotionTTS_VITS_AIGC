import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVITS(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=256, emotion_dim=1, hidden_dim=512):
        super(SimpleVITS, self).__init__()

        # 文本编码器：嵌入 + 双层 Transformer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 情感融合：简单连接再线性映射到同一维度
        self.emotion_fc = nn.Linear(emotion_dim, embed_dim)

        # 解码器：卷积上采样 + 输出波形
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)  # 输出波形
        )

    def forward(self, text, emotion):
        # text: [B, T], emotion: [B, 1]
        x = self.embedding(text)  # [B, T, D]
        x = x.permute(1, 0, 2)  # [T, B, D] 适配 Transformer
        x = self.text_encoder(x)  # [T, B, D]
        x = x.permute(1, 0, 2)  # [B, T, D]

        # 情感调制
        e = self.emotion_fc(emotion).unsqueeze(1)  # [B, 1, D]
        e = e.expand(-1, x.size(1), -1)  # [B, T, D]
        x = x + e  # 简单相加融合情绪信息

        # 解码成波形
        x = x.permute(0, 2, 1)  # [B, D, T] 适配 Conv1d
        waveform = self.decoder(x)  # [B, 1, T]
        waveform = waveform.squeeze(1)  # [B, T]

        return waveform


def build_vits_model(vocab_size=5000):
    return SimpleVITS(vocab_size=vocab_size)

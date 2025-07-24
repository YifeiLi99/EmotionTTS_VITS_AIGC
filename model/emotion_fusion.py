import torch
import torch.nn as nn

class EmotionFusion(nn.Module):
    def __init__(self, fusion_method="concat", text_dim=256, emotion_dim=64):
        super(EmotionFusion, self).__init__()
        self.fusion_method = fusion_method.lower()
        self.text_dim = text_dim
        self.emotion_dim = emotion_dim

        if self.fusion_method == "concat":
            self.output_dim = text_dim + emotion_dim

        elif self.fusion_method == "add":
            assert text_dim == emotion_dim, "Add方式要求text和emotion维度一致"
            self.output_dim = text_dim

        elif self.fusion_method == "film":
            self.output_dim = text_dim
            self.film_layer = nn.Linear(emotion_dim, text_dim * 2)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(self, text_emb, emotion_emb):
        """
        text_emb: [B, T, text_dim]
        emotion_emb: [B, emotion_dim] or [B, 1, emotion_dim]
        """
        if emotion_emb.dim() == 2:
            emotion_emb = emotion_emb.unsqueeze(1)  # [B, 1, D]

        if self.fusion_method == "concat":
            # 广播 emotion 到每个时间步
            emotion_broadcast = emotion_emb.expand(-1, text_emb.size(1), -1)
            return torch.cat([text_emb, emotion_broadcast], dim=-1)

        elif self.fusion_method == "add":
            emotion_broadcast = emotion_emb.expand(-1, text_emb.size(1), -1)
            return text_emb + emotion_broadcast

        elif self.fusion_method == "film":
            gamma_beta = self.film_layer(emotion_emb.squeeze(1))  # [B, 2 * D]
            gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各 [B, D]
            gamma = gamma.unsqueeze(1).expand_as(text_emb)
            beta = beta.unsqueeze(1).expand_as(text_emb)
            return gamma * text_emb + beta

import torch
import torch.nn as nn

class EmotionFusion(nn.Module):
    def __init__(self, fusion_method="concat", text_dim=256, emotion_input_dim=1, emotion_hidden_dim=64):
        super(EmotionFusion, self).__init__()
        self.fusion_method = fusion_method.lower()
        self.text_dim = text_dim
        self.emotion_hidden_dim = emotion_hidden_dim

        # 映射层：将原始标量 → 嵌入向量（如 [B, 1] → [B, 64]）
        self.emotion_proj = nn.Linear(emotion_input_dim, emotion_hidden_dim)

        if self.fusion_method == "concat":
            self.output_dim = int(text_dim + emotion_hidden_dim)
        elif self.fusion_method == "add":
            assert text_dim == emotion_hidden_dim, "Add方式要求text和emotion维度一致"
            self.output_dim = int(text_dim)
        elif self.fusion_method == "film":
            self.output_dim = int(text_dim)
            self.film_layer = nn.Linear(emotion_hidden_dim, text_dim * 2)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(self, text_emb, emotion_scalar):
        """
        text_emb: [B, T, text_dim]
        emotion_scalar: [B, 1] 或 [B, 1, 1]，标量情绪强度
        """
        # 如果输入是三维的，压成二维模式
        if emotion_scalar.dim() == 3:
            emotion_scalar = emotion_scalar.squeeze(2)  # 变成 [B, 1]

        # 情绪嵌入映射（统一处理）
        emotion_emb = self.emotion_proj(emotion_scalar)  # [B, hidden_dim]
        emotion_emb = emotion_emb.unsqueeze(1)  # [B, 1, hidden_dim]

        if self.fusion_method == "concat":
            emotion_broadcast = emotion_emb.expand(-1, text_emb.size(1), -1)
            return torch.cat([text_emb, emotion_broadcast], dim=-1)

        elif self.fusion_method == "add":
            emotion_broadcast = emotion_emb.expand(-1, text_emb.size(1), -1)
            return text_emb + emotion_broadcast

        elif self.fusion_method == "film":
            gamma_beta = self.film_layer(emotion_emb.squeeze(1))  # [B, 2 * D]
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1).expand_as(text_emb)
            beta = beta.unsqueeze(1).expand_as(text_emb)
            return gamma * text_emb + beta

        return 0

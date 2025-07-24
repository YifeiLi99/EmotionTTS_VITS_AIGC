import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.emotion_fusion import EmotionFusion
from config import fusion_method
from model.flow_affine import ResidualCouplingBlock
from model.posterior_encoder import PosteriorEncoder


# 位置编码模块
class PositionalEncoding(nn.Module):
    # d_model：每个位置要生成多长的向量（通常 = hidden_dim）
    # max_len：最多支持的时间步长度，比如最长句子为1000个字
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # 创建空张量，初始化
        # 和下面的embedding大小一致
        pe = torch.zeros(max_len, d_model)
        # 构造位置索引
        position = torch.arange(0, max_len).unsqueeze(1)
        # 控制维度频率
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 填入sin，cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        # 加batch维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        # 定义为模型的“常量”，会被保存但不参与训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, D]  → 输出：加了位置偏置信息的 [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]  # 取前T个位置
        return x

# 预测持续时间模块
class DurationPredictor(nn.Module):
    """
    预测字符隐向量的持续时间（帧数）
    输入是 [B, C, T] 的文本隐表示，输出是每个时间步的 duration（帧长度）
    """
    def __init__(self, in_channels, filter_channels, kernel_size=3, dropout=0.5, num_layers=2):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels if i == 0 else filter_channels,
                              filter_channels,
                              kernel_size,
                              padding=(kernel_size - 1) // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(filter_channels),
                    nn.Dropout(dropout)
                )
            )

        self.linear_out = nn.Conv1d(filter_channels, 1, kernel_size=1)  # 输出为 scalar（log-duration）

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, C, T] 文本隐表示
            mask: [B, 1, T]，可选遮罩，True 处为 padding
        Returns:
            log_durations: [B, 1, T]，对数持续时间
        """
        for layer in self.layers:
            x = layer(x)

        out = self.linear_out(x)  # [B, 1, T]

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

#文字与音频帧对齐模块
class LengthRegulator(nn.Module):
    """
    根据预测的 duration，将字符隐表示复制扩展，实现对齐
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, durations):
        """
        Args:
            x: [B, T, C] 文本隐表示（T为字符数）
            durations: [B, T] 每个 token 的持续时间（整数）
        Returns:
            output: [B, T_total, C] 拉长后的帧表示
        """
        output = []
        for batch_x, batch_d in zip(x, durations):  # 遍历每个 batch
            expanded = self._expand(batch_x, batch_d)
            output.append(expanded)

        # 将不同 batch pad 到相同长度
        output = self._pad(output)
        return output  # [B, max_T, C]

    def _expand(self, enc, dur):
        """
        Args:
            enc: [T, C]
            dur: [T]
        Returns:
            expanded: [sum(dur), C]
        """
        out = []
        for i, d in enumerate(dur):
            repeat_times = max(int(d.item()), 0)
            out.append(enc[i].unsqueeze(0).repeat(repeat_times, 1))  # [d, C]
        if len(out) > 0:
            return torch.cat(out, dim=0)  # [T_total, C]
        else:
            return torch.zeros((0, enc.size(-1)), device=enc.device)

    def _pad(self, batch):
        """
        将 batch 中不同长度的序列 pad 到统一长度
        Args:
            batch: List of [T_i, C]
        Returns:
            Tensor: [B, max_T, C]
        """
        max_len = max([item.size(0) for item in batch])
        out = []
        for item in batch:
            pad_len = max_len - item.size(0)
            if pad_len > 0:
                pad_tensor = torch.zeros((pad_len, item.size(1)), device=item.device)
                padded = torch.cat([item, pad_tensor], dim=0)
            else:
                padded = item
            out.append(padded.unsqueeze(0))  # [1, T, C]
        return torch.cat(out, dim=0)  # [B, T, C]


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
        embedded = self.embedding(text)  # [B, T_text, hidden]
        _, (h_n, _) = self.text_encoder(embedded)  # [2, B, hidden]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, hidden*2]

        e = self.emotion_proj(emotion)  # [B, hidden*2]
        combined = h + e  # [B, hidden*2]

        waveform = self.decoder(combined)  # [B, waveform_len]
        return waveform


class SimpleVITS(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=256, emotion_dim=1, hidden_dim=512):
        super(SimpleVITS, self).__init__()

        # 文本编码器：嵌入 + 双层 Transformer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 情感融合：情绪强度映射到 embedding 维度
        self.emotion_fc = nn.Linear(emotion_dim, embed_dim)

        # 上采样模块：将 token-level 特征上采样成接近波形长度
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # ×2
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # ×4
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # ×8
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # ×16
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # ×32
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # ×64
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # ×128
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, 1, kernel_size=4, stride=2, padding=1),  # ×256
        )

    def forward(self, text, emotion):
        # text: [B, T], emotion: [B, 1]

        # 1. 文本嵌入编码
        x = self.embedding(text)  # [B, T, D]
        x = self.text_encoder(x)  # [B, T, D]

        # 2. 情绪调制：简单加法
        e = self.emotion_fc(emotion).unsqueeze(1)  # [B, 1, D]
        e = e.expand(-1, x.size(1), -1)  # [B, T, D]
        x = x + e  # [B, T, D]

        # 3. 上采样：调整为类似波形长度
        x = x.permute(0, 2, 1)  # [B, D, T] 适配 ConvTranspose1d
        waveform = self.upsample(x)  # [B, 1, T_up]
        waveform = waveform.squeeze(1)  # [B, T_up]

        return waveform


# 最终完全版vits
class FullVITS(nn.Module):
    def __init__(self, vocab_size=5000, emotion_dim=1, hidden_dim=256):
        super(FullVITS, self).__init__()

        #### 1. TextEncoder: transformer（文本编码器）
        # 词向量映射（Embedding）
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # 位置信息注入（PositionalEncoding）
        self.positional_encoding = PositionalEncoding(hidden_dim)
        # 使用encoder的block（编码器层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        # encoder是多个encoder layer拼的（编码器）
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        #### 2. EmotionEmbedding: 映射+融合
        self.emotion_fusion = EmotionFusion(
            fusion_method=fusion_method,  # 如 "concat"
            text_dim=256,  # TextEncoder 输出维度
            emotion_input_dim=1,
            emotion_hidden_dim=64  # 你设置的情绪向量维度
        )
        # 记录情绪融合后的维度
        self.emo_text_dim = self.emotion_fusion.output_dim

        #### 3. DurationPredictor: 音素持续时间建模（持续时间预测器）
        self.duration_predictor = DurationPredictor(
            in_channels=self.emo_text_dim,
            filter_channels=256,
            kernel_size=3,
            dropout=0.5,
            num_layers=2
        )

        #### 4. PosteriorEncoder: mel -> latent（后验编码器）
        self.posterior_encoder = PosteriorEncoder(
            in_channels=80,  # mel 特征维度
            hidden_channels=256,  # 中间层维度
            latent_dim=192  # 隐空间维度（可调）
        )

        #### 5. Normalizing Flow（正则化流）
        self.flow = ResidualCouplingBlock(
            channels=192,  # 与 PosteriorEncoder 输出 latent_dim 保持一致
            hidden_channels=256,
            kernel_size=5,
            num_flows=4
        )

        #### 6.Length Regulator（长度调节器）
        self.length_regulator = LengthRegulator()

        #### 7. Decoder: waveform生成模块（可复用 HiFi-GAN 或精简版）（声码器 / 波形解码器）
        self.decoder = nn.Sequential(
            nn.Conv1d(self.emo_text_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=5, padding=2),  # 输出一维波形
        )

    def forward(self, text, emotion, mel=None):
        # 1. 文本编码部分
        x = self.embedding(text)  # [B, T] → [B, T, D]
        x = self.positional_encoding(x)  # 添加位置信息
        text_features = self.text_encoder(x)  # Transformer 编码后输出 [B, T, D]

        # 2. 情绪嵌入（从标量变向量）
        fused_emb = self.emotion_fusion(text_features, emotion)  # [B, T, D+E]

        # 3. 持续时间预测
        log_durations = self.duration_predictor(fused_emb.permute(0, 2, 1), mask=None)
        # 将 log_durations 变为整数 duration
        durations = torch.clamp(torch.round(torch.exp(log_durations.squeeze(1))), min=1).long()  # [B, T]

        # 4. 后验编码器（仅训练阶段使用），提取 z_post
        if mel is not None:
            z_post, mu, log_var = self.posterior_encoder(mel)  # [B, latent_dim, T]
        # 5. flow，将 z_post → z_p
            z_p, log_det = self.flow(z_post)
        else:
            z_post, mu, log_var, z_p, log_det = None, None, None, None, None

        # 6. 长度调节器
        regulated = self.length_regulator(fused_emb, durations)  # [B, T', C]

        # 7.解码器
        x_dec = regulated.permute(0, 2, 1)  # [B, C, T] → 匹配 Conv1d 输入
        waveform = self.decoder(x_dec).squeeze(1)  # [B, 1, T] → [B, T]

        return waveform, z_post, mu, log_var, z_p, log_det


def build_vits_model(model_type="test", vocab_size=5000):
    if model_type == "test":
        return TestVITS(vocab_size=vocab_size)
    elif model_type == "simple":
        return SimpleVITS(vocab_size=vocab_size)
    elif model_type == "full":
        return FullVITS(vocab_size=vocab_size)
    else:
        raise ValueError(f"❌ 不支持的模型类型: {model_type}")

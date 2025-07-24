import torch
import torch.nn as nn


class ConvNormReLU(nn.Module):
    """
    一个卷积块：Conv1d + LayerNorm + ReLU，自动处理维度转换。
    输入输出都是 [B, C, T]。
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)           # [B, C_out, T]
        x = x.transpose(1, 2)      # [B, T, C_out]
        x = self.norm(x)           # [B, T, C_out]
        x = x.transpose(1, 2)      # [B, C_out, T]
        x = self.relu(x)
        return x

class PosteriorEncoder(nn.Module):
    """
    后验编码器：将 mel-spectrogram 编码为潜在变量 z
    输入：mel [B, 80, T]，输出：z, mu, log_var ∈ [B, latent_dim, T]
    """
    def __init__(self, in_channels, hidden_channels, latent_dim, kernel_size=5, num_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                ConvNormReLU(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size
                )
            )

        self.proj_mean = nn.Conv1d(hidden_channels, latent_dim, kernel_size=1)
        self.proj_log_var = nn.Conv1d(hidden_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        """
        x: [B, 80, T]  # mel 频谱
        返回：z, mu, log_var: [B, latent_dim, T]
        """
        for conv in self.convs:
            x = conv(x)

        mu = self.proj_mean(x)
        log_var = self.proj_log_var(x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps  # 重参数采样

        return z, mu, log_var

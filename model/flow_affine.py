import torch
import torch.nn as nn
import torch.nn.functional as F

class Flip(nn.Module):
    """
    对称变换。用于交换通道顺序：x[:, 0::2] <-> x[:, 1::2]
    x = [x0, x1, x2, x3] → flip → [x3, x2, x1, x0]
    增强耦合结构的表达能力
    """
    def __init__(self):
        super().__init__()
    def forward(self, x, mask=None):
        # 交换 channel 维度顺序，增强耦合维度
        return x.flip(1), torch.tensor(0.0, device=x.device)  # 第二个返回值 log_det 恒为0
    def reverse(self, x, mask=None):
        # 翻转是可逆操作，flip 两次即还原
        return x.flip(1)

class AffineCouplingLayer(nn.Module):
    """
        仿射耦合层（Affine Coupling Layer）
        x1 = x1 * exp(s) + t
        log_det = sum(s)
    """
    def __init__(self, channels, hidden_channels, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv1d(channels // 2, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1)  # 输出 s 和 t
        )

    def forward(self, x, mask=None):
        x0, x1 = x.chunk(2, dim=1)
        h = self.net(x0)
        log_s, t = h.chunk(2, dim=1)
        s = torch.tanh(log_s)  # 限制范围防止不稳定
        x1 = x1 * torch.exp(s) + t
        z = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(s, dim=[1, 2])  # [B]
        return z, log_det

    def reverse(self, z, mask=None):
        z0, z1 = z.chunk(2, dim=1)
        h = self.net(z0)
        log_s, t = h.chunk(2, dim=1)
        s = torch.tanh(log_s)
        z1 = (z1 - t) * torch.exp(-s)
        x = torch.cat([z0, z1], dim=1)
        return x

# 残差耦合模块
# 由多个耦合层（coupling layers）+ flip 操作组成，用于将 z_post 映射为标准高斯分布中的 z_p，或反过来
class ResidualCouplingBlock(nn.Module):
    # channels：潜在变量的维度，与 PosteriorEncoder 输出一致
    # hidden_channels：Flow 中间层通道数
    # num_flows：建议设置为 4~6，代表耦合层+flip对的数量
    def __init__(self, channels, hidden_channels, kernel_size=5, num_flows=4):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(num_flows):
            self.flows.append(
                AffineCouplingLayer(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size
                )
            )
            self.flows.append(Flip())  # 加入 Flip 层增加耦合通道

    # 训练阶段，z_post → z_p
    def forward(self, x, mask=None):
        log_det_tot = 0.0
        for flow in self.flows:
            x, log_det = flow(x, mask=mask)
            log_det_tot += log_det
        return x, log_det_tot

    # 推理阶段，z_p → z_post
    def reverse(self, x, mask=None):
        for flow in reversed(self.flows):
            x = flow.reverse(x, mask=mask)
        return x

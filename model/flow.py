import torch
import torch.nn as nn
import torch.nn.functional as F

class Flip(nn.Module):
    """
    对称变换。用于交换通道顺序：x[:, 0::2] <-> x[:, 1::2]
    x = [x0, x1, x2, x3] → flip → [x3, x2, x1, x0]
    增强不同维度之间的耦合能力
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        # 交换 channel 维度顺序，增强耦合维度
        return x.flip(1), 0.0  # 第二个返回值 log_det 恒为0

    def reverse(self, x, mask=None):
        # 翻转是可逆操作，flip 两次即还原
        return x.flip(1)

class ResidualCouplingLayer(nn.Module):
    """
    加性耦合层（Additive Coupling Layer）
    输入张量分为两部分，前一半作为条件生成变换参数
    """
    def __init__(self, channels, hidden_channels, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv1d(channels // 2, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, channels // 2, kernel_size=1)
        )

    def forward(self, x, mask=None):
        # x: [B, C, T]
        x0, x1 = x.chunk(2, dim=1)  # 分为两部分 [B, C//2, T]
        h = self.net(x0)            # 基于 x0 生成残差项
        x1 = x1 + h                 # 加性耦合：x1 += f(x0)
        z = torch.cat([x0, x1], dim=1)  # 拼接还原
        log_det = 0.0
        return z, log_det

    def reverse(self, z, mask=None):
        z0, z1 = z.chunk(2, dim=1)
        h = self.net(z0)
        z1 = z1 - h  # 反操作：减去
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
                ResidualCouplingLayer(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size
                )
            )
            self.flows.append(Flip())  # 加入 Flip 层增加耦合通道

    # 训练阶段，z_post → z_p
    def forward(self, x, mask=None):
        log_det_tot = 0
        for flow in self.flows:
            x, log_det = flow(x, mask=mask)
            log_det_tot += log_det
        return x, log_det_tot

    # 推理阶段，z_p → z_post
    def reverse(self, x, mask=None):
        for flow in reversed(self.flows):
            x = flow.reverse(x, mask=mask)
        return x

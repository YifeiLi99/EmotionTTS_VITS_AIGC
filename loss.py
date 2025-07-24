import torch
import torch.nn.functional as F
import math
from loss_stft import STFTLoss

def vits_loss(waveform_pred, waveform_gt, mu, log_var, z_p, log_det):
    """
    VITS 三项损失：重构 + KL 散度 + Flow NLL

    Args:
        waveform_pred: [B, T]，模型预测波形
        waveform_gt:   [B, T]，真实波形
        mu, log_var:   后验编码器输出
        z_p:           Flow 映射后的隐变量
        log_det:       Flow 模块输出的 log|detJ|

    Returns:
        total_loss: 总损失
        recon_loss: 波形重构 MSE
        kl_loss:    后验 KL 散度
        flow_loss:  Flow NLL
    """
    # 1. 重构误差
    # 初始化
    stft_loss_fn = STFTLoss().to(waveform_pred.device)
    # 重建损失
    recon_l1 = F.l1_loss(waveform_pred, waveform_gt)
    recon_stft = stft_loss_fn(waveform_pred, waveform_gt)
    # 融合两个重建损失
    recon_loss = recon_l1 + recon_stft * 1.0  # STFT 权重可调

    # 2. KL Loss
    kl_loss = 0.5 * torch.mean(torch.sum(
        mu ** 2 + torch.exp(log_var) - log_var - 1,
        dim=1
    ))

    # 3. Flow Loss（负对数似然）
    # 注意：log_det 为 [B]，z_p 为 [B, C, T]
    log_prob = -0.5 * torch.sum(z_p ** 2 + math.log(2 * math.pi), dim=[1, 2])  # [B]
    #平衡各项 loss，避免 flow dominate 一切
    flow_loss = 0.001 * torch.mean(-log_prob - log_det)  # NLL

    # 加 loss 权重
    total_loss = 1.0 * recon_loss + 1.0 * kl_loss + 0.001 * flow_loss
    return total_loss, recon_loss, kl_loss, flow_loss

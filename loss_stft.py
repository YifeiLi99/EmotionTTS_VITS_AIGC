import torch
import torch.nn.functional as F
import torchaudio.transforms as T

class STFTLoss(torch.nn.Module):
    def __init__(self, fft_size=1024, hop_size=256, win_length=1024):
        super(STFTLoss, self).__init__()
        self.n_fft = fft_size  # ✅ 添加 n_fft 成员变量
        self.stft = T.Spectrogram(n_fft=fft_size, hop_length=hop_size, win_length=win_length, power=None)

    def forward(self, pred_waveform, gt_waveform):
        """
        pred_waveform: [B, T] or [B, 1, T] - 预测波形
        gt_waveform:   [B, T] or [B, 1, T] - 真实波形
        """
        # ✅ 如果是 [B, 1, T]，则 squeeze 掉 channel
        if pred_waveform.dim() == 3 and pred_waveform.shape[1] == 1:
            pred_waveform = pred_waveform.squeeze(1)
        if gt_waveform.dim() == 3 and gt_waveform.shape[1] == 1:
            gt_waveform = gt_waveform.squeeze(1)

        # ✅ 如果长度不足，则跳过计算
        if pred_waveform.shape[-1] < self.n_fft:
            return torch.tensor(0.0, device=pred_waveform.device)

        # ✅ STFT
        spec_pred = self.stft(pred_waveform)
        spec_gt = self.stft(gt_waveform)
        loss = F.l1_loss(spec_pred, spec_gt)
        return loss

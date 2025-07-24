import torchaudio
import torch

def waveform_to_mel(waveform, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
    """
    将 waveform 转为 mel-spectrogram，适用于 PosteriorEncoder 输入
    waveform: Tensor of shape [B, 1, T]
    return: mel-spectrogram of shape [B, 80, T']
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to(waveform.device)

    mel = mel_transform(waveform.squeeze(1))  # [B, 80, T']
    return mel

###用于将多个不同的batch合并在一起。如果文字或者语音长度不一，就加padding变成一样长。

import torch
from torch.nn.utils.rnn import pad_sequence
import os
from config import PROCESSED_DIR

def vits_collate_fn(batch):
    """
    用于 VITS 情感语音合成的数据批处理函数
    对 text 和 waveform 进行 padding，并返回长度信息

    Args:
        batch: List[Dict]，每个样本为：{
            'text': LongTensor [T_text],
            'emotion': FloatTensor [1],
            'waveform': FloatTensor [T_wav]
        }

    Returns:
        Dict[str, Tensor]：包括对齐后的 batch 数据和长度信息
    """
    # 提取各字段
    texts = [item["text"] for item in batch]  # List[Tensor]
    emotions = [item["emotion"] for item in batch]  # List[Tensor]
    waveforms = [item["waveform"] for item in batch]  # List[Tensor]

    # 记录原始长度
    text_lengths = torch.LongTensor([t.size(0) for t in texts])
    waveform_lengths = torch.LongTensor([w.size(0) for w in waveforms])

    # padding 对齐
    text_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # (B, T_text)
    waveform_padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)  # (B, T_wav)

    # 拼接情感极性 (B, 1)
    emotion_tensor = torch.cat(emotions, dim=0).unsqueeze(1)

    return {
        "text": text_padded,  # LongTensor [B, T_text]
        "text_lengths": text_lengths,  # LongTensor [B]
        "emotion": emotion_tensor,  # FloatTensor [B, 1]
        "waveform": waveform_padded,  # FloatTensor [B, T_wav]
        "waveform_lengths": waveform_lengths  # LongTensor [B]
    }


# 测试代码（调试用）
if __name__ == "__main__":
    from vits_dataset import VITSEmotionDataset, char_tokenizer
    from torch.utils.data import DataLoader

    jsonl_path = os.path.join(PROCESSED_DIR, "metadata_emovie.jsonl")
    dataset = VITSEmotionDataset(jsonl_path=jsonl_path, tokenizer=char_tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=vits_collate_fn)

    for batch in dataloader:
        print("text:", batch["text"].shape)
        print("text_lengths:", batch["text_lengths"])
        print("emotion:", batch["emotion"].shape)
        print("waveform:", batch["waveform"].shape)
        print("waveform_lengths:", batch["waveform_lengths"])
        break

###用于把JSONL文件，变换为VITS模型所需要的输入格式

import json
import torch
from torch.utils.data import Dataset
import torchaudio
from config import PROCESSED_DIR
import os

class VITSEmotionDataset(Dataset):
    # 自定义用于 VITS 训练的情感语音合成数据集加载器
    def __init__(self, jsonl_path, tokenizer, sampling_rate=22050):
        # jsonl_path (str): JSONL 文件路径，每行一个样本（包含 text / polarity / wav_path）
        # tokenizer (callable): 文本分词器，输入字符串，返回 List[int]
        # sampling_rate (int): 希望统一的音频采样率（默认 22050Hz）
        self.data = []
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate

        # 加载 JSONL 数据
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get("text", "").strip()
                    polarity = float(item.get("emotion_polarity", 0.0))
                    wav_path = item.get("wav_path", "")

                    # 检查字段是否为空或路径不存在
                    if not text or not wav_path or not os.path.exists(wav_path):
                        print(f"⚠️ 跳过无效样本: 缺字段或找不到文件: {item}")
                        continue

                    self.data.append({
                        "text": text,
                        "polarity": polarity,
                        "wav_path": wav_path
                    })

                except Exception as e:
                    print(f"⚠️ 跳过格式错误样本: {line.strip()}，错误: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 文本处理：转为 token ID 序列
        text = sample["text"]
        token_ids = self.tokenizer(text)  # List[int]
        # print("原始文本:", text)
        # print("分词后 token_ids 类型:", type(token_ids))
        # print("token_ids:", token_ids)
        text_tensor = torch.LongTensor(token_ids)

        # 情感强度处理：转为 FloatTensor
        polarity_tensor = torch.tensor([sample["polarity"]], dtype=torch.float32)

        # 加载音频并重采样
        wav_path = sample["wav_path"]
        try:
            # waveform shape: [1, T]
            waveform, sr = torchaudio.load(wav_path)
        except Exception as e:
            raise RuntimeError(f"❌ 加载音频失败: {wav_path}，错误: {e}")
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        return {
            "text": text_tensor,  # 文本 token 序列
            "emotion": polarity_tensor,  # 情感极性张量（1维）
            "waveform": waveform.squeeze(0)  # 去掉通道维度，shape: [T]
        }


# 示例 tokenizer（将每个字符映射为其 Unicode 编码）
def char_tokenizer(text):
    """
    字符级 tokenizer 示例：每个汉字或字符转为其 Unicode 编码整数
    """
    return [ord(c) for c in text]


# 独立运行测试示例（建议调试用）
if __name__ == "__main__":
    jsonl_path = os.path.join(PROCESSED_DIR, "metadata_emovie.jsonl")
    dataset = VITSEmotionDataset(jsonl_path, tokenizer=char_tokenizer)
    sample = dataset[0]
    print("Text Tensor:", sample["text"])
    print("Emotion Polarity:", sample["emotion"])
    print("Waveform Shape:", sample["waveform"].shape)

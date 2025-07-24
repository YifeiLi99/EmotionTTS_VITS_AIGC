import os
import torch
import torchaudio
from config import DEVICE, WEIGHTS_DIR, MODEL_TYPE
from data.vits_dataset import char_tokenizer, get_vocab_size_from_tokenizer
from model.VITS_model import build_vits_model

# ----------------------------
# 1. 加载模型
# ----------------------------
VOCAB_SIZE = get_vocab_size_from_tokenizer(char_tokenizer)
model = build_vits_model(model_type=MODEL_TYPE, vocab_size=VOCAB_SIZE).to(DEVICE)
ckpt_path = os.path.join(WEIGHTS_DIR, "best_model.pt")

assert os.path.exists(ckpt_path), f"❌ 模型文件未找到: {ckpt_path}"
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()
print(f"✅ 已加载模型权重: {ckpt_path}")

# ----------------------------
# 2. 准备输入文本和情绪
# ----------------------------
text = "今天的天气真不错啊，我们一起去郊游吧！"
text_tensor = torch.LongTensor([char_tokenizer(text)]).to(DEVICE)

# 用随机值模拟 64 维情绪向量 (VITS 基于情绪 embedding)
emotion_tensor = torch.randn(1, 1).to(DEVICE)  # 适配情绪融合模块的 Linear(64, hidden_dim)

# ----------------------------
# 3. 构造 mel（可选）
# ----------------------------
mel_dummy = torch.randn(1, 80, 100).to(DEVICE)  # 模拟 mel，仅为结构通畅

# ----------------------------
# 4. 模型推理
# ----------------------------
with torch.no_grad():
    waveform_pred, *_ = model(
        text_tensor,
        emotion_tensor,
        mel=mel_dummy
    )

# ----------------------------
# 5. 音频归一化并保存
# ----------------------------
waveform_pred = waveform_pred.cpu()
waveform_pred = waveform_pred / waveform_pred.abs().max() * 0.9  # 标准化防止爆音
output_path = "output.wav"
torchaudio.save(output_path, waveform_pred.T, sample_rate=22050)
print(f"✅ 音频已保存到: {output_path}")
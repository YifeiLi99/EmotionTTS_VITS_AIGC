import os
import torch
import torchaudio
from model.VITS_model import build_vits_model
from config import DEVICE, MODEL_TYPE, WEIGHTS_DIR
from data.vits_dataset import char_tokenizer, get_vocab_size_from_tokenizer
import numpy as np

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
# 2. 准备输入数据
# ----------------------------
text = "我真的非常开心。"  # 你想测试的文本
emotion_id = 0             # 改成你模型支持的情绪编号：0=高兴，1=悲伤，2=愤怒 等

# 文本编码
text_ids = char_tokenizer(text)
text_tensor = torch.LongTensor(text_ids).unsqueeze(0).to(DEVICE)  # [1, T]
text_lengths = torch.LongTensor([len(text_ids)]).to(DEVICE)

# 情绪编码
emotion_tensor = torch.LongTensor([[emotion_id]]).float().to(DEVICE)  # [1, 1] => FloatTensor

# ----------------------------
# 3. 构造 mel（暂用 dummy，模型内部实际可能忽略）
# ----------------------------
mel_dummy = torch.randn(1, 80, 100).to(DEVICE)  # 只是为了打通结构，可替换为真实 mel

# ----------------------------
# 4. 推理生成 waveform
# ----------------------------
with torch.no_grad():
    waveform_pred, *_ = model(
        text_tensor,
        emotion_tensor,
        mel=mel_dummy
    )  # 输出: [B, T]

# ----------------------------
# 5. 保存输出音频
# ----------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test.wav")

# 防止太小声，可适度放大
waveform_pred = waveform_pred.cpu().clamp(-1, 1) * 2.5

torchaudio.save(output_path, waveform_pred, sample_rate=22050)
print(f"✅ 合成完毕，输出音频: {output_path}")

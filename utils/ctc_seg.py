import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctc_segmentation import (
    ctc_segmentation,
    CtcSegmentationParameters,
    prepare_text,
    determine_utterance_segments
)
import json

# ===================== 配置路径 =====================
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
WAV_PATH = "example.wav"           # 输入 WAV 音频（16kHz 单声道）
TXT_PATH = "example.txt"           # 输入文本，逐字内容，无标点
OUTPUT_PATH = "duration.json"     # 输出帧级时长对齐结果

# ===================== 加载模型 =====================
print("[INFO] 加载模型...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# ===================== 加载文本 =====================
with open(TXT_PATH, "r", encoding="utf-8") as f:
    transcript = f.read().strip().replace(" ", "")
    if not transcript:
        raise ValueError("输入文本为空")

# ===================== 加载音频 =====================
wav, sr = torchaudio.load(WAV_PATH)
if sr != 16000:
    raise ValueError("音频采样率必须为16kHz")

# ===================== 推理 logits =====================
print("[INFO] 推理生成 logits...")
inputs = processor(wav.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1).cpu().numpy()  # [T, V]

# ===================== 构造 CTC 参数 =====================
print("[INFO] 构建 CTC 分割参数...")
config = CtcSegmentationParameters()

# 设置 index_duration（每帧持续时间）
if hasattr(processor.feature_extractor, "hop_length"):
    config.index_duration = processor.feature_extractor.hop_length / processor.feature_extractor.sampling_rate
else:
    config.index_duration = 20 / 1000  # fallback: assume 20ms per frame

char_list = processor.tokenizer.convert_ids_to_tokens(list(range(len(processor.tokenizer))))

# ===================== 字符映射 & 准备文本 =====================
transcript_clean = list(transcript)  # 拆成单字列表

# 检查每个字是否在字典中可以对齐
valid_chars = [c for c in transcript_clean if c in char_list]
if len(valid_chars) == 0:
    raise ValueError("输入文本中的字无法映射到模型的 vocabulary 中。请确认字符集是否兼容。")

gt, utt_begin_indices = prepare_text(config, valid_chars, char_list)

# ===================== 计算对齐 =====================
print("[INFO] 执行 CTC 对齐...")
_, timings, alignments = ctc_segmentation(config, log_probs, gt)
segments = determine_utterance_segments(config, utt_begin_indices, alignments, timings, valid_chars)

# ===================== 写入输出 =====================
print("[INFO] 写入 duration.json")
duration_info = []
for i, (start, end, char, _, _) in enumerate(segments):
    if char == "" or isinstance(char, float):
        continue
    duration_info.append({
        "char": char,
        "start_frame": int(start),
        "end_frame": int(end),
        "duration": int(end - start)
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(duration_info, f, ensure_ascii=False, indent=2)

print(f"✅ 对齐完成，共处理 {len(duration_info)} 个字，输出保存至: {OUTPUT_PATH}")

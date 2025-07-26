import os
import torch
import torchaudio
import json
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctc_segmentation import (
    ctc_segmentation,
    CtcSegmentationParameters,
    prepare_text,
    determine_utterance_segments
)

# ===================== 配置路径 =====================
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
WAV_PATH = "example.wav"           # 输入 WAV 音频（16kHz 单声道）
TXT_PATH = "example.txt"           # 输入文本，逐字内容，无标点
OUTPUT_PATH = "duration.json"     # 输出帧级时长对齐结果

# ===================== 加载模型 =====================
print("[INFO] 加载模型...")
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
except Exception as e:
    raise RuntimeError(f"[ERROR] 模型加载失败: {e}")

# ===================== 加载文本 =====================
with open(TXT_PATH, "r", encoding="utf-8") as f:
    transcript = f.read().strip().replace(" ", "")
    if not transcript:
        raise ValueError("[ERROR] 输入文本为空")
    transcript_chars = list(transcript)

# ===================== 加载音频 =====================
wav, sr = torchaudio.load(WAV_PATH)
if sr != 16000:
    raise ValueError("[ERROR] 音频采样率必须为16kHz")

# ===================== 推理 logits =====================
print("[INFO] 推理生成 logits...")
inputs = processor(wav.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1).cpu().numpy()
print("[DEBUG] log_probs.shape:", log_probs.shape)

# ===================== 构造 CTC 参数 =====================
print("[INFO] 构建 CTC 分割参数...")
config = CtcSegmentationParameters()

# 设置 index_duration
try:
    hop_length = getattr(processor.feature_extractor, "hop_length", None)
    sr = processor.feature_extractor.sampling_rate
    if hop_length is not None:
        config.index_duration = hop_length / sr
    else:
        config.index_duration = 20 / 1000  # 默认 20ms 每帧
except Exception:
    config.index_duration = 20 / 1000

# 获取字符列表
char_list = processor.tokenizer.convert_ids_to_tokens(list(range(len(processor.tokenizer))))
print("[DEBUG] 模型词表样例:", char_list[:20])

# 筛选 transcript 中合法字符
valid_chars = [c for c in transcript_chars if c in char_list]
print("[DEBUG] 有效字符:", valid_chars)
if len(valid_chars) == 0:
    raise ValueError("❌ 输入文本无法与模型词表对齐，请检查字符集")

# 准备 ground truth
gt, utt_begin_indices = prepare_text(config, valid_chars, char_list)

# ===================== 计算对齐 =====================
print("[INFO] 执行 CTC 对齐...")
try:
    _, timings, char_probs = ctc_segmentation(config, log_probs, gt)
    segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, valid_chars)
except Exception as e:
    raise RuntimeError(f"[ERROR] 对齐失败: {e}")

# ===================== 写入输出 =====================
print("[INFO] 写入 duration.json")
duration_info = []
for seg in segments:
    start, end, char = seg[:3]
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

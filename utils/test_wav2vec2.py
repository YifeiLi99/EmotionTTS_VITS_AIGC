from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch

# 模型选择：使用 jonatasgrosman 提供的 fine-tuned 模型
#MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
MODEL_NAME = "wbbbbb/wav2vec2-large-chinese-zh-cn"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

waveform, sr = torchaudio.load("example.wav")
if sr != 16000:
    raise ValueError("请确保音频为 16kHz 采样率")
waveform = waveform.squeeze()

inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(**inputs).logits  # [1, time, vocab_size]

print(f"logits shape: {logits.shape}")

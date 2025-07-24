import os
from pathlib import Path
from tqdm import tqdm
import subprocess

# 获取当前脚本所在目录
base_dir = Path(__file__).resolve().parent
# 输入原始 wav 目录   原始 wav（22kHz）
src_dir = base_dir / "processed" / "mfa_train_data" / "wav"
# 输出目标 wav 目录   16kHz 单声道 wav
dst_dir = base_dir / "processed" / "mfa_train_data" / "wav16k"
dst_dir.mkdir(parents=True, exist_ok=True)

for wav_file in tqdm(list(src_dir.glob("*.wav")), desc="转换音频"):
    out_path = dst_dir / wav_file.name
    command = [
        "ffmpeg",
        "-y",
        "-i", str(wav_file),
        "-ac", "1",          # 单声道
        "-ar", "16000",      # 采样率 16kHz
        str(out_path)
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"✅ 已完成转换，共 {len(list(dst_dir.glob('*.wav')))} 条音频。")

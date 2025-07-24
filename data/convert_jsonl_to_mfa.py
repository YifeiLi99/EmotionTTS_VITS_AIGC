import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from config import PROCESSED_DIR

# 原始 JSONL 路径
jsonl_path = os.path.join(PROCESSED_DIR,"train.jsonl")

# 输出目录
output_dir = os.path.join(PROCESSED_DIR,"mfa_train_data")
os.makedirs(output_dir, exist_ok=True)
#输出二级目录
wav_out_dir = os.path.join(output_dir,"wav")
lab_out_dir = os.path.join(output_dir,"lab")
os.makedirs(wav_out_dir, exist_ok=True)
os.makedirs(lab_out_dir, exist_ok=True)

# 化为path路径
wav_out_dir = Path(wav_out_dir)
lab_out_dir = Path(lab_out_dir)

with open(jsonl_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(tqdm(f, desc="转换中")):
        data = json.loads(line)
        audio_path = data["wav_path"]
        text = data["text"]

        # 目标文件名：统一编号
        file_id = f"{idx:04d}"

        # 拷贝音频
        wav_dst = wav_out_dir / f"{file_id}.wav"
        shutil.copy(audio_path, wav_dst)

        # 写入 lab 文件
        lab_dst = lab_out_dir / f"{file_id}.lab"
        with open(lab_dst, "w", encoding="utf-8") as lab_file:
            lab_file.write(text.strip())

print(f"✅ 已完成转换，共处理 {idx+1} 条样本。")
print(f"➡️ WAV 文件输出目录: {wav_out_dir}")
print(f"➡️ LAB 文件输出目录: {lab_out_dir}")

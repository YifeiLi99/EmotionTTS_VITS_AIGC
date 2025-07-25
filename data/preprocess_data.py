"""
用于把最原始的散乱的训练集文件，整理成一个JSONL文件，录入信息方便后续操作
"""

import os
import json
import csv
from config import RAW_DATA_DIR, RAW_LABELS_FILE, PROCESSED_DIR

#输出的jsonl文件名字
jsonl_path = os.path.join(PROCESSED_DIR, "metadata_emovie.jsonl")
label_map_path = os.path.join(PROCESSED_DIR, "polarity_map.json")

# 定义情绪极性映射表：将数值标签（如 -0.5）映射为字符串情绪类别（如 negative）
polarity_map = {
    "-1.0": "very_negative",
    "-0.5": "negative",
    "0.0": "neutral",
    "0.5": "positive",
    "1.0": "very_positive"
}

# 保存标签映射表（可视化或训练时备用）
with open(label_map_path, 'w', encoding='utf-8') as f:
    json.dump(polarity_map, f, ensure_ascii=False, indent=2)
print(f"标签映射表已保存到: {label_map_path}")

# 读取 labels.csv 并生成 jsonl 文件
with open(RAW_LABELS_FILE, 'r', encoding='utf-8') as csvfile, \
        open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)

    # 遍历标签 CSV 文件中的每一行，解析音频路径、文本内容、情绪极性并统一格式
    for row in reader:
        wav_filename = row['filename']
        text = row['text'].strip()

        # 尝试将标签转为标准小数精度字符串键
        try:
            polarity_float = float(row['polarity_label'])
            polarity_key = f"{polarity_float:.1f}"
        except Exception as e:
            print(f"[错误] 行解析失败，文件: {wav_filename}，异常: {e}，已跳过")
            continue

        if polarity_key not in polarity_map:
            print(f"[警告] 未知极性标签: {polarity_key}，文件名: {wav_filename}，已跳过")
            continue

        emotion_class = polarity_map[polarity_key]
        wav_path = os.path.join(RAW_DATA_DIR, wav_filename)

        sample = {
            "text": text,
            "emotion_polarity": polarity_float,
            "emotion_class": emotion_class,
            "wav_path": wav_path.replace('\\', '/')  # 保持兼容性（适配 Windows 路径）
        }

        jsonlfile.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"✅ 处理完成，已保存到: {jsonl_path}")

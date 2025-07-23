###用于把一个完整的JSONL训练集，自由划分成train,val,test三个JSONL

import os
import random
from config import PROCESSED_DIR

# ========== 配置参数 ==========
# 原始 jsonl 数据路径
SOURCE_JSONL = os.path.join(PROCESSED_DIR,"metadata_emovie.jsonl")
SAVE_DIR = PROCESSED_DIR                      # 输出保存目录
TRAIN_RATIO = 0.7                             # 训练集占比
VAL_RATIO = 0.2                               # 验证集占比
TEST_RATIO = 0.1                              # 测试集占比
RANDOM_SEED = 42                              # 固定随机种子，保证可复现性

# 检查比例之和是否为 1
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "比例加起来应为1"

# ========== 加载原始 JSONL 数据 ==========
with open(SOURCE_JSONL, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"原始数据总数: {len(lines)} 条")

# ========== 随机打乱并划分 ==========
random.seed(RANDOM_SEED)
random.shuffle(lines)

total = len(lines)
num_train = int(total * TRAIN_RATIO)
num_val = int(total * VAL_RATIO)
num_test = total - num_train - num_val  # 剩余样本全部归入 test

# 拆分三部分数据
train_data = lines[:num_train]
val_data = lines[num_train:num_train + num_val]
test_data = lines[num_train + num_val:]

# ========== 保存函数 ==========
def save_jsonl(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(data)
    print(f"已保存 {filepath} ({len(data)} 条样本)")

# ========== 保存至目标目录 ==========
save_jsonl(os.path.join(SAVE_DIR, "train.jsonl"), train_data)
save_jsonl(os.path.join(SAVE_DIR, "val.jsonl"), val_data)
save_jsonl(os.path.join(SAVE_DIR, "test.jsonl"), test_data)

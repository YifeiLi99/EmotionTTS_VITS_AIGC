import os
import torch
from pathlib import Path

# ========== 基础路径设置 ==========
ROOT = Path(__file__).resolve().parent

# ========== 数据路径 ==========
DATA_ROOT = ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

# 当前使用数据集 (可更改)
NOW_DATASET = RAW_DIR / "EMOVIE_DATASET"
# 处理后的数据集目录 (可更改)
PROCESSED_DATASET_DIR = PROCESSED_DIR / "EMOVIE_DATASET"
PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 原始音频路径
RAW_DATA_DIR = NOW_DATASET / "wavs"
# labels.csv 路径
RAW_LABELS_FILE = NOW_DATASET / "labels.csv"


# ========== 模型输出路径 ==========
WEIGHTS_DIR = ROOT / "weights"  # 训练好的模型保存路径
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)



# ========== 训练日志路径 ==========
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========== 选择模型 ===================
MODEL_TYPE = "full"

# 情感模块
fusion_method = "concat"
text_dim = 256
emotion_dim = 64

# 持续时间预测
duration_predictor_config = {
    "in_channels": 320,  # 等于 text_encoder_dim + emotion_dim（如256+64）
    "filter_channels": 256,
    "kernel_size": 3,
    "dropout": 0.5,
    "num_layers": 2
}

# PosteriorEncoder模块
in_channels = 80  # mel通道数
hidden_channels = 192
latent_dim = 64
kernel_size = 5
num_layers = 6

# ========== 模型训练参数 =================
#设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4

#早停耐心值
PATIENCE = 10

# mel谱参数（如果后续训练VITS用到）
sampling_rate = 24000  # EMOVIE 官方采样率 24kHz
n_mels = 80
hop_length = 256
win_length = 1024

# ========== 情感类别设置 ==========
# 后续可以通过 labels.csv 动态生成 emotion2id.json
emotion_list = [
    "happy", "angry", "sad", "neutral", "fear", "surprise", "disgust"
]
emotion2id = {emo: idx for idx, emo in enumerate(emotion_list)}

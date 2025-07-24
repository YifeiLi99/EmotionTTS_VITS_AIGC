import os
import torch

# ========== 基础路径设置 ==========
ROOT = os.path.abspath(os.path.dirname(__file__))

# 数据路径（根据你的项目结构自定义）
DATA_ROOT = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

RAW_DATA_DIR = os.path.join(RAW_DIR, "EMOVIE DATASET", "wavs")  # 原始音频路径
RAW_LABELS_FILE = os.path.join(RAW_DIR, "EMOVIE DATASET", "labels.csv")  # labels.csv 路径

# 模型输出路径
WEIGHTS_DIR = os.path.join(ROOT, "weights")  # 训练好的模型保存路径
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 训练日志路径
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
# =========== 选择模型 ===================
MODEL_TYPE = "simple"

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

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

#早停耐心值
PATIENCE = 3

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

import os

# ========== 基础路径设置 ==========
project_root = os.path.dirname(os.path.abspath(__file__))

# 数据路径（根据你的项目结构自定义）
data_root = os.path.join(project_root, "data", "emovie")
raw_data_dir = os.path.join(data_root, "wavs")               # 原始音频路径
labels_file = os.path.join(data_root, "labels.csv")          # labels.csv 路径

# 特征提取后保存路径
feature_dir = os.path.join(data_root, "processed")           # mel谱等特征保存路径
os.makedirs(feature_dir, exist_ok=True)

# 模型输出路径
weights_dir = os.path.join(project_root, "weights")          # 训练好的模型保存路径
os.makedirs(weights_dir, exist_ok=True)

# 训练日志路径
tensorboard_log_dir = os.path.join(project_root, "logs")     # tensorboard 日志
os.makedirs(tensorboard_log_dir, exist_ok=True)

# ========== 模型训练参数 ==========

batch_size = 16
learning_rate = 1e-4
epochs = 100

# mel谱参数（如果后续训练VITS用到）
sampling_rate = 24000       # EMOVIE 官方采样率 24kHz
n_mels = 80
hop_length = 256
win_length = 1024

# ========== 情感类别设置 ==========
# 后续可以通过 labels.csv 动态生成 emotion2id.json
emotion_list = [
    "happy", "angry", "sad", "neutral", "fear", "surprise", "disgust"
]
emotion2id = {emo: idx for idx, emo in enumerate(emotion_list)}

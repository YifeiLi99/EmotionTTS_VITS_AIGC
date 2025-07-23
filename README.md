# 🎤 EmotionTTS_VITS_AIGC：基基于VITS的情感可控文本转语音AIGC项目

## 📌 项目简介

本项目基于 **VITS** 模型，构建了一个 **AIGC情感可控语音生成系统**，实现从**文本输入**到**多情感类别的语音输出**，支持完整的
**训练、推理、Gradio可视化交互**。  
本项目为**AIGC方向入门级项目**，雕虫小技不值一提，还望各位海涵。

## 📁 项目结构

```
EmotiSpeech-VITS/
├── config.py                # 配置文件（路径、超参数）
├── data/                    # 数据集目录（含原始及预处理）
├── model/                   # 模型结构模块（VITS+情感嵌入）
├── preprocess_data.py       # 数据预处理脚本
├── train.py                 # 模型训练主流程
├── inference.py             # 推理脚本（单条/批量）
├── gradio_app.py            # Gradio在线推理界面
├── train_log_visualize.py   # 训练日志可视化
├── requirements.txt         # Python依赖
└── README.md                # 项目说明文件
```

## 🎯 项目功能亮点

- ✅ **情感可控TTS生成**：支持输入文本一键生成**不同情感语音**（如快乐、愤怒、悲伤等）
- ✅ **完整训练-推理-部署流程**：
    - 数据预处理 ➡️ 特征提取 ➡️ VITS训练 ➡️ Gradio推理
- ✅ **Gradio交互式界面**：
    - 直接网页上传文本、选择情感，实时试听合成结果
- ✅ **训练日志可视化**：
    - 支持训练Loss曲线、音频对比试听、模型版本管理

## 🚀 快速启动

### 1. 克隆项目

```bash
git clone https://github.com/YifeiLi99/EmotiSpeech-VITS.git
cd EmotiSpeech-VITS
```

### 2. 安装环境

```bash
conda create -n emotienv python=3.10
conda activate emotienv
pip install -r requirements.txt
```

### 3. 数据准备

- 推荐使用公开数据集：`RAVDESS`、`EmoV-DB`、`IEMOCAP`
- 数据准备指南见：`data/README.md`

### 4. 启动训练

```bash
python train.py
```

### 5. 启动Gradio推理

```bash
python gradio_app.py
```

## 📊 训练效果示例

| 情感类别 | 试听样例                |
|------|---------------------|
| 快乐   | ✅ [试听链接/Gradio演示]() |
| 愤怒   | ✅ [试听链接/Gradio演示]() |
| 悲伤   | ✅ [试听链接/Gradio演示]() |

## 📝 后续可改进方向

> 暂无，待续

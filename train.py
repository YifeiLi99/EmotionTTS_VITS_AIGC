import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from data.vits_dataset import VITSEmotionDataset, char_tokenizer
from data.collate_fn import vits_collate_fn
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, LOG_DIR, WEIGHTS_DIR, DEVICE, PROCESSED_DIR

# ======================== 参数设置 ========================
JSONL_PATH = os.path.join(PROCESSED_DIR, "metadata_emovie.jsonl")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, 'best_model.pt')
LOG_PATH = os.path.join(LOG_DIR, 'train_log.txt')
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard001')
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# ======================= 日志 =============================
tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
log_file = open(LOG_PATH, 'w', encoding='utf-8')

# ======================= 数据准备 ============================
dataset = VITSEmotionDataset(jsonl_path=JSONL_PATH, tokenizer=char_tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vits_collate_fn)


# ========== 测试用模型，后续可无视 ==========
class DummyVITS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)  # 占位层

    def forward(self, text, emotion):
        # 模拟一个输出（真实模型需要完整 VITS 构建）
        # 用 text.device 确保生成的张量在同一设备上（无论 CPU 还是 GPU）
        dummy_input = torch.randn(text.shape[0], 100).to(DEVICE)
        return self.linear(dummy_input)


# ======================= 模型准备 ============================
model = DummyVITS()
model.to(DEVICE)

# ======================= 优化器 & Scheduler =================
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# ======================== 早停 ===================


# ========== 训练主循环 ==========
if __name__ == "__main__":
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in dataloader:
            text = batch["text"]
            text.to(DEVICE)
            text_lengths = batch["text_lengths"]
            text_lengths.to(DEVICE)
            emotion = batch["emotion"]
            emotion.to(DEVICE)
            waveform = batch["waveform"]
            waveform.to(DEVICE)
            waveform_lengths = batch["waveform_lengths"]
            waveform_lengths.to(DEVICE)

            # --------- 前向传播 ---------
            outputs = model(text, emotion)  # 占位模型输出

            # --------- 损失函数（占位） ---------
            loss = F.mse_loss(outputs, torch.zeros_like(outputs))  # 假设输出应为 0

            # --------- 反向传播 ---------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --------- 日志记录 ---------
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                tb_writer.add_scalar("Loss/train", loss.item(), step)
            step += 1

        # --------- 保存检查点 ---------
        ckpt_path = os.path.join(WEIGHTS_DIR, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

    tb_writer.close()

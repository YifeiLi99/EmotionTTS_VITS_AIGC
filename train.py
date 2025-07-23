import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from data.vits_dataset import VITSEmotionDataset, char_tokenizer
from data.collate_fn import vits_collate_fn
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, LOG_DIR, WEIGHTS_DIR, DEVICE, PROCESSED_DIR, PATIENCE

# ======================== 参数设置 ========================
TRAIN_JSONL = os.path.join(PROCESSED_DIR, "train.jsonl")
VAL_JSONL = os.path.join(PROCESSED_DIR, "val.jsonl")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, 'best_model.pt')
LOG_PATH = os.path.join(LOG_DIR, 'train_log.txt')
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard001')
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# ======================= 日志 =============================
tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
log_file = open(LOG_PATH, 'w', encoding='utf-8')

# ======================= 数据准备 ============================
train_dataset = VITSEmotionDataset(jsonl_path=TRAIN_JSONL, tokenizer=char_tokenizer)
val_dataset = VITSEmotionDataset(jsonl_path=VAL_JSONL, tokenizer=char_tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vits_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=vits_collate_fn)

# ========== 占位模型 ==========
class DummyVITS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)

    def forward(self, text, emotion):
        dummy_input = torch.randn(text.shape[0], 100).to(text.device)
        return self.linear(dummy_input)

# ======================= EarlyStopping 修正版 ====================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

# ======================= 模型准备 ============================
model = DummyVITS().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
early_stopper = EarlyStopping(patience=PATIENCE)

# ======================= 验证函数 ============================
def evaluate(model, val_loader):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            text = batch["text"].to(DEVICE)
            emotion = batch["emotion"].to(DEVICE)
            outputs = model(text, emotion)
            loss = F.mse_loss(outputs, torch.zeros_like(outputs))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

# ========== 训练主循环 ==========
if __name__ == "__main__":
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            text = batch["text"].to(DEVICE)
            emotion = batch["emotion"].to(DEVICE)
            waveform = batch["waveform"].to(DEVICE)
            text_lengths = batch["text_lengths"].to(DEVICE)
            waveform_lengths = batch["waveform_lengths"].to(DEVICE)

            outputs = model(text, emotion)
            loss = F.mse_loss(outputs, torch.zeros_like(outputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                tb_writer.add_scalar("Loss/train", loss.item(), step)
            step += 1

        # ========== 每轮验证 ==========
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        scheduler.step(val_loss)

        # ========== 保存最优模型 & EarlyStopping ==========
        if early_stopper.step(val_loss):
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"✅ 最优模型已保存: {CHECKPOINT_PATH}")
        else:
            print(f"😴 验证集无提升，EarlyStopping 计数: {early_stopper.counter}/{early_stopper.patience}")

        if early_stopper.should_stop():
            print("⛔ 提前停止：验证集损失连续未提升")
            break

        # 每轮保存一次权重
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"epoch_{epoch}.pth"))

    tb_writer.close()
    log_file.close()

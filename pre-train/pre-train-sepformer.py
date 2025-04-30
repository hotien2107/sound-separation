# Import thư viện
from google.colab import drive
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
import librosa
import pandas as pd
import numpy as np
import os

# Mount Google Drive
drive.mount("/content/drive")
project_path = '/content/drive/MyDrive/Project/SpeechSeparation'
training_csv = os.path.join(project_path, "mix_2_spk_tr_1.csv")
training_df = pd.read_csv(training_csv)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================
# 1. Định nghĩa Dataset
# =====================
class SpeechDataset(Dataset):
    def __init__(self, df, project_path, sr=8000, augment=True):
        self.df = df
        self.project_path = project_path
        self.sr = sr
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Đọc đường dẫn
        audio1_path = os.path.join(self.project_path, self.df['audio1'][idx])
        audio2_path = os.path.join(self.project_path, self.df['audio2'][idx])
        noise1 = self.df['noise1'][idx]
        noise2 = self.df['noise2'][idx]

        # Load audio
        speech1, _ = librosa.load(audio1_path, sr=self.sr)
        speech2, _ = librosa.load(audio2_path, sr=self.sr)
        s1 = torch.tensor(speech1, dtype=torch.float32)
        s2 = torch.tensor(speech2, dtype=torch.float32)

        # Cắt/pad về cùng độ dài
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len].unsqueeze(0)
        s2 = s2[:min_len].unsqueeze(0)

        # Tạo input/output
        input_X = adjust_dB(s1, noise1) + adjust_dB(s2, noise2)
        target_X = torch.cat([s1, s2], dim=0)

        # Augmentation: Thêm nhiễu
        if self.augment:
            input_X = add_noise(input_X, noise_level=0.005)

        return input_X.squeeze(0), target_X

    def add_noise(self, audio, noise_level=0.005):
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    def adjust_dB(self, audio, dB):
        return audio * (10 ** (dB / 20))

# Hàm collate để xử lý độ dài biến đổi
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = pad_sequence(inputs, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)
    return inputs, targets

# =====================
# 2. Định nghĩa Loss
# =====================
class ImprovedSISNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_target):
        # Chuẩn hóa
        y_pred = y_pred - y_pred.mean(dim=-1, keepdim=True)
        y_target = y_target - y_target.mean(dim=-1, keepdim=True)

        # Tính SISNR
        s_target = torch.sum(y_pred * y_target, dim=-1, keepdim=True) * y_target
        s_target /= (torch.sum(y_target ** 2, dim=-1, keepdim=True) + self.eps)
        e_noise = y_pred - s_target
        sisnr = 10 * torch.log10(
            (torch.sum(s_target ** 2, dim=-1) + self.eps) /
            (torch.sum(e_noise ** 2, dim=-1) + self.eps)
        )
        return -sisnr.mean()

# =====================
# 3. Khởi tạo mô hình
# =====================
from speechbrain.inference.separation import SepformerSeparation

model = SepformerSeparation.from_hparams(
    source="speechbrain/resepformer-wsj02mix",
    savedir="pretrained_sepformer"
).to(device)
scaler = GradScaler()
criterion = ImprovedSISNRLoss().to(device)

# =====================
# 4. Thiết lập huấn luyện
# =====================
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# DataLoader
train_dataset = SpeechDataset(training_df, project_path, augment=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

# =====================
# 5. Vòng lặp huấn luyện
# =====================
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (input_X, target_X) in enumerate(train_loader):
        input_X = input_X.unsqueeze(1).to(device)  # [B, 1, T]
        target_X = target_X.to(device)            # [B, 2, T]

        optimizer.zero_grad()

        with autocast():
            # Dự đoán
            est_sources = model(input_X)

            # Tính loss cho từng speaker
            loss1 = criterion(est_sources[:, 0, :], target_X[:, 0, :])
            loss2 = criterion(est_sources[:, 1, :], target_X[:, 1, :])
            loss = (loss1 + loss2) / 2

        # Backpropagation với mixed precision
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # Cập nhật learning rate
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"=== Epoch {epoch} | Average Loss: {avg_loss:.4f} ===")

    # Lưu checkpoint
    torch.save(model.state_dict(), f"sepformer_epoch_{epoch}.pth")
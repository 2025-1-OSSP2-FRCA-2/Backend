"""
Usage:
  # Inference on a single clip
  python daisee_efficientnetv2s.py --inference --video ./testing.avi --clip_len 8 --model_path ./daisee_model.pt
"""

import os
import glob
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import pandas as pd

# Enable cuDNN auto-tuner for fixed-size inputs
torch.backends.cudnn.benchmark = True
# Fixed seed for reproducibility
torch.manual_seed(0)

# Logging configuration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Global transform for all modes
transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),  # PIL Image를 Tensor로 변환
    # transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def find_video_path(data_dir, subset, clip_id):
    stem, _ = os.path.splitext(clip_id)
    pattern = os.path.join(data_dir, subset, '*', stem, f"{stem}.*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def read_labels(csv_file, data_dir=None):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'ClipID':'clip','Boredom':'boredom',
        'Confusion':'confusion','Engagement':'engagement',
        'Frustration':'frustration'
    })
    if data_dir:
        subset = os.path.splitext(os.path.basename(csv_file))[0].replace('Labels','')
        df['video_path'] = df['clip'].astype(str).apply(
            lambda c: find_video_path(data_dir, subset, c)
        )
    else:
        df['video_path'] = df['clip']
    df = df.dropna(subset=['video_path'])
    return df[['video_path','boredom','confusion','engagement','frustration']]


class DAiSEEDataset(Dataset):
    def __init__(self, csv_file, clip_len=8, transform=None, data_dir=None):
        df = read_labels(csv_file, data_dir)
        if df.empty:
            raise ValueError(f"No valid entries in {csv_file}")
        self.clip_len = clip_len
        self.transform = transform
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # ordinal target: for each y in [0,3], vector [y>0, y>1, y>2]
        y = torch.tensor([
            row['boredom'], row['confusion'],
            row['engagement'], row['frustration']
        ], dtype=torch.long)
        targets = (y.unsqueeze(1) > torch.arange(3)).float()  # [4,3]

        video, _, _ = read_video(row['video_path'], pts_unit='sec')  # [T,H,W,C]
        video = video.permute(3, 0, 1, 2)  # [C,T,H,W]
        T = video.size(1)
        if T == 0:
            raise RuntimeError(f"Empty video: {row['video_path']}")
        idxs = torch.linspace(0, T-1, steps=self.clip_len).clamp(0, T-1).long()
        clip = video[:, idxs, :, :]
        if self.transform:
            clip = torch.stack([self.transform(clip[:, t]) for t in range(self.clip_len)], dim=1)
        return clip, targets


class VideoEfficientNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[1].in_features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 4*3)
        )  # 4 dims × 3 thresholds

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        f = self.features(x)
        f = self.pool(f).view(-1, f.size(1))
        f = f.view(B, T, -1).mean(1)
        logits = self.head(f).view(B, 4, 3)
        return logits  # [B,4,3]


def train(train_csv, val_csv, batch_size, clip_len, epochs,
          lr, device, output_model, checkpoint_interval, data_dir):
    train_ds = DAiSEEDataset(train_csv, clip_len, transform, data_dir)
    val_ds = DAiSEEDataset(val_csv, clip_len, transform, data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = VideoEfficientNet(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val = float('inf')

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        total_loss = 0
        for clips, targets in train_loader:
            clips, targets = clips.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * clips.size(0)
        avg_train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        dim_correct = {k:0 for k in ['boredom','confusion','engagement','frustration']}
        total = 0
        with torch.no_grad():
            for clips, targets in val_loader:
                clips, targets = clips.to(device), targets.to(device)
                logits = model(clips)
                loss = criterion(logits, targets)
                total_val_loss += loss.item() * clips.size(0)
                prob = torch.sigmoid(logits)
                preds = prob.gt(0.5).sum(dim=2)  # [B,4]
                gt = targets.sum(dim=2).long()
                correct += (preds == gt).all(dim=1).sum().item()
                for i,k in enumerate(['boredom','confusion','engagement','frustration']):
                    dim_correct[k] += (preds[:,i] == gt[:,i]).sum().item()
                total += clips.size(0)
        avg_val_loss = total_val_loss / len(val_ds)
        val_acc = correct / total

        dim_acc = {k: dim_correct[k]/total for k in dim_correct}
        avg_dim_acc = sum(dim_acc.values()) / 4

        msg = (f"Epoch {epoch}/{epochs} - "
               f"Train Loss: {avg_train_loss:.4f}, "
               f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
               + ", ".join([f"{d} Acc: {dim_acc[d]:.4f}" for d in dim_acc]))
        print(msg)
        logger.info(msg)
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), output_model)
        if epoch % checkpoint_interval == 0:
            ckpt = f"checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt)
    return model


def evaluate(test_csv, model, batch_size, clip_len, device, data_dir):
    ds = DAiSEEDataset(test_csv, clip_len, transform, data_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, targets in loader:
            clips, targets = clips.to(device), targets.to(device)
            logits = model(clips)
            loss = criterion(logits, targets)
            total_loss += loss.item() * clips.size(0)
            prob = torch.sigmoid(logits)
            preds = prob.gt(0.5).sum(dim=2)
            gt = targets.sum(dim=2).long()
            correct += (preds == gt).all(dim=1).sum().item()
            total += clips.size(0)
    return total_loss / len(ds), correct / total


def inference(video_path, model_path, clip_len, device):
    model = VideoEfficientNet(pretrained=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    video, _, _ = read_video(video_path, pts_unit='sec')
    video = video.permute(3,0,1,2)
    T = video.size(1)
    if T == 0:
        raise RuntimeError(f"No frames in {video_path}")
    idxs = torch.linspace(0, T-1, steps=clip_len).clamp(0, T-1).long()
    clip = video[:, idxs, :, :]
    clip = torch.stack([transform(clip[:, t]) for t in range(clip_len)], dim=1)
    clip = clip.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(clip)
        prob = torch.sigmoid(logits)
        preds = prob.gt(0.5).sum(dim=2).squeeze(0).tolist()
    return preds, prob.squeeze(0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--val_csv', type=str)
    parser.add_argument('--test_csv', type=str)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--model_path', type=str, default='daisee_model.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--clip_len', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--video', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    model = None
    if args.train:
        model = train(
            args.train_csv, args.val_csv,
            args.batch_size, args.clip_len,
            args.epochs, args.lr, device,
            args.output_model, args.checkpoint_interval,
            args.data_dir
        )
    if args.evaluate:
        if model is None:
            model = VideoEfficientNet(pretrained=True).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        loss, acc = evaluate(
            args.test_csv, model,
            args.batch_size, args.clip_len,
            device, args.data_dir
        )
        print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")
    if args.inference:
        preds, probs = inference(
            args.video, args.model_path,
            args.clip_len, device
        )
        print("Predicted labels (0-3) per dimension:", preds)
        print("Probabilities per threshold:", probs)

if __name__ == '__main__':
    main()

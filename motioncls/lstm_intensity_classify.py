#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM で (T,263) の可変長系列を 3段階の感情強度（L/M/H）に分類する学習・評価・推論ワンパックスクリプト
- データ：.npy（辞書 or 配列）。辞書の 'label' は無視し、ファイル名末尾の強度サフィックスからラベルを決定。
- ラベル規則： *_L.npy, *_M.npy, *_H.npy のみ学習対象（大小文字は不問）。それ以外は読み飛ばし。
- データ構成： data/train と data/val を推奨。val が無ければ train から層化分割（強度で層化）。
- 推論：1ファイル指定で確率付き予測。

使い方（学習）例:
    python lstm_intensity_classify.py --mode train --data_root data --epochs 20 --batch_size 16

使い方（単発推論）例:
    python lstm_intensity_classify.py --mode infer --ckpt checkpoints/best.pt --input_npy path/to/sample_H.npy
"""

import os
import csv
import re
import math
import random
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

# ========== 0. 定数 ==========
INTENSITY_CLASSES = ["low", "middle", "high"]  # 表示用
LMH2IDX = {"l": 0, "m": 1, "h": 2}
IDX2LMH = {v: k for k, v in LMH2IDX.items()}
IDX2NAME = {0: "low", 1: "middle", 2: "high"}

LMH_SUFFIX_RE = re.compile(r"_([LMH])\.npy$", re.IGNORECASE)

# ======================================= 1. 乱数固定・デバイス =======================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 2. データセット ======================
class IntensitySequenceDataset(Dataset):
    """
    data_root 下の .npy を再帰的に探索し、
    - ファイル名末尾が *_L.npy / *_M.npy / *_H.npy のみ採用（大小文字は不問）
    - 中身は dict({'motion': (T,263)}) でも array((T,263)) でも可。いずれも 'motion' を取り出す。
    - 返り値: (tensor[T,263], T, label_idx, path)
    """
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.samples: List[Tuple[str, int]] = []  # (path, label_idx)

        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if not fn.lower().endswith(".npy"):
                    continue
                m = LMH_SUFFIX_RE.search(fn)
                if not m:
                    # ラベル規則に合わないものは除外
                    continue
                lmh = m.group(1).lower()  # 'l'/'m'/'h'
                label_idx = LMH2IDX.get(lmh, None)
                if label_idx is None:
                    continue
                fpath = os.path.join(dirpath, fn)
                self.samples.append((fpath, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No usable LMH .npy found in: {root_dir} (need *_L/M/H.npy)")

        # 簡単な統計
        stats = { "low": 0, "middle": 0, "high": 0 }
        for _, idx in self.samples:
            stats[IDX2NAME[idx]] += 1
        print("[Dataset] Loaded counts per intensity:", stats)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label_idx = self.samples[idx]
        raw = np.load(fpath, allow_pickle=True)
        # dict か配列かを判定
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
            raw = raw.item()
        if isinstance(raw, dict):
            motion = raw["motion"]
        else:
            motion = raw

        if motion.ndim != 2 or motion.shape[1] != 263:
            raise ValueError(f"Expect motion (T,263), got {motion.shape} at {fpath}")

        T = motion.shape[0]
        x = torch.from_numpy(motion).float()
        y = torch.tensor(label_idx, dtype=torch.long)
        return x, T, y, fpath

def collate_pad(batch):
    xs, lengths, ys, paths = zip(*batch)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [B,Tmax,263]
    lengths = torch.tensor(lengths, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    return x_padded, lengths, ys, paths

# ====================== 3. モデル ======================
class LSTMClassifierPacked(nn.Module):
    def __init__(self, input_size=263, hidden_size=120, num_classes=3,
                 num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=(dropout if num_layers > 1 else 0.0))
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x_padded, lengths):
        packed = pack_padded_sequence(
            x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            last_fwd = h_n[-2]
            last_bwd = h_n[-1]
            last_layer_h = torch.cat([last_fwd, last_bwd], dim=1)
        else:
            last_layer_h = h_n[-1]
        out = self.relu(last_layer_h)
        logits = self.fc(out)
        return logits

    @torch.no_grad()
    def predict_proba(self, x_padded, lengths):
        logits = self.forward(x_padded, lengths)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x_padded, lengths):
        probs = self.predict_proba(x_padded, lengths)
        return probs.argmax(dim=1), probs

# ====================== 4. 学習・評価ループ ======================
def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    return correct / y.size(0)

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    epoch_loss, epoch_acc, n = 0.0, 0.0, 0
    for x_padded, lengths, y, _ in loader:
        x_padded = x_padded.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_padded, lengths)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.size(0)
        epoch_loss += loss.item() * bs
        epoch_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return epoch_loss / n, epoch_acc / n

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss, epoch_acc, n = 0.0, 0.0, 0
    all_y, all_pred = [], []
    for x_padded, lengths, y, _ in loader:
        x_padded = x_padded.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        logits = model(x_padded, lengths)
        loss = criterion(logits, y)

        bs = y.size(0)
        epoch_loss += loss.item() * bs
        epoch_acc += accuracy_from_logits(logits, y) * bs
        n += bs

        all_y.append(y.cpu())
        all_pred.append(logits.argmax(dim=1).cpu())
    if n == 0:
        return float("nan"), float("nan"), None
    y_cat = torch.cat(all_y)
    pred_cat = torch.cat(all_pred)
    return epoch_loss / n, epoch_acc / n, (y_cat.numpy(), pred_cat.numpy())

def stratified_split(items: List[Tuple[str, int]], val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for p, idx in items:
        by_label.setdefault(idx, []).append((p, idx))
    train_list, val_list = [], []
    for idx, lst in by_label.items():
        random.shuffle(lst)
        k = max(1, int(len(lst) * val_ratio))
        val_list += lst[:k]
        train_list += lst[k:]
    return train_list, val_list

def make_loaders(data_root: str, batch_size: int, num_workers: int,
                 auto_split_if_no_val: bool, seed: int):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    def _subds(items):
        class _Sub(Dataset):
            def __init__(self, items): self.items = items
            def __len__(self): return len(self.items)
            def __getitem__(self, i):
                path, label_idx = self.items[i]
                raw = np.load(path, allow_pickle=True)
                if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
                    raw = raw.item()
                motion = raw["motion"] if isinstance(raw, dict) else raw
                if motion.ndim != 2 or motion.shape[1] != 263:
                    raise ValueError(f"Expect motion (T,263), got {motion.shape} at {path}")
                x = torch.from_numpy(motion).float()
                y = torch.tensor(label_idx, dtype=torch.long)
                T = motion.shape[0]
                return x, T, y, path
        return _Sub(items)

    # val ディレクトリが存在し、かつ中にLMH対応ファイルがある場合はそれを使う
    if os.path.isdir(val_dir) and any(LMH_SUFFIX_RE.search(fn or "") for fn in os.listdir(val_dir)):
        train_ds = IntensitySequenceDataset(train_dir)
        val_ds   = IntensitySequenceDataset(val_dir)
    else:
        full_ds = IntensitySequenceDataset(train_dir)
        # 内部サンプル（path, idx）を層化分割
        train_items, val_items = stratified_split(full_ds.samples, val_ratio=0.2, seed=seed)
        train_ds = _subds(train_items)
        val_ds   = _subds(val_items)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad
    )
    return train_loader, val_loader

# ====================== 5. メイン（学習 / 推論） ======================
def train_main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Info] device: {device}")

    train_loader, val_loader = make_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        auto_split_if_no_val=True,
        seed=args.seed
    )

    model = LSTMClassifierPacked(
        input_size=263, hidden_size=args.hidden_size, num_classes=3,
        num_layers=args.num_layers, bidirectional=args.bidirectional, dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience,
        threshold=args.min_delta, verbose=True
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_acc = float("-inf")
    best_path = os.path.join(args.ckpt_dir, getattr(args, "ckpt_name", "best.pt"))

    no_improve = 0
    patience = args.early_stop_patience
    min_delta = args.min_delta

    log_path = os.path.join(args.ckpt_dir, args.log_file)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0)
        va_loss, va_acc, _ = eval_one_epoch(model, val_loader, criterion, device)
        if not math.isnan(va_acc):
            scheduler.step(va_acc)

        lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  lr={lr:.2e}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc, lr])

        improved = (va_acc is not None) and (va_acc > best_val_acc + min_delta)
        if improved:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "config": vars(args),
                        "classes": INTENSITY_CLASSES}, best_path)
            print(f"  -> Saved best to: {best_path} (val_acc={best_val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"[EarlyStopping] val_accが {patience} エポック改善しなかったため打ち切り")
            break

    print(f"[Done] Best val_acc = {best_val_acc:.4f}. Checkpoint: {best_path}")
    print(f"[Log] 学習ログを保存しました: {log_path}")

def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = LSTMClassifierPacked(
        input_size=263,
        hidden_size=cfg.get("hidden_size", 120),
        num_classes=3,
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
        dropout=cfg.get("dropout", 0.0)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

@torch.no_grad()
def infer_one(args):
    device = get_device()
    model = load_model(args.ckpt, device)

    # 入力読み込み
    raw = np.load(args.input_npy, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        raw = raw.item()
    motion = raw["motion"] if isinstance(raw, dict) else raw

    # 参照ラベル（任意）：ファイル名末尾の L/M/H
    ref_label = None
    m = LMH_SUFFIX_RE.search(os.path.basename(args.input_npy))
    if m:
        ref_label = m.group(1).lower()  # 'l'/'m'/'h'

    x = torch.from_numpy(motion).float().unsqueeze(0)  # [1,T,263]
    lengths = torch.tensor([motion.shape[0]], dtype=torch.long)
    x = x.to(device)
    lengths = lengths.to(device)

    pred_idx, probs = model.predict(x, lengths)
    pred_idx = pred_idx.item()
    probs = probs.squeeze(0).cpu().numpy()

    print("=== Inference Result ===")
    if ref_label is not None:
        print(f"  (Ref) label from filename: {ref_label.upper()} ({IDX2NAME[LMH2IDX[ref_label]]})")
    print(f"  Pred: {IDX2NAME[pred_idx]} ({IDX2LMH[pred_idx].upper()})")
    for i, p in enumerate(probs):
        print(f"    {IDX2NAME[i]:>6s}: {p*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)

    # data / train
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)  # Windows/WSLは0推奨

    # model
    parser.add_argument("--hidden_size", type=int, default=120)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)

    # optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    # ckpt
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt", type=str, help="path to checkpoint for infer")
    parser.add_argument("--ckpt_name", type=str, default="best.pt",
                        help="checkpoint filename (inside ckpt_dir)")

    # infer
    parser.add_argument("--input_npy", type=str, help="one .npy to run inference")

    # 早期終了＆LRスケジューラ
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="val_accの改善がこのエポック数連続で止まったら学習を打ち切る")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="改善とみなす最小差分（val_accの増加量）")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="ReduceLROnPlateauのpatience（改善が止まったらLRを下げるまでの猶予）")
    parser.add_argument("--log_file", type=str, default="train_log.csv",
                        help="学習ログを保存するCSVファイル名")

    args = parser.parse_args()

    if args.mode == "train":
        train_main(args)
    else:
        if not args.ckpt or not args.input_npy:
            raise SystemExit("--mode infer requires --ckpt and --input_npy")
        infer_one(args)

if __name__ == "__main__":
    main()

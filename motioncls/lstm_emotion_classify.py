#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM で (T,263) の可変長系列を 6感情に分類する学習・評価・推論ワンパックスクリプト
- データ：.npy（辞書 or 配列）。辞書なら 'motion' (T,263) と 'label' (文字列) を推奨。
- ラベル：sadness, anger, joy, fear, shame, pride の6種類のみ学習対象。
- ラベルが無い場合はファイル名から推定（文字列一致）。
- 学習：train/ と val/ が無ければ train から自動分割（層化）も可。
- 推論：1ファイルだけ渡して予測（確率付き）も可。

使い方（学習）例:
    python lstm_emotion_classify.py --mode train --data_root data --epochs 20 --batch_size 16

使い方（単発推論）例:
    python lstm_emotion_classify.py --mode infer --ckpt checkpoints/best.pt --input_npy path/to/sample.npy
"""

import os
import csv
import re
import math
import json
import random
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ==========
# 0. 定数
# ==========
EMOTIONS = ["sadness", "anger", "joy", "fear", "shame", "pride"]
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMO = {i: e for e, i in EMO2IDX.items()}


# =======================================
# 1. ユーティリティ（乱数固定・デバイス）
# =======================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# 2. データセット定義
# ======================
class EmotionSequenceDataset(Dataset):
    """
    data_root 下の .npy を再帰的に探索し、
    - dict: {'motion': (T,263) np.ndarray, 'label': 'anger'} を優先
    - array: (T,263) のみなら、ファイル名からラベルを抽出
    で (tensor[T,263], T, label_idx) を返す。
    """

    def __init__(self, root_dir: str, allow_missing_label_from_name: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.samples = []  # list of tuples: (npy_path, label_str)

        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if not fn.lower().endswith(".npy"):
                    continue
                fpath = os.path.join(dirpath, fn)
                # まず辞書を見て label があればそれを採用
                label_from_dict = self._peek_label_if_exists(fpath)
                if label_from_dict is not None:
                    if label_from_dict in EMO2IDX:
                        self.samples.append((fpath, label_from_dict))
                    # 対象外ラベルはスキップ
                    continue

                # 辞書にラベルが無ければ、ファイル名から抽出（例: JP_06_anger_3_M.npy）
                if allow_missing_label_from_name:
                    label_from_name = self._label_from_filename(fn)
                    if label_from_name in EMO2IDX:
                        self.samples.append((fpath, label_from_name))
                # それでも見つからなければスキップ

        if len(self.samples) == 0:
            raise RuntimeError(f"No usable .npy found in: {root_dir}")

        # 簡単な統計
        stats = {}
        for _, lab in self.samples:
            stats[lab] = stats.get(lab, 0) + 1
        print("[Dataset] Loaded counts per label:", stats)

    def _peek_label_if_exists(self, npy_path: str) -> Optional[str]:
        try:
            obj = np.load(npy_path, allow_pickle=True)
        except Exception:
            return None
        # dict のとき
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            # np.save(dict) だと object 配列で1要素になることがある
            obj = obj.item()
        if isinstance(obj, dict):
            label = obj.get("label", None)
            if isinstance(label, (str, bytes)):
                label = label.decode() if isinstance(label, bytes) else label
                label = label.strip().lower()
                return label
        return None

    def _label_from_filename(self, filename: str) -> Optional[str]:
        low = filename.lower()
        # ファイル名に emotions のいずれかが含まれていれば採用
        for emo in EMOTIONS:
            if emo in low:
                return emo
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label_str = self.samples[idx]
        raw = np.load(fpath, allow_pickle=True)

        # dict か配列かを判定して motion を取り出す
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
            raw = raw.item()
        if isinstance(raw, dict):
            motion = raw["motion"]
        else:
            motion = raw  # 2D array を想定

        # shape チェック
        if motion.ndim != 2 or motion.shape[1] != 263:
            raise ValueError(f"Expect motion (T,263), got {motion.shape} at {fpath}")

        T = motion.shape[0]
        x = torch.from_numpy(motion).float()  # [T,263]
        y = torch.tensor(EMO2IDX[label_str], dtype=torch.long)
        return x, T, y, fpath


def collate_pad(batch):
    """
    batch: list of (x[T,263], T, y, path)
    → (x_padded[B, Tmax, 263], lengths[B], y[B], paths)
    ※ forward 側でソートするので、ここではソートしません
    """
    xs, lengths, ys, paths = zip(*batch)
    # pad_sequence は [T,feat] のリスト → [Tmax,B,feat]（batch_first=False）
    # 今回は batch_first=True にしたいので convert
    # 一度 [Tmax, B, 263] で出してから permute でも良いが、
    # batch_first=True を直接使います。
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [B,Tmax,263]
    lengths = torch.tensor(lengths, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    return x_padded, lengths, ys, paths


# ======================
# 3. モデル定義
# ======================
class LSTMClassifierPacked(nn.Module):
    """
    ユーザー提示の実装を 263入力/6クラスに合わせて調整。
    - forward は生 logits を返す（学習で CrossEntropyLoss を使うため）
    - 予測時は predict_proba / predict を利用
    """

    def __init__(self, input_size=263, hidden_size=120, num_classes=6,
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
        # （ソートはどちらでも）pack時に enforce_sorted=False を使うと楽です
        packed = pack_padded_sequence(
            x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: [num_layers * num_directions, B, hidden_size]

        if self.lstm.bidirectional:
            # 最終層の forward/backward を取り出して連結
            last_fwd = h_n[-2]  # [B, H]
            last_bwd = h_n[-1]  # [B, H]
            last_layer_h = torch.cat([last_fwd, last_bwd], dim=1)  # [B, 2H]
        else:
            last_layer_h = h_n[-1]  # [B, H]

        out = self.relu(last_layer_h)
        logits = self.fc(out)  # 単方向: [B,H]→[B,C], 双方向: [B,2H]→[B,C]
        return logits

    @torch.no_grad()
    def predict_proba(self, x_padded, lengths):
        logits = self.forward(x_padded, lengths)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x_padded, lengths):
        probs = self.predict_proba(x_padded, lengths)
        return probs.argmax(dim=1), probs


# ======================
# 4. 学習・評価ループ
# ======================
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


def stratified_split(paths_labels: List[Tuple[str, str]],
                     val_ratio: float = 0.2,
                     seed: int = 42):
    random.seed(seed)
    by_label: Dict[str, List[Tuple[str, str]]] = {}
    for p, lab in paths_labels:
        by_label.setdefault(lab, []).append((p, lab))
    train_list, val_list = [], []
    for lab, lst in by_label.items():
        random.shuffle(lst)
        k = max(1, int(len(lst) * val_ratio))
        val_list += lst[:k]
        train_list += lst[k:]
    return train_list, val_list


def make_loaders(data_root: str, batch_size: int, num_workers: int,
                 auto_split_if_no_val: bool, seed: int):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if os.path.isdir(val_dir) and any(fn.endswith(".npy") for fn in os.listdir(val_dir)):
        train_ds = EmotionSequenceDataset(train_dir)
        val_ds = EmotionSequenceDataset(val_dir)
    else:
        # train の中だけに .npy を入れておき、自動で層化分割
        full_ds = EmotionSequenceDataset(train_dir)
        # 内部のサンプル一覧を層化分割
        train_items, val_items = stratified_split(full_ds.samples, val_ratio=0.2, seed=seed)

        # 分割用の薄いラッパーDataset
        class _SubDS(Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                path, lab = self.items[i]
                # 元クラスの __getitem__ 相当
                raw = np.load(path, allow_pickle=True)
                if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
                    raw = raw.item()
                if isinstance(raw, dict):
                    motion = raw["motion"]
                else:
                    motion = raw
                x = torch.from_numpy(motion).float()
                y = torch.tensor(EMO2IDX[lab], dtype=torch.long)
                T = motion.shape[0]
                return x, T, y, path

        train_ds = _SubDS(train_items)
        val_ds = _SubDS(val_items)

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


# ======================
# 5. メイン（学習 / 推論）
# ======================
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
        input_size=263, hidden_size=args.hidden_size, num_classes=len(EMOTIONS),
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

    # === ログファイル準備 ===
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

        # === ログ追記 ===
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc, lr])

        # 改善判定と保存
        improved = (va_acc is not None) and (va_acc > best_val_acc + min_delta)
        if improved:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "config": vars(args),
                        "classes": EMOTIONS}, best_path)
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
        num_classes=len(EMOTIONS),
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
        dropout=cfg.get("dropout", 0.0)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# === 推論用ヘルパー（infer_one / infer_dir から呼ぶ） ===
def _extract_label_from_obj_or_name(obj, npy_path: str) -> Optional[str]:
    label_str = None
    if isinstance(obj, dict):
        label_str = obj.get("label", None)
        if isinstance(label_str, bytes):
            label_str = label_str.decode()
        if isinstance(label_str, str):
            label_str = label_str.strip().lower()
    if not label_str:
        low = os.path.basename(npy_path).lower()
        for emo in EMOTIONS:
            if emo in low:
                label_str = emo
                break
    return label_str


def _load_motion_and_label(npy_path: str):
    raw = np.load(npy_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        raw = raw.item()
    if isinstance(raw, dict):
        motion = raw["motion"]
        label_str = _extract_label_from_obj_or_name(raw, npy_path)
    else:
        motion = raw
        label_str = _extract_label_from_obj_or_name({}, npy_path)
    return motion, label_str


@torch.no_grad()
def infer_one(args):
    device = get_device()
    model = load_model(args.ckpt, device)

    # 1ファイル読み込み
    raw = np.load(args.input_npy, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        raw = raw.item()
    if isinstance(raw, dict):
        motion = raw["motion"]
        label_str = raw.get("label", None)
    else:
        motion = raw
        label_str = None

    if label_str is None:
        # ファイル名から推定だけしておく（実ラベルが分かっていれば比較用）
        label_str = None
        low = os.path.basename(args.input_npy).lower()
        for emo in EMOTIONS:
            if emo in low:
                label_str = emo
                break

    x = torch.from_numpy(motion).float().unsqueeze(0)  # [1,T,263]
    lengths = torch.tensor([motion.shape[0]], dtype=torch.long)
    x = x.to(device)
    lengths = lengths.to(device)

    pred_idx, probs = model.predict(x, lengths)
    pred_idx = pred_idx.item()
    probs = probs.squeeze(0).cpu().numpy()

    print("=== Inference Result ===")
    if label_str is not None:
        print(f"  (Ref) label from file: {label_str}")
    print(f"  Pred: {IDX2EMO[pred_idx]}")
    for i, p in enumerate(probs):
        print(f"    {IDX2EMO[i]:>7s}: {p * 100:.2f}%")


# 追加: ディレクトリ一括推論
@torch.no_grad()
def infer_dir(args):
    device = get_device()
    model = load_model(args.ckpt, device)

    # 対象ファイル列挙
    npy_paths = []
    for dp, _, fns in os.walk(args.input_dir):
        for fn in fns:
            if fn.lower().endswith(".npy"):
                npy_paths.append(os.path.join(dp, fn))
    if len(npy_paths) == 0:
        print(f"[InferDir] No .npy found in: {args.input_dir}")
        return

    results = []
    n_cls = len(EMOTIONS)
    conf = np.zeros((n_cls, n_cls), dtype=int)
    correct_per = {e: 0 for e in EMOTIONS}
    total_per = {e: 0 for e in EMOTIONS}
    overall_correct = 0
    overall_total = 0

    for p in sorted(npy_paths):
        try:
            motion, label_str = _load_motion_and_label(p)
        except Exception as e:
            print(f"[Warn] Failed to load {p}: {e}")
            continue

        x = torch.from_numpy(motion).float().unsqueeze(0).to(device)  # [1,T,263]
        lengths = torch.tensor([motion.shape[0]], dtype=torch.long).to(device)
        pred_idx, probs = model.predict(x, lengths)
        pred_idx = int(pred_idx.item())
        pred_label = IDX2EMO[pred_idx]
        prob_vec = probs.squeeze(0).cpu().numpy()

        is_correct = None
        if label_str in EMO2IDX:
            true_idx = EMO2IDX[label_str]
            total_per[label_str] += 1
            conf[true_idx, pred_idx] += 1
            if pred_idx == true_idx:
                correct_per[label_str] += 1
                overall_correct += 1
            overall_total += 1
            is_correct = (pred_idx == true_idx)

        row = {
            "path": p,
            "true_label": label_str or "",
            "pred_label": pred_label,
            "pred_prob": float(prob_vec[pred_idx]),
            "correct": is_correct,
        }
        # 各クラス確率も展開
        for i, emo in enumerate(EMOTIONS):
            row[f"prob_{emo}"] = float(prob_vec[i])
        results.append(row)

    # サマリ出力
    print("=== Directory Inference Summary ===")
    print(f"Total files found: {len(npy_paths)}")
    if overall_total > 0:
        overall_acc = overall_correct / overall_total
        print(f"Evaluated (with label): {overall_total}")
        print(f"Overall accuracy: {overall_acc:.4f} ({overall_correct}/{overall_total})")
        print("Per-label accuracy:")
        for emo in EMOTIONS:
            t = total_per[emo]
            if t > 0:
                acc = correct_per[emo] / t
                print(f"  {emo:7s}: {acc:.4f} ({correct_per[emo]}/{t})")
        print("Confusion matrix (rows=true, cols=pred):")
        print(conf)
    else:
        print("No usable labels found; accuracy not computed (results CSV only).")

    # CSV 保存
    if args.save_csv:
        keys = list(results[0].keys()) if results else []
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"[Saved] {args.save_csv}")


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

    # --- 既存の parser 定義に追記 ---
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="val_accの改善がこのエポック数連続で止まったら学習を打ち切る")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="改善とみなす最小差分（val_accの増加量）")

    # （任意）学習が停滞したら学習率を下げるスケジューラも使いたい場合
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="ReduceLROnPlateauのpatience（改善が止まったらLRを下げるまでの猶予）")
    parser.add_argument("--log_file", type=str, default="train_log.csv",
                        help="学習ログを保存するCSVファイル名")

    # 変更: argparse にオプション追加
    parser.add_argument("--input_dir", type=str, help="run inference for all .npy files under this directory")
    parser.add_argument("--save_csv", type=str, default=None, help="path to save per-file predictions as CSV")

    args = parser.parse_args()

    # 変更: main() の infer 分岐を少し拡張
    if args.mode == "train":
        train_main(args)
    else:
        if not args.ckpt:
            raise SystemExit("--mode infer requires --ckpt")
        if args.input_npy and args.input_dir:
            raise SystemExit("Use either --input_npy or --input_dir (not both)")
        if args.input_npy:
            infer_one(args)
        elif args.input_dir:
            infer_dir(args)
        else:
            raise SystemExit("--mode infer requires --input_npy or --input_dir")


if __name__ == "__main__":
    main()

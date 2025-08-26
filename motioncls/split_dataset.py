#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train/ にある .npy を train/ val/ test/ にランダム分割するスクリプト
PyCharm などでそのまま実行できます。

- data_root はここで固定してあります
- デフォルトでは move（移動）です。元の train/ にファイルは残りません。
  コピーしたい場合は copy=True にしてください。
"""

import os
import random
import shutil

def split_dataset(data_root: str,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42,
                  copy: bool = False):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"
    random.seed(seed)

    # 元のtrainフォルダ
    src_dir = os.path.join(data_root, "train")
    all_files = [f for f in os.listdir(src_dir) if f.endswith(".npy")]
    random.shuffle(all_files)

    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val

    print(f"[Info] total={n_total} → train={n_train}, val={n_val}, test={n_test}")

    # 出力フォルダ
    out_train = os.path.join(data_root, "train")
    out_val   = os.path.join(data_root, "val")
    out_test  = os.path.join(data_root, "test")
    for d in [out_train, out_val, out_test]:
        os.makedirs(d, exist_ok=True)

    # 割り当て
    subsets = {
        out_train: all_files[:n_train],
        out_val:   all_files[n_train:n_train+n_val],
        out_test:  all_files[n_train+n_val:],
    }

    for out_dir, files in subsets.items():
        for fname in files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(out_dir, fname)
            if copy:
                shutil.copy(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

    print("[Done] Splitting complete.")

# =========================
# 実行ブロック（ここを書き換えて設定）
# =========================
if __name__ == "__main__":
    # ★ここを自分の環境に合わせて変更してください
    DATA_ROOT = "data"       # data/train/ にファイルが入っている想定
    TRAIN_RATIO = 0.7
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15
    SEED        = 42
    COPY        = False       # True にするとコピーで分ける

    split_dataset(DATA_ROOT, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED, COPY)

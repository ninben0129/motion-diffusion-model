import os
import numpy as np

# ラベル候補リスト
motionList = [
    "neutral","anger","contempt","disgust","fear","gratitude","guilt",
    "jealousy","joy", "pride","sadness","shame","surprise"
]

# 入力フォルダ（学習用の .npy がある場所）
input_dir = "./train_npy"       # ★適宜変更してください
# 出力フォルダ（新しい形式で保存する場所）
output_dir = "./train_npy_labeled"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".npy"):
        continue

    fpath = os.path.join(input_dir, fname)
    data = np.load(fpath)

    # label 判定（ファイル名に含まれるかどうかをチェック）
    label = "neutral"
    lower_fname = fname.lower()
    for lbl in motionList:
        if lbl in lower_fname:
            label = lbl
            break

    # 保存形式を辞書にする
    payload = {
        "motion": data.astype(np.float32),  # もとの配列 (T,263)
        "label": label
    }

    save_path = os.path.join(output_dir, fname)
    np.save(save_path, payload)
    print(f"Saved: {save_path}, label={label}, shape={data.shape}")

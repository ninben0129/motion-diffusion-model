import random

# 感情カテゴリ
emotions = [
    "joy", "sadness", "anger", "surprise", "fear", "disgust",
    "contempt", "gratitude", "guilt", "jealousy", "shame", "pride"
]

# ファイルから読み込み（空行除去）
with open("HumanML3D_0524_person/test.txt", "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

# 感情ごとにランダム選択
selected = {}
for emotion in emotions:
    matched = [name for name in filenames if emotion in name.lower()]
    if len(matched) >= 3:
        selected[emotion] = random.sample(matched, 3)
    else:
        selected[emotion] = matched  # 3つ未満ならそのまま

# 出力を userstudy_list.txt に書き込む
with open("userstudy_list.txt", "w", encoding="utf-8") as f:
    for emotion in emotions:
        f.write(f"{emotion}:\n")
        for name in selected[emotion]:
            f.write(f"{name}\n")
        f.write("\n")

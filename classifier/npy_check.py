import numpy as np

# 調べたいnpyファイルのパス
# file_path = 'results.npy'
file_path = 'joint_vec2.npy'

# npyファイルの読み込み
data = np.load(file_path, allow_pickle=True)  # allow_pickle=True は必要に応じて

# 中身の基本情報を表示
print("Type:", type(data))
if isinstance(data, np.ndarray):
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)

# 要素の一部を表示（大きすぎる場合のために一部だけ）
print("\nSample content:")
if data.ndim == 0:
    print(data.item())
else:
    print(data[:1])

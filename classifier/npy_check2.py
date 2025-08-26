import numpy as np

# file_path = 'results.npy'
file_path = 'gen263D_2.npy'
data = np.load(file_path, allow_pickle=True).item()  # .item()でdictとして取り出す

motion = data['motion']
print("motion shape:", motion.shape)

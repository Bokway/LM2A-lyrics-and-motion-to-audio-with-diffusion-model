import numpy as np
import matplotlib.pyplot as plt

# 要查看哪个 npz 就填哪个
# Fill in the path of the npz file you want to inspect

path = '/mnt/mydev2/Bob/LM2ANew/testnpz/sample_00000062.npz'  # change this to your desired npz file

data = np.load(path, allow_pickle=True)

print(" Loaded:", path)
print()

# 打印所有 key
print("Keys:")
for k in data.files:
    print(" -", k)
print()

# 读取内容
mel = data["mel"]
motion = data["motion"]
lyrics = data["lyrics"]
sr = data["sr"]
hop = data["hop_length"]

print("mel shape:", mel.shape)          # (80, T_mel) (80,516)
print("motion shape:", motion.shape)    # (T_motion, D_motion) (180,234)
print("lyrics shape:", lyrics.shape)    # (T_lyrics, D_lyrics) (180,768)
print("sample rate:", sr)       # 22050
print("hop_length:", hop)       #256
print()

# ---- 可视化 mel spectrogram ----
plt.figure(figsize=(10, 4))
mel2 = np.squeeze(mel)   # 移除尺寸为 1 的 batch 维
plt.imshow(mel2, aspect="auto", origin="lower")
plt.title("Mel Spectrogram")
plt.xlabel("Frames")
plt.ylabel("Mel bins")
plt.colorbar()
plt.show()

# ---- 简单检查 mel ----
print("mel min/max:", np.min(mel), np.max(mel))


# ---- 简单检查 motion ----
print("motion first row (前10个值):", motion[0][:10])
print("motion mean:", motion.mean(), "std:", motion.std())
print()

# ---- 简单检查 lyrics embedding ----
print("lyrics embedding mean:", lyrics.mean(), "std:", lyrics.std())
print("lyrics embedding first row (前10个):", lyrics[0][:10])


"""
mel：
使用的是 BigVGAN 的标准 hop = 256
采样率 = 22050
所以：
mel FPS ≈ 22050 / 256 = 86.13 帧每秒
那：
6 秒 × 86.1328fps ≈ 516 帧
就正好是最终得到的 516。

motion：
motion 的 FPS = 30 帧每秒
所以：
6 秒 × 30fps = 180 帧

lyrics embedding：
lyrics embedding 也是和 motion 对齐的，所以也是 180 帧。

"""
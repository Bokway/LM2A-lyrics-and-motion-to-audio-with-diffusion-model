import matplotlib.pyplot as plt
import json
import numpy as np

# 读取你的JSON结果文件
with open("D:\\lm2d\\results\\evaluation_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 提取所有样本的CLAP值
# 提取所有样本的Beat F1值
beat_f1_vals = [v["beat_f1"] for v in results["per_sample_metrics"].values() if v["beat_f1"] is not None]

# 绘制直方图+均值线
plt.figure(figsize=(8, 4))
plt.hist(beat_f1_vals, bins=15, color="#2196F3", alpha=0.7, edgecolor="black")
plt.axvline(x=np.mean(beat_f1_vals), color="red", linestyle="--", label=f"Mean F1: {np.mean(beat_f1_vals):.4f}")
plt.title("Distribution of Beat F1 Score (Motion Rhythm Alignment)")
plt.xlabel("Beat F1 Score")
plt.ylabel("Number of Samples")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.savefig("beat_f1_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 提取所有样本的Beat F1值
clap_vals = [v["cosine_similarity"] for v in results["per_sample_metrics"].values() if v["cosine_similarity"] is not None]
clap_mean = np.mean(clap_vals)

# 绘制直方图（聚焦高值区间，更美观）
plt.figure(figsize=(7, 4))
# 设置bins=20，让分布更细腻；颜色选柔和的绿色，符合学术风格
n, bins, patches = plt.hist(clap_vals, bins=20, color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=0.5)

# 画均值线+标注
plt.axvline(x=clap_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {clap_mean:.4f}")

# 调整坐标轴，聚焦0.9-1.0区间（突出高分）
#plt.xlim(0.9, 1.0)
#plt.ylim(0, max(n)+2)  # y轴自适应，避免顶格

# 标题和标签（清晰体现评估维度）
plt.title("Distribution of CLAP Cosine Similarity (Lyrics Semantic Matching)", fontsize=11)
plt.xlabel("CLAP Cosine Similarity", fontsize=10)
plt.ylabel("Number of Samples", fontsize=10)

# 图例+网格（提升可读性）
plt.legend(fontsize=9)
plt.grid(axis="y", alpha=0.3, linestyle=":")

# 保存（高分辨率，避免裁剪）
plt.tight_layout()
plt.savefig("clap_histogram.png", dpi=300, bbox_inches="tight")
plt.close()



acoustic_vals = [v["acoustic_similarity"] for v in results["per_sample_metrics"].values() if v["acoustic_similarity"] is not None]
acoustic_mean = np.mean(acoustic_vals)

plt.figure(figsize=(7, 4))
n, bins, patches = plt.hist(acoustic_vals, bins=20, color="#FF9800", alpha=0.8, edgecolor="black", linewidth=0.5)
plt.axvline(x=acoustic_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {acoustic_mean:.4f}")

# 适配声学相似度的高值（≈0.9682），x轴范围0.8-1.0
#plt.xlim(0.8, 1.0)
#plt.ylim(0, max(n)+2)

plt.title("Distribution of Acoustic Similarity (MFCC Feature)")
plt.xlabel("Acoustic Cosine Similarity (MFCC)")
plt.ylabel("Number of Samples")
plt.legend(fontsize=9)
plt.grid(axis="y", alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig("acoustic_histogram.png", dpi=300, bbox_inches="tight")
plt.close()
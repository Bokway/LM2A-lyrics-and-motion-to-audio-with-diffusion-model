import torch
import laion_clap
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

class CLAPEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 初始化 LAION-CLAP 模型
        # 默认会下载 630k 样本预训练的版本，这是目前最稳健的版本
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt() 
        self.model.to(device)
        self.model.eval()
        self.device = device

    def get_embeddings(self, file_paths):
        with torch.no_grad():
            embeddings = self.model.get_audio_embedding_from_filelist(x=file_paths)
        
        # 增加健壮性检查
        if hasattr(embeddings, 'cpu'):
            return embeddings.cpu().numpy()
        return embeddings  # 如果已经是 numpy 数组则直接返回

    def compute_metrics(self, gt_files, gen_files):
        """
        计算成对的余弦相似度
        """
        print(f"Extracting embeddings for {len(gt_files)} pairs...")
        
        # 提取 GT 和 Generated 的特征向量
        gt_embeds = self.get_embeddings(gt_files)
        gen_embeds = self.get_embeddings(gen_files)
        
        sims = []
        for i in range(len(gt_embeds)):
            # 1.0 - cosine_distance = cosine_similarity
            sim = 1.0 - cosine(gt_embeds[i], gen_embeds[i])
            sims.append(sim)
        
        sims = np.array(sims)
        return {
            "per_sample": sims,
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims))
        }

"""
# 使用示例
if __name__ == "__main__":
    # 假设你已经准备好了文件路径列表
    # gt_list = ["path/to/gt1.wav", "path/to/gt2.wav", ...]
    # gen_list = ["path/to/gen1.wav", "path/to/gen2.wav", ...]
    
    evaluator = CLAPEvaluator()
    results = evaluator.compute_metrics(gt_list, gen_list)
    print(f"Real CLAP Mean Similarity: {results['mean']:.4f}")
"""


if __name__ == "__main__":
    print("clap module loaded")

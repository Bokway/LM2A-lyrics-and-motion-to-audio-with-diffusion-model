import numpy as np
import librosa
from sklearn.cluster import KMeans
from scipy.stats import norm


def _load_audio(path, sr=22050):
	y, sr = librosa.load(path, sr=sr, mono=True)
	return y, sr


def _embed_mfcc(path, sr=22050, n_mfcc=40):
	y, sr = _load_audio(path, sr=sr)
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
	return mfcc.mean(axis=1)


def compute_ndb(gt_files, gen_files, K=50, embed_fn=None, alpha=0.05, sr=22050):
	"""
	Compute Number of Statistically Different Bins (NDB).

	Procedure (approximate):
	- Fit KMeans with K clusters on GT embeddings (bins)
	- Assign both GT and GEN embeddings to bins
	- For each bin, perform a two-proportion z-test comparing GEN count vs GT proportion
	- Count bins with significant difference after Bonferroni correction

	Returns: dict with NDB count, p-values, counts, cluster_centers
	"""
	if embed_fn is None:
		embed_fn = lambda p: _embed_mfcc(p, sr=sr)

	gt_embs = [np.asarray(embed_fn(p), dtype=np.float64) for p in gt_files]
	gen_embs = [np.asarray(embed_fn(p), dtype=np.float64) for p in gen_files]

	gt_embs = np.stack(gt_embs, axis=0)
	gen_embs = np.stack(gen_embs, axis=0)

	n_gt = gt_embs.shape[0]
	n_gen = gen_embs.shape[0]
	K_use = min(K, n_gt)
	kmeans = KMeans(n_clusters=K_use, random_state=0).fit(gt_embs)
	centers = kmeans.cluster_centers_
	gt_assign = kmeans.predict(gt_embs)
	gen_assign = kmeans.predict(gen_embs)

	counts_gt = np.bincount(gt_assign, minlength=K_use)
	counts_gen = np.bincount(gen_assign, minlength=K_use)

	p_gt = counts_gt / float(n_gt)
	p_gen = counts_gen / float(n_gen)

	pvals = np.ones(K_use)
	sig_mask = np.zeros(K_use, dtype=bool)
	# Bonferroni correction
	#alpha_b = alpha / float(K_use)
	alpha_b = alpha

	for i in range(K_use):
		# two-proportion z-test comparing proportions, using pooled estimator
		pooled = (counts_gt[i] + counts_gen[i]) / float(n_gt + n_gen)
		se = np.sqrt(pooled * (1 - pooled) * (1.0 / n_gt + 1.0 / n_gen))
		if se == 0:
			pvals[i] = 1.0
			continue
		z = (p_gen[i] - p_gt[i]) / se
		pval = 2.0 * (1.0 - norm.cdf(abs(z)))
		pvals[i] = pval
		# 新增：打印每个聚类的关键值，看差异
		print(f"聚类{i}: p_gt={p_gt[i]:.4f}, p_gen={p_gen[i]:.4f}, pval={pval:.4f}")
		sig_mask[i] = pval < alpha_b

	ndb = int(sig_mask.sum())
	return {"ndb": ndb, "sig_mask": sig_mask, "pvals": pvals,
			"counts_gt": counts_gt, "counts_gen": counts_gen, "centers": centers}


if __name__ == "__main__":
	print("ndb module loaded")


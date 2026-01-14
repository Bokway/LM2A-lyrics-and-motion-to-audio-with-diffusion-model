import numpy as np
import librosa
from scipy import linalg


def _load_audio(path, sr=22050):
	y, sr = librosa.load(path, sr=sr, mono=True)
	return y, sr


def _embed_mfcc(path, sr=22050, n_mfcc=40):
	y, sr = _load_audio(path, sr=sr)
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
	return mfcc.mean(axis=1)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
	# from standard FID implementation, adapted for audio embeddings
	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)
	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	diff = mu1 - mu2
	# scipy.linalg.sqrtm sometimes returns just the matrix or (matrix, info)
	res = linalg.sqrtm(sigma1.dot(sigma2))
	if isinstance(res, tuple):
		covmean = res[0]
	else:
		covmean = res
	if not np.isfinite(covmean).all():
		offset = np.eye(sigma1.shape[0]) * eps
		res2 = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
		covmean = res2[0] if isinstance(res2, tuple) else res2

	# numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		covmean = covmean.real

	tr_covmean = np.trace(covmean)
	return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_fad(gt_files, gen_files, embed_fn=None, sr=22050):
	"""
	Compute an approximate Frechet Audio Distance between two lists of audio files.

	- gt_files, gen_files: lists of file paths (must be same-length lists for per-sample pairing,
	  but FAD here treats them as two sets, order is irrelevant).
	- embed_fn: optional callable(path)->1D embedding. Defaults to mean-pooled MFCCs.

	Returns: scalar FAD value and dict with statistics.
	"""
	if embed_fn is None:
		embed_fn = lambda p: _embed_mfcc(p, sr=sr)

	gt_embs = []
	for p in gt_files:
		gt_embs.append(np.asarray(embed_fn(p), dtype=np.float64))
	gen_embs = []
	for p in gen_files:
		gen_embs.append(np.asarray(embed_fn(p), dtype=np.float64))

	gt_embs = np.stack(gt_embs, axis=0)
	gen_embs = np.stack(gen_embs, axis=0)

	mu1 = gt_embs.mean(axis=0)
	mu2 = gen_embs.mean(axis=0)
	sigma1 = np.cov(gt_embs, rowvar=False)
	sigma2 = np.cov(gen_embs, rowvar=False)

	fad = frechet_distance(mu1, sigma1, mu2, sigma2)
	return fad, {"mu_gt": mu1, "mu_gen": mu2, "cov_gt": sigma1, "cov_gen": sigma2}


if __name__ == "__main__":
	print("fad module loaded")


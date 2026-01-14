import numpy as np
import librosa
from scipy.special import rel_entr


def _load_audio(path, sr=22050):
	y, sr = librosa.load(path, sr=sr, mono=True)
	return y, sr


def _embed_mfcc(path, sr=22050, n_mfcc=40):
	y, sr = _load_audio(path, sr=sr)
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
	return mfcc.mean(axis=1)


def _kl(p, q, eps=1e-12):
	p = np.asarray(p, dtype=np.float64) + eps
	q = np.asarray(q, dtype=np.float64) + eps
	return float(np.sum(rel_entr(p, q)))


def _js(p, q, eps=1e-12):
	p = np.asarray(p, dtype=np.float64) + eps
	q = np.asarray(q, dtype=np.float64) + eps
	m = 0.5 * (p + q)
	return 0.5 * (_kl(p, m) + _kl(q, m))


def compute_js_kl(gt_files, gen_files, embed_fn=None, bins=100, sr=22050):
	"""
	Compute JS divergence and KL divergence on histograms of embedding dimensions.

	Returns a dict with per-dimension JS/KL and their means.
	"""
	if embed_fn is None:
		embed_fn = lambda p: _embed_mfcc(p, sr=sr)

	gt_embs = [np.asarray(embed_fn(p), dtype=np.float64) for p in gt_files]
	gen_embs = [np.asarray(embed_fn(p), dtype=np.float64) for p in gen_files]

	gt_embs = np.stack(gt_embs, axis=0)
	gen_embs = np.stack(gen_embs, axis=0)

	dims = gt_embs.shape[1]
	js_per_dim = []
	kl_per_dim = []

	for d in range(dims):
		a = gt_embs[:, d]
		b = gen_embs[:, d]
		mn = min(a.min(), b.min())
		mx = max(a.max(), b.max())
		if mn == mx:
			js_per_dim.append(0.0)
			kl_per_dim.append(0.0)
			continue
		hist_a, _ = np.histogram(a, bins=bins, range=(mn, mx), density=True)
		hist_b, _ = np.histogram(b, bins=bins, range=(mn, mx), density=True)
		# normalize to probability mass
		hist_a = hist_a / (hist_a.sum() + 1e-12)
		hist_b = hist_b / (hist_b.sum() + 1e-12)
		kl = _kl(hist_a, hist_b)
		js = _js(hist_a, hist_b)
		kl_per_dim.append(kl)
		js_per_dim.append(js)

	return {"js_per_dim": np.array(js_per_dim), "kl_per_dim": np.array(kl_per_dim),
			"js_mean": float(np.mean(js_per_dim)), "kl_mean": float(np.mean(kl_per_dim))}


if __name__ == "__main__":
	print("js_kl module loaded")


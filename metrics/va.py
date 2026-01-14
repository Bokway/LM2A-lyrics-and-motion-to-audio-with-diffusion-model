import numpy as np
from scipy.spatial.distance import cosine


def compute_va_metrics(gt_va, gen_va):
    """
    Compute distances and cosine similarities in valence-arousal space.

    - gt_va, gen_va: arrays of shape (N,2) or lists of 2-tuples.

    Returns dict with per-sample euclidean distances and cosine similarities and their means.
    """
    gt = np.asarray(gt_va, dtype=np.float64)
    gen = np.asarray(gen_va, dtype=np.float64)
    if gt.shape != gen.shape:
        raise ValueError('gt_va and gen_va must have same shape')
    diffs = gt - gen
    dists = np.linalg.norm(diffs, axis=1)
    cosims = []
    for a, b in zip(gt, gen):
        # cosine similarity; handle zero vectors
        if np.allclose(a, 0) or np.allclose(b, 0):
            cosims.append(0.0)
        else:
            cos = 1.0 - cosine(a, b)
            cosims.append(cos)
    cosims = np.array(cosims, dtype=np.float64)
    return {"per_sample_dist": dists, "dist_mean": float(dists.mean()),
            "per_sample_cosine": cosims, "cosine_mean": float(cosims.mean())}


if __name__ == "__main__":
    print("va module loaded")

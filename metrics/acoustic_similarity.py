import numpy as np
import librosa
from scipy.spatial.distance import cosine


def _load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def _embed_mfcc(path, sr=22050, n_mfcc=40):
    y, sr = _load_audio(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def compute_pairwise_cosine(gt_files, gen_files, embed_fn=None, sr=22050):
    """
    Compute cosine similarity between matching pairs of gt_files and gen_files.

    If a CLAP model is available, user can pass an `embed_fn` that returns an embedding per file.
    Otherwise this falls back to MFCC mean embeddings.

    Returns: dict with per-sample cosine similarities and mean/std.
    """
    if embed_fn is None:
        embed_fn = lambda p: _embed_mfcc(p, sr=sr)

    sims = []
    for g, s in zip(gt_files, gen_files):
        a = np.asarray(embed_fn(g), dtype=np.float64)
        b = np.asarray(embed_fn(s), dtype=np.float64)
        # cosine distance -> similarity
        sim = 1.0 - cosine(a, b) if (a is not None and b is not None) else 0.0
        sims.append(sim)

    sims = np.array(sims, dtype=np.float64)
    return {"per_sample": sims, "mean": float(sims.mean()), "std": float(sims.std())}


if __name__ == "__main__":
    print("acoustic_similarity module loaded")

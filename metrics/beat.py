import numpy as np
import librosa


def _load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def _beat_times(path, sr=22050):
    y, sr = _load_audio(path, sr=sr)
    # use onset detection + beat tracker
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    return times


def _match_beats(ref_times, est_times, tol=0.07):
    """Match estimated beats to reference beats within tolerance (seconds).

    Returns: matches (list of tuples (ref_idx, est_idx, error)), unmatched counts
    """
    ref_times = np.asarray(ref_times)
    est_times = np.asarray(est_times)
    matched_ref = set()
    matched_est = set()
    matches = []
    for i, rt in enumerate(ref_times):
        if est_times.size == 0:
            continue
        # find closest est beat
        diffs = np.abs(est_times - rt)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol and j not in matched_est:
            matched_ref.add(i)
            matched_est.add(j)
            matches.append((i, j, float(est_times[j] - rt)))
    return matches, list(set(range(len(ref_times))) - matched_ref), list(set(range(len(est_times))) - matched_est)


def compute_beat_metrics(gt_files, gen_files, sr=22050, tol=0.07):
    """
    For each pair of GT and generated file, compute beat-hit rate, mean beat error for matched beats,
    and F1-like alignment score.

    Returns dict with per-sample metrics and aggregated statistics.
    """
    per_hits = []
    per_precision = []
    per_recall = []
    per_f1 = []
    per_err = []

    for g, s in zip(gt_files, gen_files):
        try:
            gt_bt = _beat_times(g, sr=sr)
        except Exception:
            gt_bt = np.array([])
        try:
            gen_bt = _beat_times(s, sr=sr)
        except Exception:
            gen_bt = np.array([])

        matches, unmatched_ref, unmatched_est = _match_beats(gt_bt, gen_bt, tol=tol)
        n_ref = len(gt_bt)
        n_est = len(gen_bt)
        n_matched = len(matches)
        precision = n_matched / n_est if n_est > 0 else 0.0
        recall = n_matched / n_ref if n_ref > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_err = np.mean([abs(e) for (_, _, e) in matches]) if matches else 0.0

        per_hits.append(n_matched)
        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(f1)
        per_err.append(mean_err)

    per_precision = np.array(per_precision)
    per_recall = np.array(per_recall)
    per_f1 = np.array(per_f1)
    per_err = np.array(per_err)

    return {
        "per_sample_hits": np.array(per_hits),
        "precision_mean": float(per_precision.mean()),
        "recall_mean": float(per_recall.mean()),
        "f1_mean": float(per_f1.mean()),
        "err_mean": float(per_err.mean()),
        "per_sample_precision": per_precision,
        "per_sample_recall": per_recall,
        "per_sample_f1": per_f1,
        "per_sample_err": per_err
    }


if __name__ == "__main__":
    print("beat module loaded")

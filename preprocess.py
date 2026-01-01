import os
import json
import numpy as np
import argparse
import torch
import librosa
from transformers import RobertaTokenizer, RobertaModel

try:
	from BigVGAN.meldataset import get_mel_spectrogram		# upload your BigVGAN folder to use its mel function
except Exception:
	get_mel_spectrogram = None

# load RoBERTa once
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.eval()


def load_wav(path, sr=None):
	y, s = librosa.load(path, sr=sr)
	return y, s


def default_bigvgan_hparams():
	# minimal set matching common configs
	class H:
		pass
	h = H()
	h.n_fft = 1024
	h.num_mels = 80
	h.sampling_rate = 22050
	h.hop_size = 256
	h.win_size = 1024
	h.fmin = 0
	h.fmax = None
	return h


def extract_mel_bigvgan(wav_path, start_seconds, sequence_seconds=6, sr=22050, hparams=None, device='cpu'):
	y, s = load_wav(wav_path, sr=sr)
	start_sample = int(start_seconds * s)
	end_sample = start_sample + int(sequence_seconds * s)
	y_seg = y[start_sample:end_sample]
	if y_seg.size == 0:
		return None, s, 256

	h = hparams if hparams is not None else default_bigvgan_hparams()
	# BigVGAN expects torch tensor in shape (B, T)
	wav_t = torch.from_numpy(y_seg).float().to(device)
	if wav_t.dim() == 1:
		wav_t = wav_t.unsqueeze(0)
			
	with torch.no_grad():
		mel = get_mel_spectrogram(wav_t, h)  # (1, 80, T)
			
	mel = mel.squeeze(0).cpu().numpy().astype(np.float32)  # (80, T)
	hop_length = int(h.hop_size)
	return mel, s, hop_length



def compute_lyrics_embedding(text, T, device='cpu'):
	toks = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
	with torch.no_grad():
		out = roberta(**{k: v.to(device) for k, v in toks.items()})
	# use mean pooling over last hidden state
	emb = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]
	emb_rep = np.tile(emb[None, :], (T, 1)).astype(np.float32)
	return emb_rep


def build_conditional_feats(smplfull, sliced, start_seconds, sequence_seconds=6, fps=30, vocab_size=512):
	"""Return per-frame motion (poses + Th + Rh) and lyrics embedding (T, emb_dim).

	This function only uses poses, Th, Rh per your request.
	"""
	start_frame = int(round(start_seconds * fps))
	total_frames = int(sequence_seconds * fps)
	frames = []
	keys = list(smplfull.keys())
	key_width = len(keys[0]) if keys else 6
	poses_len = None
	for i in range(total_frames):
		idx = start_frame + i
		k = str(idx).zfill(key_width)
		if k in smplfull:
			ann = smplfull[k]['annots'][0]
			poses = ann.get('poses', [[ ]])[0]
			Th = ann.get('Th', [[0,0,0]])[0]
			Rh = ann.get('Rh', [[0,0,0]])[0]
			poses = np.asarray(poses, dtype=np.float32)
			Th = np.asarray(Th, dtype=np.float32)
			Rh = np.asarray(Rh, dtype=np.float32)
			vec = np.concatenate([poses, Th, Rh]).astype(np.float32)
			frames.append(vec)
			if poses_len is None:
				poses_len = poses.shape[0]
		else:
			# fill with previous or zeros
			if len(frames) > 0:
				frames.append(frames[-1].copy())
			else:
				if poses_len is None:
					zero_poses = np.zeros((1,), dtype=np.float32)
				else:
					zero_poses = np.zeros((poses_len,), dtype=np.float32)
				zero_Th = np.zeros((3,), dtype=np.float32)
				zero_Rh = np.zeros((3,), dtype=np.float32)
				frames.append(np.concatenate([zero_poses, zero_Th, zero_Rh]).astype(np.float32))

	motion = np.stack(frames, axis=0).astype(np.float32)

	# --- 特征提取：平滑 + 中央差分 + 归一化准备 ---
	# 1) 平滑：使用高斯核在时间轴上对每个维度做轻度低通，抑制噪声放大
	def gaussian_smooth(x, sigma=1.0):
		# x: (T, D)
		if sigma <= 0:
			return x
		T = x.shape[0]
		# kernel size: cover +/- 3 sigma
		radius = max(1, int(3.0 * sigma))
		xs = np.arange(-radius, radius+1)
		kernel = np.exp(-0.5 * (xs / sigma) ** 2)
		kernel = kernel / kernel.sum()
		out = np.empty_like(x)
		for d in range(x.shape[1]):
			out[:, d] = np.convolve(x[:, d], kernel, mode='same')
		return out

	# apply smoothing (small sigma to keep temporal detail)
	motion_s = gaussian_smooth(motion, sigma=1.0)

	# 2) 中央差分计算速度与加速度（按 fps 标度，速度单位: pos/sec）
	dt = 1.0 / float(fps)
	T, D = motion_s.shape

	vel = np.zeros_like(motion_s, dtype=np.float32)
	acc = np.zeros_like(motion_s, dtype=np.float32)

	if T >= 2:
		# interior points: central diff
		for t in range(1, T-1):
			vel[t] = (motion_s[t+1] - motion_s[t-1]) / (2.0 * dt)
		# boundaries: forward/backward diff
		vel[0] = (motion_s[1] - motion_s[0]) / dt if T >= 2 else 0.0
		vel[T-1] = (motion_s[T-1] - motion_s[T-2]) / dt if T >= 2 else 0.0

	if T >= 3:
		for t in range(1, T-1):
			acc[t] = (motion_s[t+1] - 2.0 * motion_s[t] + motion_s[t-1]) / (dt * dt)
		# boundaries: use zero or one-sided second difference
		acc[0] = (motion_s[2] - 2.0 * motion_s[1] + motion_s[0]) / (dt * dt) if T >= 3 else 0.0
		acc[T-1] = (motion_s[T-1] - 2.0 * motion_s[T-2] + motion_s[T-3]) / (dt * dt) if T >= 3 else 0.0

	# concat: [pose, velocity, acceleration] -> final dim = D * 3
	motion = np.concatenate([motion_s, vel, acc], axis=1).astype(np.float32)

	# lyrics: find sliced key same as earlier logic
	keys = list(sliced.keys())
	found_key = None
	for k in keys:
		try:
			#from LM2ANew import preprocess as _p  # no-op to avoid lint
			# fallback: parse like before if dataset.parse_timestamp exists
			import re
			# use simple numeric parse here
			if ':' in k:
				val = float(k.split(':')[-1])
			else:
				val = float(k)
		except Exception:
			val = 0.0
		if abs(val - start_seconds) < 1e-3:
			found_key = k
			break
	if found_key is None and len(keys) > 0:
		dists = []
		for k in keys:
			try:
				if ':' in k:
					t = float(k.split(':')[-1])
				else:
					t = float(k)
			except Exception:
				t = 0.0
			dists.append((abs(t - start_seconds), k))
		found_key = sorted(dists, key=lambda x: x[0])[0][1]

	lyric_vec = None
	if found_key is not None:
		text = sliced.get(found_key, '')
		# compute roberta embedding and repeat to T
		lyric_vec = compute_lyrics_embedding(text, motion.shape[0])
	else:
		# fallback to zeros bag-of-words-like vector
		lyric_vec = np.zeros((motion.shape[0], 768), dtype=np.float32)

	return motion, lyric_vec


def make_dataset(root_in, out_dir, sequence_seconds=6, fps=30, sr=22050, use_bigvgan=True, hparams=None, device='cpu'):
	os.makedirs(out_dir, exist_ok=True)
	years = [os.path.join(root_in, d) for d in os.listdir(root_in) if os.path.isdir(os.path.join(root_in, d))]
	count = 0
	# We'll perform two passes: 1) write augmented npz (pose+vel+acc) and
	# accumulate running mean/var for motion features; 2) normalize saved npz
	# by the computed mean/std and write motion_stats.npz
	# Welford online algorithm for mean/variance
	motion_count = 0
	motion_mean = None
	motion_M2 = None
	for year in years:
		for song in os.listdir(year):
			song_path = os.path.join(year, song)
			if not os.path.isdir(song_path):
				continue
			sliced_path = os.path.join(song_path, 'sliced.json')
			smpl_path = os.path.join(song_path, 'smplfull.json')
			audio_path = os.path.join(song_path, 'audio.wav')

			if not (os.path.exists(sliced_path) and os.path.exists(smpl_path) and os.path.exists(audio_path)):
				print(f"skip {song_path}: missing files")
				continue

			try:
				sliced = json.load(open(sliced_path, 'r', encoding='utf-8'))
			except Exception:
				try:
					sliced = json.load(open(sliced_path, 'r'))
				except Exception as e2:
					print(f"skip {song_path} sliced read failed: {e2}")
					continue

			try:
				smplfull = json.load(open(smpl_path, 'r', encoding='utf-8'))
			except Exception:
				try:
					smplfull = json.load(open(smpl_path, 'r'))
				except Exception as e3:
					print(f"skip {song_path} smplfull read failed: {e3}")
					continue

			for k in list(sliced.keys())[:-1]:
				try:
					start_seconds = float(k.split(':')[-1]) if ':' in k else float(k)
				except Exception as e:
					print(f"skip slice {k} parse error: {e}")
					continue

				if use_bigvgan and get_mel_spectrogram is not None:
					mel, s, hop = extract_mel_bigvgan(audio_path, start_seconds, sequence_seconds=sequence_seconds, sr=sr, hparams=hparams, device=device)
				else:
					mel, s, hop = extract_mel_bigvgan(audio_path, start_seconds, sequence_seconds=sequence_seconds, sr=sr, hparams=None, device=device)

				if mel is None:
					print(f"skip slice {k} audio invalid")
					continue

				motion, lyrics_emb = build_conditional_feats(smplfull, sliced, start_seconds, sequence_seconds=sequence_seconds, fps=fps)
				out_name = f"sample_{count:08d}.npz"
				out_path = os.path.join(out_dir, out_name)
				# save augmented (not yet normalized)
				np.savez_compressed(out_path, mel=mel, motion=motion, lyrics=lyrics_emb, sr=s, hop_length=hop)

				# update running mean/var over motion frames
				# motion: (T, D)
				m = motion.reshape(-1, motion.shape[1])
				if motion_mean is None:
					motion_mean = np.zeros((m.shape[1],), dtype=np.float64)
					motion_M2 = np.zeros((m.shape[1],), dtype=np.float64)
					motion_count = 0
				# Welford update per-dimension
				for row in m:
					motion_count += 1
					delta = row - motion_mean
					motion_mean += delta / motion_count
					delta2 = row - motion_mean
					motion_M2 += delta * delta2

				count += 1

	# end for years

	# compute mean/std
	if motion_count > 1:
		motion_var = motion_M2 / (motion_count - 1)
		motion_std = np.sqrt(motion_var)
	else:
		motion_std = np.ones_like(motion_mean)

	# save stats
	stats_path = os.path.join(out_dir, 'motion_stats.npz')
	np.savez_compressed(stats_path, mean=motion_mean.astype(np.float32), std=motion_std.astype(np.float32))
	print('wrote motion stats to', stats_path)

	# second pass: normalize motion in all saved npz files (overwrite)
	files = [f for f in os.listdir(out_dir) if f.endswith('.npz') and f != 'motion_stats.npz']
	for f in files:
		p = os.path.join(out_dir, f)
		try:
			d = np.load(p, allow_pickle=True)
			motion = d['motion'].astype(np.float32)
			# normalize per-dim (avoid divide by zero)
			std_safe = np.where(motion_std == 0, 1.0, motion_std)
			motion_norm = (motion - motion_mean.astype(np.float32)) / std_safe.astype(np.float32)
			# rewrite npz keeping other fields
			mel = d['mel']
			lyrics = d['lyrics']
			sr = int(d.get('sr', 22050))
			hop = int(d.get('hop_length', 256))
			np.savez_compressed(p, mel=mel, motion=motion_norm, lyrics=lyrics, sr=sr, hop_length=hop)
		except Exception as e:
			print('warning normalizing', p, e)

	return count


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('--root', default=r'/mnt/mydev2/Bob/LM2ANew/dataset')  # your dataset path
	p.add_argument('--out', default=r'/mnt/mydev2/Bob/LM2ANew/npz')	  # your output path
	p.add_argument('--sequence_seconds', type=int, default=6)
	p.add_argument('--fps', type=int, default=30)
	p.add_argument('--sr', type=int, default=22050)
	p.add_argument('--no_bigvgan', action='store_true')
	args = p.parse_args()

	print(f"Preprocess: input={args.root}  -> output={args.out}")
	# Prefer using the BigVGAN model's own hparams when available so mel extraction
	# exactly matches what BigVGAN expects (avoids mismatch between mel conventions).
	h = default_bigvgan_hparams()
	if (not args.no_bigvgan) and (get_mel_spectrogram is not None):
		try:
			import bigvgan
			print('Loading BigVGAN model to obtain hparams (this may take a moment)...')
			model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_22khz_80band', use_cuda_kernel=False)
			h = model.h
			# keep model around only briefly; it's fine to let it be GC'd afterwards
			print('Using BigVGAN model.h for mel extraction')
		except Exception as e:
			print('Could not load BigVGAN model; falling back to default hparams:', e)

	n = make_dataset(args.root, args.out, sequence_seconds=args.sequence_seconds, fps=args.fps, sr=args.sr, use_bigvgan=not args.no_bigvgan, hparams=h, device='cpu')
	print('wrote samples:', n)


	"""
	before running this, 
	1.make sure you have uploaded BigVGAN folder to the same directory as this preprocess.py
	and have installed its dependencies 
	2.modify the root and out path to your dataset path and desired output path
	3.the dataset is that one with smplfull.json, sliced.json and audio.wav in each song folder extrated from JustDance the game
	

	"""

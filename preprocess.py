import os
import json
import numpy as np
import argparse
import torch
import librosa
from transformers import RobertaTokenizer, RobertaModel
import soundfile as sf

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


def get_max_valid_time(sliced, smplfull, sequence_seconds=6, fps=30):
    max_lyric_time = 0.0
    for k in sliced.keys():
        try:
            # 解析“时:分:秒”格式的时间字符串
            time_str = k.strip('"')  # 去掉可能的引号
            parts = time_str.split(':')
            # 处理不同格式（比如“00:26.050”是分:秒，“02:58.530”是分:秒，若有小时则是时:分:秒）
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                start_seconds = minutes * 60 + seconds
            elif len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                start_seconds = hours * 3600 + minutes * 60 + seconds
            else:
                start_seconds = float(time_str)  # 兜底：纯数字格式
            
            end_seconds = start_seconds + sequence_seconds
            if end_seconds > max_lyric_time:
                max_lyric_time = end_seconds
        except Exception as e:
            print(f"解析歌词时间{k}失败: {e}")
            continue

    # 动作时间计算
    max_motion_time = 0.0
    if smplfull:
        keys = list(smplfull.keys())
        if keys:
            last_frame_idx = int(max(keys, key=lambda x: int(x)))
            max_motion_time = last_frame_idx / fps  

    max_valid_time = min(max_lyric_time, max_motion_time)
    if max_valid_time < sequence_seconds:
        max_valid_time = sequence_seconds
    
    # 打印日志确认
    print(f"歌词最大有效时间：{max_lyric_time:.2f} 秒")
    print(f"动作最大有效时间：{max_motion_time:.2f} 秒")
    print(f"最终max_valid_time：{max_valid_time:.2f} 秒")
    return max_valid_time


def extract_mel_bigvgan(wav_path=None, y=None, s=None, start_seconds=0, sequence_seconds=6, sr=22050, hparams=None, device='cpu'):
    """
    提取Mel频谱，支持两种调用方式：
    1. 传统方式：传入wav_path，自动加载音频
    2. 优化方式：直接传入y（音频数据）和s（采样率），无需读取文件（无临时文件）
    """
    # 优先使用传入的y和s，否则从wav_path加载音频
    if y is None or s is None:
        if wav_path is None:
            raise ValueError("必须传入wav_path，或同时传入y和s")
        y, s = load_wav(wav_path, sr=sr)
    
    # 计算音频切片的起止样本索引
    start_sample = int(start_seconds * s)
    end_sample = start_sample + int(sequence_seconds * s)
    
    # 校验：切片结束索引超出音频长度则返回None
    if end_sample > len(y):
        print(f"音频切片长度不足：请求{sequence_seconds}s，实际剩余{ (len(y)-start_sample)/s :.2f}s")
        return None, s, 256
    
    y_seg = y[start_sample:end_sample]
    # 校验：空音频片段直接返回None
    if y_seg.size == 0:
        print("音频切片为空，跳过")
        return None, s, 256

    # 获取BigVGAN参数
    h = hparams if hparams is not None else default_bigvgan_hparams()
    # 转换为torch张量（BigVGAN要求形状为(B, T)）
    wav_t = torch.from_numpy(y_seg).float().to(device)
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    
    # 无梯度提取Mel频谱
    with torch.no_grad():
        if get_mel_spectrogram is None:
            print("警告：get_mel_spectrogram未加载，无法提取Mel频谱")
            return None, s, 256
        mel = get_mel_spectrogram(wav_t, h)  # (1, 80, T)
    
    # 调整形状并转换为numpy数组
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
	"""Return per-frame motion (poses + Th + Rh) and lyrics embedding (T, emb_dim), 额外返回歌词文本和动作帧数区间.

	This function only uses poses, Th, Rh per your request.
	"""
	# 1. 计算动作帧数区间（记录起始和结束帧数）
	frame_start = int(round(start_seconds * fps))
	frame_end = frame_start + int(sequence_seconds * fps)  # 结束帧数（不含）
	total_frames = int(sequence_seconds * fps)
	frames = []
	keys = list(smplfull.keys())
	key_width = len(keys[0]) if keys else 6
	poses_len = None
	for i in range(total_frames):
		idx = frame_start + i
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

	# --- 使用正确的时间解析逻辑匹配歌词切片 ---
	keys = list(sliced.keys())
	found_key = None
	# 先尝试精确匹配
	for k in keys:
		try:
			# 正确解析当前歌词切片k的时间
			time_str_k = k.strip('"')
			parts_k = time_str_k.split(':')
			if len(parts_k) == 2:
				minutes_k = float(parts_k[0])
				seconds_k = float(parts_k[1])
				start_sec_k = minutes_k * 60 + seconds_k
			elif len(parts_k) == 3:
				hours_k = float(parts_k[0])
				minutes_k = float(parts_k[1])
				seconds_k = float(parts_k[2])
				start_sec_k = hours_k * 3600 + minutes_k * 60 + seconds_k
			else:
				start_sec_k = float(time_str_k)
			# 与真实start_seconds比较（放宽阈值，避免浮点误差）
			if abs(start_sec_k - start_seconds) < 1e-3:
				found_key = k
				break
		except Exception as e:
			continue
	# 精确匹配失败时，找时间最近的切片
	if found_key is None and len(keys) > 0:
		dists = []
		for k in keys:
			try:
				time_str_k = k.strip('"')
				parts_k = time_str_k.split(':')
				if len(parts_k) == 2:
					minutes_k = float(parts_k[0])
					seconds_k = float(parts_k[1])
					start_sec_k = minutes_k * 60 + seconds_k
				elif len(parts_k) == 3:
					hours_k = float(parts_k[0])
					minutes_k = float(parts_k[1])
					seconds_k = float(parts_k[2])
					start_sec_k = hours_k * 3600 + minutes_k * 60 + seconds_k
				else:
					start_sec_k = float(time_str_k)
			except Exception:
				start_sec_k = 0.0
			# 计算真实时间的距离
			dists.append((abs(start_sec_k - start_seconds), k))
		# 按距离排序，取最近的
		dists.sort(key=lambda x: x[0])
		found_key = dists[0][1]

	# 获取歌词原始文本
	lyric_text = ""
	lyric_vec = None
	if found_key is not None:
		lyric_text = sliced.get(found_key, '')  # 保存原始歌词文本
		# compute roberta embedding and repeat to T
		lyric_vec = compute_lyrics_embedding(lyric_text, motion.shape[0])
	else:
		# fallback to zeros bag-of-words-like vector
		lyric_vec = np.zeros((motion.shape[0], 768), dtype=np.float32)

	# 返回歌词文本、动作起始/结束帧数（原来只返回motion和lyric_vec）
	return motion, lyric_vec, lyric_text, frame_start, frame_end


def make_dataset(root_in, out_dir, sequence_seconds=6, fps=30, sr=22050, use_bigvgan=True, hparams=None, device='cpu'):
    sample_list = []
    os.makedirs(out_dir, exist_ok=True)
    years = [os.path.join(root_in, d) for d in os.listdir(root_in) if os.path.isdir(os.path.join(root_in, d))]
    count = 0
    # Welford online algorithm for mean/variance（运动特征统计）
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

            # 校验必要文件是否存在
            if not (os.path.exists(sliced_path) and os.path.exists(smpl_path) and os.path.exists(audio_path)):
                print(f"skip {song_path}: 缺失必要文件（sliced/smplfull/audio）")
                continue

            # 加载歌词文件
            try:
                sliced = json.load(open(sliced_path, 'r', encoding='utf-8'))
            except Exception:
                try:
                    sliced = json.load(open(sliced_path, 'r'))
                except Exception as e2:
                    print(f"skip {song_path} sliced.json读取失败: {e2}")
                    continue

            # 加载动作文件
            try:
                smplfull = json.load(open(smpl_path, 'r', encoding='utf-8'))
            except Exception:
                try:
                    smplfull = json.load(open(smpl_path, 'r'))
                except Exception as e3:
                    print(f"skip {song_path} smplfull.json读取失败: {e3}")
                    continue
            
            # 1. 计算该歌曲的最大有效时间（歌词和动作的最小有效结束时间）
            max_valid_time = get_max_valid_time(sliced, smplfull, sequence_seconds, fps)
            
            # 2. 加载完整音频并裁切为有效部分（无临时文件保存）
            y, s = load_wav(audio_path, sr=sr)
            max_valid_sample = int(max_valid_time * s)
            y_valid = y[:max_valid_sample]  # 裁切：仅保留有效音频部分
            
            """
			# 保存裁切后的音频以供检查（可选）
            valid_audio_path = os.path.join(song_path, "audio_valid.wav")
            try:
                sf.write(valid_audio_path, y_valid, s)
                print(f"成功保存裁切音频：{valid_audio_path}")
            except Exception as e:
                print(f"保存裁切音频失败：{valid_audio_path}，错误：{e}")
			"""

            # 提取歌曲名称
            song_name = os.path.basename(song_path)

            # 遍历歌词切片，仅处理有效时间范围内的切片
            for k in list(sliced.keys())[:-1]:
                try:
                    # ========== 替换为正确的时间解析逻辑（和get_max_valid_time一致） ==========
                    time_str = k.strip('"')
                    parts = time_str.split(':')
                    if len(parts) == 2:
                        minutes = float(parts[0])
                        seconds = float(parts[1])
                        start_seconds = minutes * 60 + seconds
                    elif len(parts) == 3:
                        hours = float(parts[0])
                        minutes = float(parts[1])
                        seconds = float(parts[2])
                        start_seconds = hours * 3600 + minutes * 60 + seconds
                    else:
                        start_seconds = float(time_str)
                except Exception as e:
                    print(f"skip slice {k}: 起始时间解析失败: {e}")
                    continue

                # 此时过滤条件才会真正生效
                slice_end_seconds = start_seconds + sequence_seconds
                if slice_end_seconds > max_valid_time:
                    print(f"skip slice {k}: 超出有效时间范围（最大有效时间：{max_valid_time:.2f}s）")
                    continue

                # 3. 优化：直接传入裁切后的音频数据y_valid和采样率s，无需读取临时音频文件
                if use_bigvgan and get_mel_spectrogram is not None:
                    mel, s, hop = extract_mel_bigvgan(
                        y=y_valid,  # 直接传入有效音频数据
                        s=s,        # 直接传入采样率
                        start_seconds=start_seconds,
                        sequence_seconds=sequence_seconds,
                        sr=sr,
                        hparams=hparams,
                        device=device
                    )
                else:
                    mel, s, hop = extract_mel_bigvgan(
                        y=y_valid,
                        s=s,
                        start_seconds=start_seconds,
                        sequence_seconds=sequence_seconds,
                        sr=sr,
                        hparams=None,
                        device=device
                    )

                # 校验Mel频谱是否有效
                if mel is None:
                    print(f"skip slice {k}: Mel频谱提取失败")
                    continue

                # 修改：接收额外返回的歌词文本和动作帧数区间
                motion, lyrics_emb, lyric_text, frame_start, frame_end = build_conditional_feats(
                    smplfull, sliced, start_seconds,
                    sequence_seconds=sequence_seconds,
                    fps=fps
                )

                # 计算时间结束秒数
                time_end = start_seconds + sequence_seconds
                
                # 保存样本（新增所有元信息，注意字符串转为np.bytes_类型）
                out_name = f"sample_{count:08d}.npz"
                out_path = os.path.join(out_dir, out_name)
                np.savez_compressed(
                    out_path,
                    # 原有特征
                    mel=mel,
                    motion=motion,
                    lyrics=lyrics_emb,
                    sr=s,
                    hop_length=hop,
                    # 新增元信息（关键：字符串用np.bytes_包装，避免读取时编码异常）
                    song_name=np.bytes_(song_name.encode('utf-8')),
                    time_start=np.float32(start_seconds),
                    time_end=np.float32(time_end),
                    lyric_text=np.bytes_(lyric_text.encode('utf-8')),
                    frame_start=np.int32(frame_start),
                    frame_end=np.int32(frame_end)
                )
                
                # 保存样本后，添加样本信息到清单（在count += 1之前）
                sample_info = {
                    "npz_name": out_name,
                    "song_name": song_name,
                    "time_start": round(float(start_seconds), 2),
                    "time_end": round(float(time_end), 2),
                    "lyric_text": lyric_text,
                    "frame_start": int(frame_start),
                    "frame_end": int(frame_end)
                }
                sample_list.append(sample_info)

                # 更新运动特征的均值和方差（Welford算法）
                m = motion.reshape(-1, motion.shape[1])
                if motion_mean is None:
                    motion_mean = np.zeros((m.shape[1],), dtype=np.float64)
                    motion_M2 = np.zeros((m.shape[1],), dtype=np.float64)
                    motion_count = 0
                # 逐行更新统计量
                for row in m:
                    motion_count += 1
                    delta = row - motion_mean
                    motion_mean += delta / motion_count
                    delta2 = row - motion_mean
                    motion_M2 += delta * delta2

                count += 1

    # 计算运动特征的均值和标准差
    if motion_count > 1:
        motion_var = motion_M2 / (motion_count - 1)
        motion_std = np.sqrt(motion_var)
    else:
        motion_std = np.ones_like(motion_mean) if motion_mean is not None else np.array([])

    # 保存运动特征统计信息
    if motion_mean is not None:
        stats_path = os.path.join(out_dir, 'motion_stats.npz')
        np.savez_compressed(stats_path, mean=motion_mean.astype(np.float32), std=motion_std.astype(np.float32))
        print(f'已保存运动特征统计信息至: {stats_path}')
    else:
        print("警告：无有效运动样本，未生成motion_stats.npz")

    # 第二遍遍历：归一化所有样本的运动特征（覆盖原文件，注意保留新增的元信息！）
    files = [f for f in os.listdir(out_dir) if f.endswith('.npz') and f != 'motion_stats.npz']
    for f in files:
        p = os.path.join(out_dir, f)
        try:
            d = np.load(p, allow_pickle=True)
            motion = d['motion'].astype(np.float32)
            # 避免除零错误
            if motion_mean is None or motion_std is None:
                continue
            std_safe = np.where(motion_std == 0, 1.0, motion_std)
            # 归一化运动特征
            motion_norm = (motion - motion_mean.astype(np.float32)) / std_safe.astype(np.float32)

            # 关键：读取所有原有特征和元信息，避免覆盖丢失
            mel = d['mel']
            lyrics = d['lyrics']
            sr = int(d.get('sr', 22050))
            hop = int(d.get('hop_length', 256))
            song_name = d.get('song_name', np.bytes_("")).decode('utf-8') 
            time_start = d.get('time_start', np.float32(0.0))
            time_end = d.get('time_end', np.float32(0.0))
            lyric_text = d.get('lyric_text', np.bytes_("")).decode('utf-8')	
            frame_start = d.get('frame_start', np.int32(0))
            frame_end = d.get('frame_end', np.int32(0))

            # 覆盖保存：保留所有信息，仅更新motion为归一化后的版本
            np.savez_compressed(
                p,
                mel=mel,
                motion=motion_norm,  # 仅更新此处
                lyrics=lyrics,
                sr=sr,
                hop_length=hop,
                # 保留所有新增元信息
                song_name=song_name,
                time_start=time_start,
                time_end=time_end,
                lyric_text=lyric_text,
                frame_start=frame_start,
                frame_end=frame_end
            )
        except Exception as e:
            print(f"警告：归一化样本 {p} 失败: {e}")

    # 函数末尾保存清单（在打印预处理完成之前）
    list_path = os.path.join(out_dir, "sample_info_list.json")
    with open(list_path, 'w', encoding='utf-8') as f:
        json.dump(sample_list, f, ensure_ascii=False, indent=2)
    print(f"已保存样本清单至：{list_path}")
    
    print(f"预处理完成，共生成 {count} 个样本")
    return count


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('--root', default=r'D:\lm2d\dataset')  # your dataset path
	p.add_argument('--out', default=r'D:\lm2d\npz')	  # your output path
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

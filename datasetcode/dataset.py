import os
import numpy as np
import torch
from torch.utils.data import Dataset

"""
作用：

读取给定目录下的所有 `.npz` 文件，并对齐时间维度：

- mel: 原始为 (n_mels, T_mel) -> 重采样并返回 (T, n_mels)
- motion: (T_motion, D_motion) -> 返回 (T, D_motion)
- lyrics: (T_lyrics, D_lyrics) -> 返回 (T, D_lyrics)

对齐策略：
- 以 `motion`或者 `mel` 的帧数作为目标长度 T（在你的 preprocess 中 motion 长度 = sequence_seconds * fps，例如 6*30=180）
- 用线性插值将 mel 的时间轴从 T_mel 重采样到 T
- 若 motion/lyrics 长度与目标不一致：截断或重复最后一帧进行填充

返回：字典，包含 `mel`, `motion`, `lyrics` 三个 torch.FloatTensor，以及 `sr`、`hop_length`、`path`

示例：
    ds = MelDataset(r"C:\...\npz")
    sample = ds[0]
    sample['mel'].shape    # (T, n_mels)
    sample['motion'].shape # (T, D_motion)
"""


def resample_mel_linear(mel, target_len):
    """线性插值重采样 mel 的时间轴。

    mel: np.ndarray, shape (n_mels, T_mel)
    target_len: int, 目标时间步数
    返回: np.ndarray, shape (n_mels, target_len)
    """
    n_mels, T_mel = mel.shape
    if T_mel == target_len:
        return mel.astype(np.float32)

    x_old = np.arange(T_mel)
    x_new = np.linspace(0, T_mel - 1, num=target_len)
    mel_rs = np.empty((n_mels, target_len), dtype=np.float32)
    for i in range(n_mels):
        mel_rs[i] = np.interp(x_new, x_old, mel[i])
    return mel_rs


def interpolate_seq(arr, target_len):
    """线性插值上/下采样序列到目标长度。

    arr: np.ndarray, shape (T, D) 或 (T,) 返回 (target_len, D) 或 (target_len,)
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        T = arr.shape[0]
        if T == target_len:
            return arr.astype(np.float32)
        x_old = np.arange(T)
        x_new = np.linspace(0, T - 1, num=target_len)
        return np.interp(x_new, x_old, arr).astype(np.float32)

    # arr ndim >=2 : treat axis 0 as time
    T, D = arr.shape
    if T == target_len:
        return arr.astype(np.float32)
    x_old = np.arange(T)
    x_new = np.linspace(0, T - 1, num=target_len)
    out = np.empty((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, arr[:, d])
    return out


def match_len(arr, target_len, mode='repeat'):
    """确保 arr 在第0维长度为 target_len。
    mode:
      - 'repeat' : 重复最后一帧填充（原实现）
      - 'interp' : 用线性插值重采样到 target_len
    返回 float32 的 np.ndarray
    """
    if arr is None:
        return None
    if mode == 'interp':
        return interpolate_seq(arr, target_len)

    # fallback: repeat mode
    cur = arr
    cur_len = cur.shape[0]
    if cur_len == target_len:
        return cur.astype(np.float32)
    if cur_len > target_len:
        return cur[:target_len].astype(np.float32)
    # cur_len < target_len -> pad by repeating last frame
    if cur_len == 0:
        pad_shape = list(cur.shape)
        pad_shape[0] = target_len - cur_len
        pad = np.zeros(tuple(pad_shape), dtype=np.float32)
        return np.concatenate([cur.astype(np.float32), pad], axis=0)
    last = cur[-1][None, :].astype(np.float32)
    reps = target_len - cur_len
    pad = np.repeat(last, reps, axis=0)
    return np.concatenate([cur.astype(np.float32), pad], axis=0)


class MelDataset(Dataset):
    def __init__(self, npz_dir, align_mode='interp'):
        """align_mode: 'repeat' (默认旧行为) or 'interp' (线性插值到 mel 长度)
        """
        self.npz_dir = npz_dir
        files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
        files = sorted(files)
        self.files = [os.path.join(npz_dir, f) for f in files]
        self.align_mode = align_mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)

        # load with fallbacks
        mel = data['mel']  # expect (n_mels, T_mel)
        motion = data['motion']
        lyrics = data['lyrics']
        sr = int(data.get('sr', 22050))
        hop = int(data.get('hop_length', 256))

        # target length: use motion's time steps if available, else use lyrics, else use mel
        if motion is not None and getattr(motion, 'shape', None) is not None:
            T = motion.shape[0]
        elif lyrics is not None and getattr(lyrics, 'shape', None) is not None:
            T = lyrics.shape[0]
        else:
            T = mel.shape[1]
        
        # Always align to mel's time dimension
        T = mel.shape[1]   # 516

        # resample mel to T (mel is (n_mels, T_mel)) -> (n_mels, T)
        if mel.ndim == 3:
            # unexpected batch dim, try to squeeze first dim
            mel = np.squeeze(mel, axis=0)
        mel_rs = resample_mel_linear(mel, T)
        # transpose to (T, n_mels) which is often more convenient for models
        mel_out = mel_rs.T

        # ensure motion and lyrics have length T
        motion_out = match_len(motion, T, mode=self.align_mode)
        # 新行为：假定 preprocessed `.npz` 已包含 pose + velocity + acceleration
        #（即维度为 78*3 = 234）。这里不再为旧格式自动补齐。
        if motion_out is not None:
            motion_out = motion_out.astype(np.float32)
        lyrics_out = match_len(lyrics, T, mode=self.align_mode)

        # to torch tensors
        mel_t = torch.from_numpy(mel_out).float()
        motion_t = torch.from_numpy(motion_out).float()
        lyrics_t = torch.from_numpy(lyrics_out).float()

        return {
            'mel': mel_t,
            'motion': motion_t,
            'lyrics': lyrics_t,
            'sr': sr,
            'hop_length': hop,
            'path': path,
        }

    @staticmethod
    def collate_fn(batch):
        """简单 collate：把键堆叠（要求所有样本已经对齐为相同 T）。
        如果你的数据在 batch 维度上长度不一，请在这里实现更复杂的 padding/裁剪。
        """
        mel = torch.stack([b['mel'] for b in batch], dim=0)
        motion = torch.stack([b['motion'] for b in batch], dim=0)
        lyrics = torch.stack([b['lyrics'] for b in batch], dim=0)
        sr = batch[0]['sr']
        hop = batch[0]['hop_length']
        paths = [b['path'] for b in batch]
        return {'mel': mel, 'motion': motion, 'lyrics': lyrics, 'sr': sr, 'hop_length': hop, 'paths': paths}


if __name__ == '__main__':
    # 快速测试用法
    ds = MelDataset("/mnt/mydev2/Bob/LM2ANew/npz")
    print('num samples:', len(ds))
    s = ds[0]
    print('mel', s['mel'].shape, 'motion', s['motion'].shape, 'lyrics', s['lyrics'].shape)
    print('sr', s['sr'], 'hop_length', s['hop_length'], 'path', s['path'])
    print('mel dtype', s['mel'].dtype, 'motion dtype', s['motion'].dtype, 'lyrics dtype', s['lyrics'].dtype)
    print("Real mel min/max:", np.min(s['mel'].numpy()), np.max(s['mel'].numpy()))

    print('mel first row (前10个值):', s['mel'][0][:10])
    print('motion first row (前10个值):', s['motion'][0][:10])
    print('lyrics first row (前10个值):', s['lyrics'][0][:10])

    

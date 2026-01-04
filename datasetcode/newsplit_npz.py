#!/usr/bin/env python3
"""
Split .npz dataset into train/val/test folders.

Usage examples:
  python3 datasetcode/split_npz.py --npz_dir /path/to/npz --out_dir /path/to/out 
  python3 datasetcode/split_npz.py --npz_dir npz --out_dir npz_split --move


a little bug: 
Remember to move the sample_info_list.json file and motion_stats.npz file out of the npz directory before running.

一个小bug：
在运行之前记得把sample_info_list.json文件和motion_stats.npz文件从npz目录下移出来

"""
import argparse
import os
import glob
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split .npz files into train/val/test sets")
    p.add_argument("--npz_dir", default="D:\\lm2d\\npz", help="Directory containing .npz files")
    p.add_argument("--out_dir", default="D:\\lm2d\\npz_split", help="Output base directory")
    # 新增测试集比例参数（默认7:2:1）
    p.add_argument("--train_ratio", type=float, default=0.7, help="Fraction for training set")
    p.add_argument("--val_ratio", type=float, default=0.2, help="Fraction for validation set")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Fraction for test set")
    # 保留固定数量参数（可选，优先级高于比例）
    p.add_argument("--train_count", type=int, default=None, help="Exact number of training examples")
    p.add_argument("--val_count", type=int, default=None, help="Exact number of validation examples")
    p.add_argument("--test_count", type=int, default=None, help="Exact number of test examples")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument("--move", action="store_true", help="Move files instead of copying")
    p.add_argument("--pattern", default="*.npz", help="Glob pattern to match npz files")
    return p.parse_args()


def main():
    args = parse_args()
    npz_dir = Path(args.npz_dir)
    if not npz_dir.exists():
        raise SystemExit(f"npz_dir does not exist: {npz_dir}")

    # 输出目录：新增test_dir
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"
    # 递归创建目录（包括父目录）
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 获取所有npz文件并打乱
    files = sorted(glob.glob(str(npz_dir / args.pattern)))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {npz_dir}")
    rnd = random.Random(args.seed)
    rnd.shuffle(files)

    # 计算各集合数量（优先固定数量，其次比例）
    n_total = len(files)
    if all([args.train_count, args.val_count, args.test_count]):
        # 手动指定数量
        n_train = args.train_count
        n_val = args.val_count
        n_test = args.test_count
    else:
        # 按比例拆分（适配1868个样本）
        ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise SystemExit("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        n_train = int(round(n_total * args.train_ratio))
        n_val = int(round(n_total * args.val_ratio))
        # 补全剩余样本到测试集（避免总数不一致）
        n_test = n_total - n_train - n_val

    # 校验数量合理性
    if n_train + n_val + n_test != n_total:
        # 微调：优先保证训练集，最后1个样本归测试集
        diff = n_total - (n_train + n_val + n_test)
        n_test += diff
    if n_train < 0 or n_val < 0 or n_test < 0:
        raise SystemExit("Invalid split counts (negative numbers)")

    # 拆分文件列表
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    # 选择复制/移动操作
    op = shutil.copy2 if not args.move else shutil.move

    # 复制/移动文件到对应目录
    print(f"Copying/moving {len(train_files)} train files...")
    for f in train_files:
        dest = train_dir / Path(f).name
        op(f, str(dest))
    
    print(f"Copying/moving {len(val_files)} val files...")
    for f in val_files:
        dest = val_dir / Path(f).name
        op(f, str(dest))
    
    print(f"Copying/moving {len(test_files)} test files...")
    for f in test_files:
        dest = test_dir / Path(f).name
        op(f, str(dest))

    # 写入清单文件（方便后续加载）
    (out_dir / "train.txt").write_text("\n".join([Path(f).name for f in train_files]))
    (out_dir / "val.txt").write_text("\n".join([Path(f).name for f in val_files]))
    (out_dir / "test.txt").write_text("\n".join([Path(f).name for f in test_files]))

    # 输出统计信息
    print("="*50)
    print(f"Total files: {n_total}")
    print(f"Train set: {len(train_files)} ({len(train_files)/n_total:.1%})")
    print(f"Val set: {len(val_files)} ({len(val_files)/n_total:.1%})")
    print(f"Test set: {len(test_files)} ({len(test_files)/n_total:.1%})")
    print(f"All files saved to: {out_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
    
    

"""
Total files: 1780
Train set: 1246 (70.0%)
Val set: 356 (20.0%)
Test set: 178 (10.0%)
All files saved to: D:\lm2d\npz_split

before you run this script, make sure you have the correct directory structure and that 
the npz files are all in the specified npz_dir.

a little bug: 
Remember to move the sample_info_list.json file and motion_stats.npz file out of the npz directory before running.

一个小bug：
在运行之前记得把sample_info_list.json文件和motion_stats.npz文件从npz目录下移出来


"""

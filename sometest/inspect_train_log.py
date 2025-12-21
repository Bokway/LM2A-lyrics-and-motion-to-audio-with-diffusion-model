#!/usr/bin/env python3
"""
Inspect and plot `train_log.csv` created by `traingpu.py`.

Usage examples:
  python3 sometest/inspect_train_log.py --csv /path/to/train_log.csv --head 10 --plot --out_dir /path/to/out

Outputs:
  - prints head/tail and basic stats
  - writes `train_val_loss.png` into `--out_dir` (or CSV dir)
  
用于将训练日志可视化和检查的脚本。
"""
import argparse
import csv
import os
import math
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description='Inspect train_log.csv and plot losses')
    p.add_argument('--csv', default='/mnt/mydev2/Bob/LM2ANew/checkpoints_adan/train_log.csv', help='Path to train_log.csv')
    p.add_argument('--head', type=int, default=10, help='Number of head rows to print')
    p.add_argument('--tail', type=int, default=5, help='Number of tail rows to print')
    p.add_argument('--plot', default=True, action='store_true', help='Generate train/val loss plot')
    p.add_argument('--out_dir', default='/mnt/mydev2/Bob/LM2ANew/checkpoints_adan', help='Directory to write plot (default: csv file dir)')
    return p.parse_args()


def read_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            # normalize row length
            while len(r) < len(header):
                r.append('')
            rows.append(dict(zip(header, r)))
    return header, rows


def to_float(x):
    try:
        if x is None or x == '':
            return None
        return float(x)
    except Exception:
        return None


def summarize(rows):
    stats = defaultdict(list)
    for r in rows:
        train = to_float(r.get('train_loss'))
        val = to_float(r.get('val_loss'))
        step = to_float(r.get('step'))
        if train is not None:
            stats['train_steps'].append(step)
            stats['train_loss'].append(train)
        if val is not None:
            stats['val_steps'].append(step)
            stats['val_loss'].append(val)
    return stats


def print_head_tail(header, rows, head=10, tail=5):
    print('\nHeader:', header)
    n = len(rows)
    print(f'Rows: {n}')
    print('\n--- Head ---')
    for r in rows[:head]:
        print(r)
    if n > head + tail:
        print('\n...')
    print('\n--- Tail ---')
    for r in rows[-tail:]:
        print(r)


def plot_losses(stats, outpath):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError('matplotlib required to plot losses: ' + str(e))

    plt.figure(figsize=(9, 5))
    if stats.get('train_steps'):
        plt.plot(stats['train_steps'], stats['train_loss'], label='train_loss', alpha=0.8)
    if stats.get('val_steps'):
        plt.plot(stats['val_steps'], stats['val_loss'], label='val_loss', marker='o')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Train / Val Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    print('Wrote plot to', outpath)


def main():
    args = parse_args()
    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise SystemExit('csv not found: ' + csv_path)
    out_dir = args.out_dir or os.path.dirname(csv_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    header, rows = read_csv(csv_path)
    print_head_tail(header, rows, head=args.head, tail=args.tail)

    stats = summarize(rows)
    def mean(x):
        return sum(x) / len(x) if x else None

    print('\nSummary:')
    print('  train points:', len(stats.get('train_loss', [])), 'mean train loss:', mean(stats.get('train_loss', [])))
    print('  val points:', len(stats.get('val_loss', [])), 'mean val loss:', mean(stats.get('val_loss', [])))

    if args.plot:
        outpath = os.path.join(out_dir, 'train_val_loss.png')
        plot_losses(stats, outpath)


if __name__ == '__main__':
    main()


"""
Before you run this script,
1. make sure you have installed the required packages
2. modify the default paths in parse_args() or provide them via command line arguments
"""
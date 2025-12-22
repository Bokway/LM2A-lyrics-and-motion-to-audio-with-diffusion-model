# LM2A:lyrics and motion to audio with diffusion model


Pipline:
1. 下载BigVGAN和dataset
2. 运行preprocess获得1868个npz
3. 运行newsplit_npz,得到train/val/test set。(注意移出motion_stats.npz)
4. 调整参数运行train训练得到ckpt
6. 调整参数运行val获得评估结果
7. sometest里面的工具可以生成loss图以及生成音频audio

Pipline too:
1. Download BigVGAN and the dataset.
2. Run preprocess to obtain 1868 npz files.
3. Run newsplit_npz to get the train/val/test sets. (Note to remove motion_stats.npz).
4. Adjust the parameters and run train to obtain the ckpt.
6. Adjust the parameters and run val to obtain the evaluation results.
7. The tools in sometest can generate loss graphs and generate audio.


My env:
  Python 3.11.7 + CUDA 12.4 + PyTorch 2.5.1+cu121

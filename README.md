# LM2A:lyrics and motion to audio with diffusion model

**My env:**
  Python 3.11.7 + CUDA 12.4 + PyTorch 2.5.1+cu121


**Pipline:**
1. Download BigVGAN and the dataset.
2. Run preprocess to obtain 1868 npz files.
3. Run newsplit_npz to get the train/val/test sets. (Note to remove motion_stats.npz).
4. Adjust the parameters and run train to obtain the ckpt.
6. Adjust the parameters and run val to obtain the evaluation results.
7. The tools in sometest can generate loss graphs and generate audio.


**Existing items:**

Data sets: Songs 2020, 2021, 2022: Each data set contains all lyrics, segmented lyrics (sliced), motion parameters' JSON files, and audio files.
Downloaded BigVGAN 

**Process:**

Preprocess the code to obtain npz -> Divide the obtained npz into train/val/test -> Train to obtain ckpt -> Assess evaluation -> Convert mel to audio



**All the used codes:**

**1.preprocess:**

This code is used to preprocess a dataset containing audio and motion data to generate feature files suitable for training machine learning models.
It extracts mel spectrograms from audio, computes velocity and acceleration from motion data, and generates text embeddings for lyrics.
The final output is compressed NumPy files (.npz) containing these features, along with computed statistics of motion features for normalization purposes.

**2.datasetcode:**

dataset.py:

this code defines a PyTorch dataset class `MelDataset` that loads and aligns audio features (mel spectrograms), motion data, and lyrics data from a specified directory, facilitating subsequent model training and evaluation.

newsplit_npz.py:

Split .npz dataset into train/val/test folders, 7:2:1.

**3.models:**

adan.py:

This code implements the Adan optimizer, an optimization algorithm that combines adaptive moment estimation with Nesterov momentum and incorporates gradient differences for improved convergence.
The Adan optimizer is suitable for training deep learning models and effectively adjusts the learning rate to accelerate the convergence process.

cross_attention.py:

this code defines a cross-attention module for fusing mel-spectrogram hidden states with motion and lyrics features.
The module uses multi-head attention mechanisms to attend to motion and lyrics features separately,
then concatenates the two attention outputs and fuses them through a linear layer to produce the final fused features.

diffusion.py:

This code implements a Gaussian Diffusion Model based on PyTorch. The model includes the forward noising process, 
loss computation, and reverse sampling process,
used to generate mel-spectrogram features of audio. 
The model progressively adds noise to real data and trains a neural network to predict the noise, enabling the ability to generate data from noise. 

embedding.py:

This code defines a timestep embedding and conditional feature projection module, 
suitable for neural network models that process time series data (motion sequences) and text data (lyrics).   

unet1d_ultimate.py:

this code defines the UNet1D_ultimate model, an optimized 1D UNet architecture tailored for audio diffusion models (such as Mel-spectrogram generation).
It integrates several improvements, including FiLM-based timestep modulation, 
smooth upsampling, and sparse Cross-Attention, to enhance the model's performance and generation quality. 

**4.train:**

This code is used to train a model based on the Adan optimizer, supporting AMP, ema, and gradient clipping.
It loads the dataset, defines the model and optimizer, performs the training loop, and periodically saves checkpoints and logs.

**5.Sample:**

This code samples mel spectrograms from a single npz file using a trained diffusion model.
It supports optional classifier-free guidance (CFG) to enhance generation quality.
It loads conditional data (motion and lyrics) from the npz and performs sampling using the specified ckpt.

**6.val:**

This code is used to evaluate the difference between generated Mel spectrograms and real Mel spectrograms,
including quantitative metric calculation and visual comparison.
Evaluation metrics include MSE, SSIM, frame-level cosine similarity, statistical distribution error, and signal-to-noise ratio (SNR).
Due to the particularity of audio generation tasks, 
the evaluation metrics in this script do not fully reflect subjective listening experience and should be used for reference only.   

**7.testwav.py:**

This code loads a mel spectrogram from a .npz file, uses BigVGAN tovocode it into a waveform, and saves the waveform as a .wav file.




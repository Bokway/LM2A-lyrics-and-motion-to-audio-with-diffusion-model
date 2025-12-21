import numpy as np
import torch
import soundfile as sf
import bigvgan

# this script loads a mel spectrogram from a .npz file and uses BigVGAN to vocode it into a waveform, then saves the waveform as a .wav file.
# make sure you have bigvgan installed: pip install bigvgan


# ======== 1. load npz ========
'/mnt/mydev2/Bob/LM2ANew/npz'
path = '/mnt/mydev2/Bob/LM2ANew/npz/sample_00001700.npz'
#path = '/mnt/mydev2/Bob/LM2ANew/samples/sample_00001750_gen.npz'
path = '/mnt/mydev2/Bob/LM2ANew/samples_adan/sample_00000016_gen.npz'
#path = '/mnt/mydev2/Bob/LM2ANew/npz_split/train/sample_00000050.npz'

path = '/mnt/mydev2/Bob/LM2ANew/testnpz/sample_00000020.npz'
path = '/mnt/mydev2/Bob/LM2ANew/testnpzwav/sample_00000129_gen.npz'

path ='/mnt/mydev2/Bob/LM2ANew/ceshi/20000ckpt/sample_00001515_gen_mel.npz'

#path ='/mnt/mydev2/Bob/LM2ANew/testnpzwav/sample_00000020_gen.npz'

# just load mel from the npz . Attention to the right path of the npz file


data = np.load(path, allow_pickle=True)

mel = data["mel"].astype(np.float32)  # (80, T)

print("mel shape:", mel.shape)     

# ======== 2. prepare mel tensor ========


mel = torch.tensor(mel)[None].to("cuda:0")  # (1, 80, T)

print("mel shape:", mel.shape)     

sr = int(data["sr"])

# ======== 3. load BigVGAN ========
model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_22khz_80band',
    use_cuda_kernel=False
)
model.remove_weight_norm()
model = model.eval().to("cuda:0")


# ======== 4. vocoding ========
with torch.inference_mode():
    wav_gen = model(mel)        # (1, 1, T)
wav = wav_gen.squeeze().cpu().numpy()

# ======== 5. save ========
save_path = '/mnt/mydev2/Bob/LM2ANew/reconstructed.wav'
save_path = '/mnt/mydev2/Bob/LM2ANew/samples/sample_00001700_origin.wav'
#save_path = '/mnt/mydev2/Bob/LM2ANew/samples/sample_00001750_gen.wav'
save_path = '/mnt/mydev2/Bob/LM2ANew/samples_adan/sample_00000016_gen.wav'
#save_path = '/mnt/mydev2/Bob/LM2ANew/samples_adan/sample_00000050_origin.wav'

save_path = '/mnt/mydev2/Bob/LM2ANew/testnpzwav/sample_00000020_reconstructed.wav'

save_path = '/mnt/mydev2/Bob/LM2ANew/testnpzwav/sample_00000129_gen_reconstructed.wav'

save_path ='/mnt/mydev2/Bob/LM2ANew/ceshi/20000ckpt/sample_00001515_gen_reconstructed.wav'

# save waveform
# attention to the right path of the saved .wav file


sf.write(save_path, wav.astype("float32"), sr)

print("Done! Saved to:", save_path)

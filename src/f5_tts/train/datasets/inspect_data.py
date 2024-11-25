import matplotlib.pyplot as plt
import torch
from f5_tts.model.dataset import load_dataset
import torchaudio

# Load the dataset
dataset = load_dataset(
    dataset_name="mls_english",
    tokenizer="pinyin"
)

mel_scale = torchaudio.transforms.MelScale(
    n_mels=128,  # Number of mel filterbanks
    n_stft=1026, # Match your spectrogram's first dimension
    sample_rate=16000
)

# Get the first sample
sample = dataset[0]

# The sample will be a dictionary with:
spectrogram = sample['mel_spec']
print(spectrogram.shape)

# Plot magnitude spectrogram
plt.imshow(torch.norm(spectrogram, dim=0).numpy(), aspect='auto', origin='lower')
plt.colorbar(label='Magnitude')
plt.title('Magnitude Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('magnitude_spectrogram.png')
plt.close()



mel_spectrogram = mel_scale(torch.norm(spectrogram, dim=0).pow(2))
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

plt.imshow(mel_spectrogram_db.numpy(), aspect='auto', origin='lower')
plt.colorbar(label='Amplitude (dB)')
plt.title('Mel Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.savefig('mel_spectrogram.png')
plt.close()

# Plot phase spectrogram
plt.imshow(torch.atan2(spectrogram[..., 1], spectrogram[..., 0]).numpy(), aspect='auto', origin='lower')
plt.colorbar(label='Phase (radians)')
plt.title('Phase Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Radians')
plt.tight_layout()
plt.savefig('phase_spectrogram.png')
plt.close()
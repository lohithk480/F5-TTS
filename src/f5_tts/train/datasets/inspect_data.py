import matplotlib.pyplot as plt
import torch
from f5_tts.model.dataset import load_dataset
import torchaudio

# Load the dataset
dataset = load_dataset(
    dataset_name="mls_english",
    tokenizer="pinyin"
)

# Get the first sample
sample = dataset[1]

# The sample will be a dictionary with:
spectrogram = sample['mel_spec']
audio_path = sample['audio_path']
audio, source_sample_rate = torchaudio.load(audio_path)
torchaudio.save('original_audio.wav', audio, sample_rate=source_sample_rate)


print(f"spectrogram.shape: {spectrogram.shape}")
print(f"spectrogram.dtype: {spectrogram.dtype}")


# Plot Log Magnitude spectrogram
plt.imshow(spectrogram[:513].numpy(), aspect='auto', origin='lower')
plt.colorbar(label='Magnitude')
plt.title('Magnitude Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('magnitude_spectrogram.png')
plt.close()

# Plot phase spectrogram
plt.imshow(spectrogram[513:].numpy(), origin='lower')
plt.colorbar(label='Phase (radians)')
plt.title('Phase Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Radians')
plt.tight_layout()
plt.savefig('phase_spectrogram.png')
plt.close()

#Spectrogram Inverse

# Convert magnitude and phase to complex spectrogram
log_magnitude = spectrogram[:513]
magnitude = torch.exp(log_magnitude)
phase = spectrogram[513:]
complex_spectrogram = torch.complex(
    magnitude * torch.cos(phase),
    magnitude * torch.sin(phase)
)

complex_spectrogram = complex_spectrogram.unsqueeze(0) #Add back in batch dimension

print(f"complex_spectrogram.shape {complex_spectrogram.shape}")

# Create inverse spectrogram transform
inverse_spec = torchaudio.transforms.InverseSpectrogram(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
).to(spectrogram.device)

inverse_spectrogram = inverse_spec(complex_spectrogram)
print(f"inverse_spectrogram.shape: {inverse_spectrogram.shape}")

torchaudio.save('inverse_spectrogram.wav', inverse_spectrogram, sample_rate=16000)


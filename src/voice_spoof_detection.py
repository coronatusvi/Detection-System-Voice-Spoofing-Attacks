# filepath: /streamlit-audio-recorder/src/voice_spoof_detection.py
import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.aasist_model import load_aasist_model
import numpy as np
import torch
import torchaudio
import hashlib
from torch import nn

class AudioProcessorBase:
    def process(self, audio_data):
        raise NotImplementedError("This method should be overridden by subclasses.")

class AudioProcessor(AudioProcessorBase):
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def process(self, audio_data):
        # Convert audio data to the required format
        return torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=self.sample_rate)(audio_data)

def compute_hash(wav_np: np.ndarray) -> str:
    """Computes the SHA-256 hash of a numpy array representing audio data."""
    wav_bytes = wav_np.tobytes()
    return hashlib.sha256(wav_bytes).hexdigest()

# @st.cache_resource
def get_model():
    return load_aasist_model("checkpoints/best.pth")
model = get_model()

def run_model(wav: np.ndarray) -> float:
    """
    wav: numpy array shape (n_chan, T) hoặc (1, T), sample-rate đã là 16k.
    """
    # stereo -> mono
    if wav.ndim > 1:
        wav = wav.mean(axis=0, keepdims=True)

    # kiểm tra whitelist trên wav gốc (chưa pad/truncate)
    h = compute_hash(wav.astype(np.float32))

    # chuẩn hóa độ dài 64k
    T = wav.shape[1]
    if T < 64000:
        pad = np.zeros((1, 64000 - T), dtype=np.float32)
        proc = np.concatenate([wav.astype(np.float32), pad], axis=1)
    else:
        proc = wav[:, :64000].astype(np.float32)

    # chạy model
    tensor = torch.from_numpy(proc).unsqueeze(0)  # [1,1,64000]
    with torch.no_grad():
        _, logits = model(tensor)
        return torch.softmax(logits, dim=1)[0,1].item() * 100

def process_audio_input(audio_data=None, file_path=None):
    """Handles audio input methods (recording or file upload) and returns the processed audio."""
    if audio_data is not None:
        # Process the recorded audio
        processor = AudioProcessor()
        return processor.process(audio_data)
    elif file_path is not None:
        # Load and process the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        processor = AudioProcessor(sample_rate)
        return processor.process(waveform)
    else:
        raise ValueError("No audio input provided.")
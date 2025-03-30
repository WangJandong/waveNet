import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset import LJDataset
from src.wavenet import WaveNet, encode_mu_law, quantize
from config import ModelConfig
from src.preprocessing import MelSpectrogram
from config import MelSpectrogramConfig
import warnings
from tqdm import tqdm
import os
import torch.nn.functional as F
import torchaudio
from torchviz import make_dot
warnings.filterwarnings('ignore')
audio_path = r'LJSpeech-1.1\\wavs\\LJ001-0008.wav'
output_audio_path = "./predicted_audio.wav"
model_path = ModelConfig.model_path


def gen():
    import torchaudio
    import matplotlib.pyplot as plt
    from tqdm import tqdm  # 导入 tqdm 库

    config = MelSpectrogramConfig()
    mel_spectrogram = MelSpectrogram(config)
    
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(audio_path)

    # 调整采样率
    if sample_rate != config.sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.sr)(waveform)

    print("waveform", waveform.shape)

    # 生成梅尔频谱图
    mel_spec = mel_spectrogram(waveform)
    print("mel_spec", mel_spec.shape)

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = WaveNet()
    mel_spec = mel_spec.to(device)

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    # model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 生成预测音频
    print("Generating audio... mel_spec.shape == ",mel_spec.shape)
    predicted_audio = model.inference(mel_spec)

    # 保存生成的音频
    torchaudio.save(output_audio_path, predicted_audio.cpu(), sample_rate=config.sr)
    print(f"Predicted audio saved to {output_audio_path}")


def draw_png():
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')  # 使用非GUI后端
    import matplotlib.pyplot as plt

    # 加载原始音频和生成音频
    original_audio, original_sr = librosa.load(audio_path)
    predicted_audio, predicted_sr = librosa.load(output_audio_path)

    # 创建一个大图，包含两个子图
    plt.figure(figsize=(14, 10))

    # 绘制原始音频波形图
    plt.subplot(2, 1, 1)  # 2行1列，第1个子图
    librosa.display.waveshow(original_audio, sr=original_sr)
    plt.title('Original Audio Waveform (LJ001-0002.wav)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 绘制生成音频波形图
    plt.subplot(2, 1, 2)  # 2行1列，第2个子图
    librosa.display.waveshow(predicted_audio, sr=predicted_sr)
    plt.title('Predicted Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 保存图像
    plt.tight_layout()  # 调整子图间距
    plt.show()
    plt.savefig('combined_waveform.png')
    print("Combined waveform saved to combined_waveform.png")
   

if __name__ == '__main__':
    gen()
    draw_png()



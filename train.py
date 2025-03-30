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

warnings.filterwarnings('ignore')

# 创建存储梅尔频谱图的目录
os.makedirs("spec", exist_ok=True)

def compute_and_save_mel_spectrograms(dataset, save_dir):
    """
    计算并保存梅尔频谱图到本地
    :param dataset: LJDataset 实例
    :param save_dir: 保存梅尔频谱图的目录
    """
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to("cpu")  # 使用 CPU 计算
    for idx in tqdm(range(len(dataset)), desc="Computing Mel Spectrograms"):
        wav = dataset[idx]
        wav = wav.unsqueeze(0)  # 增加 batch 维度
        mel_spec = featurizer(wav.to("cpu")).squeeze(0)  # 计算梅尔频谱图
        save_path = os.path.join(save_dir, f"{idx}.pt")
        torch.save(mel_spec, save_path)  # 保存梅尔频谱图

class PrecomputedLJDataset(LJDataset):
    """
    加载预处理的梅尔频谱图
    """
    def __init__(self, df, spec_dir):
        super().__init__(df)
        self.spec_dir = spec_dir

    def __getitem__(self, idx):
        wav = super().__getitem__(idx)
        mel_spec_path = os.path.join(self.spec_dir, f"{idx}.pt")
        mel_spec = torch.load(mel_spec_path)  # 加载预处理的梅尔频谱图
        return wav, mel_spec

def main():
    # 读取数据
    df = pd.read_csv("LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
    train, test = train_test_split(df, test_size=0.2, random_state=10)

    # 预计算梅尔频谱图
    train_spec_dir = "spec/train"
    test_spec_dir = "spec/test"
    os.makedirs(train_spec_dir, exist_ok=True)
    os.makedirs(test_spec_dir, exist_ok=True)

    # 如果梅尔频谱图未计算，则计算并保存
    if not os.listdir(train_spec_dir):
        train_dataset = LJDataset(train)
        compute_and_save_mel_spectrograms(train_dataset, train_spec_dir)
    if not os.listdir(test_spec_dir):
        test_dataset = LJDataset(test)
        compute_and_save_mel_spectrograms(test_dataset, test_spec_dir)

    # 创建数据集和数据加载器
    train_dataset = PrecomputedLJDataset(train, train_spec_dir)
    test_dataset = PrecomputedLJDataset(test, test_spec_dir)

    model_config = ModelConfig()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=model_config.batch_size,
                                  num_workers=model_config.num_workers,
                                  shuffle=False,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=model_config.batch_size,
                                 num_workers=model_config.num_workers,
                                 pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    # self, num_mels=80, kernel_size=2, residual_channels=120, skip_channels=240, dilation_depth=8, dilation_repeat=2, quantized_values=256
    # model = WaveNet(80,2,20,40,6,2,128)
    model = WaveNet()

    if model_config.from_pretrained:
        model.load_state_dict(torch.load(model_config.model_path))
        print("load from: "+ model_config.model_path)
    model.to(device)
    
    # 计算并打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.lr)
    best_loss = 10.0

    # 训练循环
    for epoch in range(0, model_config.num_epochs + 1):
        model.to(device)
        train_losses = []
        train_accuracy = []
        model.train()
        print(f"Epoch {epoch}/{model_config.num_epochs}")

        # 使用 tqdm 包装 train_dataloader
        pbar = tqdm(train_dataloader, desc="Training", unit="batch")
        for batch_idx, (wavs, mels) in enumerate(pbar):
            wavs = wavs.to(device)
            mels = mels.to(device)

            zero_frame = torch.zeros((wavs.shape[0], 1)).to(device)
            padded_wavs = torch.cat([zero_frame, wavs[:, :-1]], dim=1)

            outputs = model(padded_wavs, mels)

            classes = outputs.argmax(dim=1)
            quantized_wavs = quantize(encode_mu_law(wavs))
            loss = nn.CrossEntropyLoss()(outputs, quantized_wavs)

            accuracy = (classes == quantized_wavs).sum().item() / classes.shape[-1] / classes.shape[0]
            train_accuracy.append(accuracy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
            train_losses.append(loss.item())

        # 验证循环
        model.eval()
        val_losses = []
        val_accuracy = []
        with torch.no_grad():
            # 使用 tqdm 包装 test_dataloader
            for wavs, mels in tqdm(test_dataloader, desc="Validation", unit="batch"):
                wavs = wavs.to(device)
                mels = mels.to(device)
                zeros = torch.zeros((wavs.shape[0], 1)).to(device)
                padded_wavs = torch.cat([zeros, wavs[:, :-1]], dim=1)

                outputs = model(padded_wavs, mels)
                classes = outputs.argmax(dim=1)
                quantized_wavs = quantize(encode_mu_law(wavs))

                accuracy = (classes == quantized_wavs).sum().item() / classes.shape[-1] / classes.shape[0]
                val_accuracy.append(accuracy)

                # loss = F.mse_loss(outputs, quantized_wavs)
                loss = nn.CrossEntropyLoss()(outputs, quantized_wavs)
                val_losses.append(loss.item())

        # 计算平均损失和准确率
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = np.mean(train_accuracy)
        val_acc = np.mean(val_accuracy)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ModelConfig.model_path)
            print("Saved model checkpoint!")
            print("best_loss:", best_loss)


   


if __name__ == '__main__':
    main()
    


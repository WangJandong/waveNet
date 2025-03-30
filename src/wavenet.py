import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from .utiles import *

class CausalConv1d(nn.Module):
    """带因果约束的1维卷积（考虑时间先后关系）"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, bias=bias
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]  # 移除右侧填充以保持因果性
        return x


class WaveNetBlock(nn.Module):
    """WaveNet基本块，包含门控机制和残差连接"""
    def __init__(self, dilation: int, dropout_prob: float = 0.2):
        super().__init__()
        # 门控卷积分支
        self.dil_sigmoid = CausalConv1d(120, 120, kernel_size=2, dilation=dilation)
        self.dil_tanh = CausalConv1d(120, 120, kernel_size=2, dilation=dilation)
        
        # 条件特征变换
        self.mel_1x1_sigmoid = nn.Conv1d(80, 120, kernel_size=1)
        self.mel_1x1_tanh = nn.Conv1d(80, 120, kernel_size=1)
        
        # 输出变换
        self.skip_1x1 = nn.Conv1d(120, 240, kernel_size=1)
        self.res_1x1 = nn.Conv1d(120, 120, kernel_size=1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        # 门控机制
        sigmoid_out = torch.sigmoid(
            self.dil_sigmoid(x) + self.mel_1x1_sigmoid(h)
        )
        tanh_out = torch.tanh(
            self.dil_tanh(x) + self.mel_1x1_tanh(h)
        )
        gated_out = sigmoid_out * tanh_out
        
        # 残差连接和跳跃连接
        skip_out = self.skip_1x1(gated_out)
        res_out = self.res_1x1(gated_out) + x
        
        return res_out, skip_out


class WaveNet(nn.Module):
    """完整的WaveNet模型实现"""
    def __init__(self, num_mels: int = 80, kernel_size: int = 2, 
                 residual_channels: int = 120, skip_channels: int = 240,
                 dilation_depth: int = 8, dilation_repeat: int = 2, 
                 quantized_values: int = 256):
        super().__init__()
        
        # 配置扩张系数
        self.dilations = [2 ** i for i in range(dilation_depth)] * dilation_repeat
        self.receptive_field = (kernel_size - 1) * sum(self.dilations) + 1
        
        # 初始变换层
        self.start_conv = nn.Conv1d(1, residual_channels, kernel_size=1)
        
        # 梅尔频谱上采样
        self.upsampling = nn.Sequential(
            nn.ConvTranspose1d(num_mels, num_mels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(num_mels, num_mels, kernel_size=800, stride=250, padding=150)
        )
        
        # WaveNet块堆叠
        self.blocks = nn.ModuleList([
            WaveNetBlock(dilation) for dilation in self.dilations
        ])
        
        # 后处理网络
        self.postprocess = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, quantized_values, kernel_size=1)
        )

    def forward(self, x: Tensor, h: Tensor, training: bool = True) -> Tensor:
        """
        前向传播
        Args:
            x: 输入音频波形 (B, T)
            h: 条件梅尔频谱 (B, num_mels, T')
            training: 是否训练模式
        Returns:
            预测的量化分布 (B, quantized_values, T)
        """
        # 输入预处理
        x = encode_mu_law(x).unsqueeze(1)  # (B, 1, T)
        output = self.start_conv(x)
        
        # 条件特征上采样
        if training:
            h = self.upsampling(h)
        
        # 通过所有WaveNet块
        skip_connections = []
        for block in self.blocks:
            output, skip = block(output, h)
            skip_connections.append(skip)
        
        # 合并跳跃连接并后处理
        output = sum(skip_connections)
        return self.postprocess(output)

    def inference(self, mel: Tensor, batch_size: Optional[int] = None) -> Tensor:
        """
        从梅尔频谱生成音频（自回归方式）
        Args:
            mel: 输入梅尔频谱 (B, num_mels, T)
            batch_size: 分批处理大小
        Returns:
            生成的音频波形 (B, T')
        """
        if batch_size and mel.size(0) > batch_size:
            return self._batch_inference(mel, batch_size)
        return self._single_inference(mel)
    
    def _batch_inference(self, mel: Tensor, batch_size: int) -> Tensor:
        """分批处理大输入"""
        results = []
        for i in range(0, mel.size(0), batch_size):
            batch = mel[i:i + batch_size]
            results.append(self._single_inference(batch))
        return torch.cat(results, dim=0)
    
    def _single_inference(self, mel: Tensor) -> Tensor:
        """处理单个批次"""
        device = mel.device
        B, _, T = mel.shape
        
        # 初始化波形缓冲区
        waveform = torch.zeros(B, self.receptive_field + T, device=device)
        waveform[:, :self.receptive_field] = 0.001  # 小初始值
        
        # 上采样条件特征
        h = self.upsampling(mel)
        
        # 自回归生成
        for t in range(T):
            # 获取当前输入窗口
            x = waveform[:, t:t + self.receptive_field]
            h_t = h[:, :, t:t + 1].expand(-1, -1, self.receptive_field)
            
            # 前向计算
            with torch.no_grad():
                logits = self(x, h_t, training=False)
                pred = torch.argmax(logits[:, :, -1], dim=1)
                sample = decode_mu_law(dequantize(pred.unsqueeze(1)))
            
            # 更新波形
            waveform[:, t + self.receptive_field] = sample.squeeze(1)
        
        return waveform[:, self.receptive_field:]
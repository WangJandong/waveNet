import torch
import torch.nn as nn
import numpy as np

def encode_mu_law(x: torch.Tensor, mu: float = 255.0) -> torch.Tensor:
    """
    Mu-law音频编码算法，将输入信号压缩到[-1, 1]范围
    
    参数:
        x: 输入张量（任意形状），应在[-1, 1]范围内
        mu: 压缩参数（默认为255，对应8位编码）
    
    返回:
        编码后的张量，范围[-1, 1]
    """
    # 将输入裁剪到[-1, 1]范围，避免数值问题
    x = torch.clamp(x, -1.0, 1.0)
    # 使用mu-law公式进行编码
    return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / np.log1p(mu)

def decode_mu_law(y: torch.Tensor, mu: float = 255.0) -> torch.Tensor:
    """
    Mu-law音频解码算法，将压缩信号还原到原始范围
    
    参数:
        y: 编码后的张量，范围[-1, 1]
        mu: 压缩参数（应与编码时一致）
    
    返回:
        解码后的张量，恢复原始范围
    """
    # 将输入裁剪到[-1, 1]范围，避免数值问题
    y = torch.clamp(y, -1.0, 1.0)
    # 使用mu-law逆公式进行解码
    return torch.sign(y) * (torch.pow(1.0 + mu, torch.abs(y)) - 1.0) / mu

def quantize(mu_law: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    将mu-law编码信号量化为离散值
    
    参数:
        mu_law: 编码后的张量，范围[-1, 1]
        bits: 量化位数（默认为8，对应256个量化级）
    
    返回:
        量化后的张量，范围[0, 2^bits-1]，类型为long
    """
    n_levels = 2 ** bits  # 计算量化级数
    # 将[-1, 1]范围映射到[0, n_levels-1]并四舍五入
    return torch.clamp(
        torch.floor((mu_law + 1.0) * 0.5 * n_levels + 0.5), 
        0, 
        n_levels - 1
    ).long()

def dequantize(quantized: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    将量化值转换回连续的mu-law编码信号
    
    参数:
        quantized: 量化后的张量，范围[0, 2^bits-1]
        bits: 量化位数（应与量化时一致）
    
    返回:
        反量化后的张量，范围[-1, 1]
    """
    n_levels = 2 ** bits  # 计算量化级数
    # 将[0, n_levels-1]范围映射回[-1, 1]
    return (quantized.float() - 0.5) / (n_levels - 1) * 2.0 - 1.0



# WaveNet 语音合成项目

基于WaveNet架构的深度学习语音合成系统

## 主要特性
- 基于原始WaveNet论文实现
- 支持LJ Speech数据集
- 可扩展的模型架构
- 训练过程可视化支持

## 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7 (GPU训练需要)

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 预处理数据集 (需要先下载LJ Speech数据集)
python src/preprocessing.py

# 开始训练
python train.py 

# 生成语音
python gen.py 
```

## 项目结构
```
.
├── config.py        # 模型配置文件
├── train.py         # 训练主程序
├── gen.py           # 语音生成脚本
├── src/             # 核心源代码
│   ├── wavenet.py   # 模型架构
│   ├── dataset.py   # 数据加载器
│   └── preprocessing.py # 音频预处理
├── spec/            # 预处理后的频谱数据
└── LJSpeech-1.1/    # 原始语音数据集
```

## 数据集准备
1. 下载LJ Speech数据集并解压到项目根目录
2. 确保目录结构为：
   ```
   LJSpeech-1.1/
   ├── metadata.csv
   └── wavs/
       ├── LJ001-0001.wav
       └── ... 
   ```

## 训练配置
在`config.py`中修改以下关键参数：
```python
batch_size = 32
num_epochs = 1000
learning_rate = 0.001
audio_sample_rate = 22050  # 必须与数据集采样率一致
```

## 许可证
MIT License

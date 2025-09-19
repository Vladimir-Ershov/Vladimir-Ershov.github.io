# MLRecap - Machine Learning Reimplementations

A collection of machine learning algorithms and neural networks implemented from scratch using PyTorch. This project focuses on understanding the fundamentals by recreating popular ML models and techniques.

## 🚀 Quick Start

### Prerequisites
- [Pixi](https://pixi.sh) package manager
- Python 3.9+
- Platform-specific requirements (see below)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd projects/MLRecap
   ```

2. **Install the PyTorch environment:**
   ```bash
   pixi install -e pytorch
   ```

3. **Activate the environment:**
   ```bash
   pixi shell -e pytorch
   ```

4. **Test your setup:**
   ```bash
   python device_info.py
   ```

## 🖥️ Platform Configuration

This project uses **platform-specific PyTorch builds** to optimize performance on different systems:

### macOS (Intel & Apple Silicon)
- **PyTorch**: CPU-only with MPS (Metal Performance Shaders) acceleration
- **GPU Support**: Apple Silicon Macs get GPU acceleration via MPS
- **Memory**: Uses system RAM, GPU memory handled automatically
- **Performance**: MPS provides significant speedup for compatible operations

### Linux (CUDA)
- **PyTorch**: CUDA 12.4 optimized builds
- **GPU Support**: NVIDIA GPUs with CUDA compute capability
- **Memory**: Dedicated GPU memory + system RAM
- **Performance**: Full CUDA acceleration for all operations

### Windows
- **Status**: Not currently configured (contributions welcome!)
- **Recommendation**: Use WSL2 with Linux configuration

## 📊 Performance Expectations

Based on `device_info.py` benchmarks:

| Platform | Device | 2048x2048 MatMul | Notes |
|----------|--------|------------------|-------|
| M4 macOS | CPU | ~16ms | ARM optimizations |
| M4 macOS | MPS | ~285ms | GPU acceleration* |
| Linux | CPU | ~50-100ms | Intel/AMD x64 |
| Linux | CUDA | ~1-5ms | Depends on GPU |

*MPS performance varies by operation type and is actively improving

## 🛠️ Environment Details

### Dependencies
- **PyTorch**: 2.2.2+ (macOS) / 2.5+ (Linux CUDA)
- **NumPy**: <2.0 (compatibility constraint)
- **Scikit-learn**: Latest stable
- **Seaborn**: Data visualization
- **Skorch**: Scikit-learn compatible PyTorch wrapper
- **Pillow**: Image processing
- **psutil**: System information

### Environment Variables
Configure these for optimal performance:

```bash
# CPU threading (adjust based on your CPU cores)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# MPS memory management (macOS only)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
```

## 📁 Project Structure

```
MLRecap/
├── README.md              # This file
├── pixi.toml             # Cross-platform environment config
├── device_info.py        # System diagnostics script
├── torch/                # PyTorch implementations
│   ├── audio.ipynb       # Audio processing models
│   └── ...               # Other neural network implementations
├── local/                # Custom ML implementations
│   └── ...               # From-scratch algorithms
└── notebooks/            # Jupyter notebooks for experiments
    └── ...
```

## 🔧 Configuration Deep Dive

### Pixi Configuration (`pixi.toml`)

The project uses a **unified configuration** that automatically selects the appropriate PyTorch build:

```toml
# Supports all platforms
platforms = ["linux-64", "osx-64", "osx-arm64"]

# Linux gets CUDA builds
[feature.pytorch.target.linux-64.dependencies]
pytorch = { version = "2.5.*", build = "*cuda12.4*" }
torchvision = { build = "*cu124*" }
pytorch-cuda = "12.4.*"

# macOS gets CPU/MPS builds
[feature.pytorch.target.osx-arm64.dependencies]
pytorch = { version = ">=2.0,<3" }
torchvision = "*"
```

### Device Detection Logic

The environment automatically selects the best device:

1. **CUDA** if available (Linux with NVIDIA GPU)
2. **MPS** if available (Apple Silicon Macs)
3. **CPU** as fallback (Intel Macs, other systems)

### Memory Management

#### macOS (MPS)
- Unified memory architecture
- GPU memory is part of system RAM
- Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO` to control usage

#### Linux (CUDA)
- Separate GPU memory pool
- Monitor with `nvidia-smi`
- Use `torch.cuda.empty_cache()` to free memory

## 🚨 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'torch'"
```bash
# Ensure you're in the pytorch environment
pixi shell -e pytorch
python -c "import torch; print(torch.__version__)"
```

#### NumPy compatibility warnings
The project constrains NumPy to <2.0 for PyTorch compatibility. This is normal.

#### MPS out of memory (macOS)
```bash
# Reduce memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5
# Or use CPU for large models
device = torch.device('cpu')
```

#### CUDA out of memory (Linux)
```python
# Clear GPU cache
torch.cuda.empty_cache()
# Reduce batch size
batch_size = 32  # instead of 128
```

### Performance Tuning

#### CPU Optimization
```bash
# Set thread count to physical cores
export OMP_NUM_THREADS=$(nproc --all)
```

#### MPS Optimization (macOS)
```python
# Use appropriate data types
tensor = torch.tensor(data, dtype=torch.float32, device='mps')

# Avoid frequent CPU-GPU transfers
# Keep data on GPU for multiple operations
```

#### CUDA Optimization (Linux)
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
```

## 🤝 Contributing

1. **Environment**: Always test on both macOS and Linux if possible
2. **Device Compatibility**: Write device-agnostic code:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')
   ```
3. **Dependencies**: Add new dependencies to `pixi.toml` with version constraints

## 💡 Project Ideas

- CNN interactive dashboard for kernels
- Neural network visualization tools
- From-scratch implementations of popular architectures
- Comparative performance analysis across platforms

## 📝 Notes

- **macOS Intel**: CPU-only, no GPU acceleration
- **Apple Silicon**: MPS acceleration available but not all operations supported
- **Linux**: Full CUDA support with proper GPU drivers
- **Windows**: Use WSL2 with Linux configuration for best experience

## 🔗 Useful Commands

```bash
# Environment management
pixi install -e pytorch          # Install/update environment
pixi shell -e pytorch           # Activate environment
pixi list -e pytorch            # List installed packages

# System diagnostics
python device_info.py           # Comprehensive system info
python -c "import torch; print(torch.cuda.is_available())"  # Quick CUDA check
python -c "import torch; print(torch.backends.mps.is_available())"  # Quick MPS check

# Jupyter notebooks
pixi run -e pytorch jupyter lab  # Start Jupyter Lab
```

---

**Last updated**: September 2025
**Tested on**: macOS 15.5 (M4), Ubuntu 22.04 (CUDA 12.4)

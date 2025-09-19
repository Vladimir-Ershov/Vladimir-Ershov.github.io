#!/usr/bin/env python3
import torch
import platform
import psutil
import subprocess
import os

def get_system_info():
    """Get detailed system information"""
    print("System Information")
    print("=" * 50)
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")

    # macOS specific GPU info
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'Chipset Model:' in line:
                        print(f"GPU: {line.split(':')[1].strip()}")
                    elif 'Total Number of Cores:' in line:
                        print(f"GPU Cores: {line.split(':')[1].strip()}")
                    elif 'Memory:' in line and i > 0:
                        print(f"GPU Memory: {line.split(':')[1].strip()}")
        except:
            print("GPU: Unable to detect (system_profiler failed)")

def get_pytorch_info():
    """Get detailed PyTorch information"""
    print(f"\nPyTorch Information")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Build: {torch.version.git_version}")
    print(f"OpenMP Support: {torch.backends.openmp.is_available()}")
    print(f"MKL Support: {torch.backends.mkl.is_available()}")
    print(f"MKL-DNN Support: {torch.backends.mkldnn.is_available()}")

def get_device_info():
    """Get detailed device information"""
    print(f"\nAcceleration Devices")
    print("=" * 50)

    # CUDA info
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nCUDA Device {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print(f"  Max Threads per Block: {props.max_threads_per_block}")
            print(f"  Max Shared Memory: {props.max_shared_memory_per_block / 1024:.1f} KB")

    # MPS info
    mps_available = torch.backends.mps.is_available()
    print(f"\nMPS (Metal) Available: {mps_available}")
    if mps_available:
        print(f"MPS Built: {torch.backends.mps.is_built()}")
        try:
            # Test MPS allocation to get memory info
            test_tensor = torch.zeros(1, device='mps')
            print(f"MPS Device: mps:0")
            print(f"MPS Memory Allocated: {torch.mps.current_allocated_memory() / (1024**2):.1f} MB")
            print(f"MPS Memory Cached: {torch.mps.driver_allocated_memory() / (1024**2):.1f} MB")
            del test_tensor
        except Exception as e:
            print(f"MPS Memory Info: Unable to get ({e})")

def benchmark_devices():
    """Benchmark available devices"""
    print(f"\nPerformance Benchmark")
    print("=" * 50)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')

    for device in devices:
        try:
            torch.cuda.synchronize() if device == 'cuda' else None

            # Simple matrix multiplication benchmark
            size = 2048
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            import time
            start = time.time()
            c = torch.mm(a, b)

            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()

            end = time.time()

            print(f"{device.upper():4}: {size}x{size} matmul in {(end-start)*1000:.1f} ms")

        except Exception as e:
            print(f"{device.upper():4}: Benchmark failed ({e})")

def main():
    get_system_info()
    get_pytorch_info()
    get_device_info()

    # Get optimal device
    if torch.cuda.is_available():
        default_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        default_device = torch.device('mps')
    else:
        default_device = torch.device('cpu')

    print(f"\nRecommended Device: {default_device}")

    # Test tensor creation
    print(f"\nTensor Test")
    print("=" * 50)
    tensor = torch.randn(3, 4, device=default_device)
    print(f"Created tensor on {tensor.device}:")
    print(tensor)

    # Run benchmark
    benchmark_devices()

    print(f"\nEnvironment Variables")
    print("=" * 50)
    env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTORCH_MPS_HIGH_WATERMARK_RATIO']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

if __name__ == "__main__":
    main()
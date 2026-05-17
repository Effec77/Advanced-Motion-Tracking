# ONNX Runtime GPU Installation Guide

**Purpose**: Speed up Stage 2 (2D Pose Estimation) by 10-15x using GPU acceleration

**Current Status**: 
- ✅ PyTorch: CUDA-enabled
- ❌ ONNX Runtime: CPU-only

**Target Status**:
- ✅ PyTorch: CUDA-enabled
- ✅ ONNX Runtime: CUDA-enabled

---

## 🚀 Quick Installation (Recommended)

### Step 1: Check CUDA Version

```bash
nvidia-smi
```

Look for the CUDA version in the output (e.g., "CUDA Version: 12.1")

### Step 2: Uninstall CPU Version

```bash
pip uninstall onnxruntime
```

### Step 3: Install GPU Version

**For CUDA 11.x**:
```bash
pip install onnxruntime-gpu
```

**For CUDA 12.x**:
```bash
pip install onnxruntime-gpu
```

Note: `onnxruntime-gpu` supports both CUDA 11.x and 12.x as of version 1.16+

### Step 4: Verify Installation

```python
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
```

**Expected Output**:
```
Available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Current Output** (before fix):
```
Available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
```

---

## 📋 Detailed Requirements

### System Requirements

1. **NVIDIA GPU**
   - Compute Capability 3.5 or higher
   - Check: https://developer.nvidia.com/cuda-gpus

2. **CUDA Toolkit**
   - Version 11.x or 12.x
   - Download: https://developer.nvidia.com/cuda-downloads

3. **cuDNN**
   - Version 8.x (compatible with your CUDA version)
   - Download: https://developer.nvidia.com/cudnn

### Python Requirements

- Python 3.8 - 3.11 (3.12 support may be limited)
- pip 19.3 or later

---

## 🔧 Installation Methods

### Method 1: Standard Installation (Easiest)

```bash
# Activate your virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Uninstall CPU version
pip uninstall onnxruntime

# Install GPU version
pip install onnxruntime-gpu

# Verify
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Method 2: Specific Version

If you need a specific version:

```bash
# Check available versions
pip index versions onnxruntime-gpu

# Install specific version
pip install onnxruntime-gpu==1.16.3
```

### Method 3: With CUDA Version Specification

For older CUDA versions, you may need specific builds:

**CUDA 11.8**:
```bash
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

**CUDA 12.x**:
```bash
pip install onnxruntime-gpu
```

---

## ✅ Verification Steps

### 1. Check ONNX Runtime Version

```python
import onnxruntime as ort
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
```

### 2. Test CUDA Execution

```python
import onnxruntime as ort
import numpy as np

# Create a simple session with CUDA
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('model.onnx', providers=providers)

# Check which provider is being used
print(f"Active provider: {session.get_providers()}")
```

### 3. Run Validation Script

```bash
python football_app/backend/models/run_validation_minimal.py
```

**Before GPU** (CPU-only):
- Stage 2: ~1-2 FPS
- 250 frames: ~2-4 minutes

**After GPU** (CUDA-enabled):
- Stage 2: ~15-25 FPS
- 250 frames: ~10-15 seconds

---

## 🐛 Troubleshooting

### Issue 1: "CUDAExecutionProvider not available"

**Cause**: CUDA libraries not found

**Solution**:
1. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

2. Check CUDA path:
   ```bash
   echo $CUDA_HOME  # Linux/Mac
   echo %CUDA_PATH%  # Windows
   ```

3. Add CUDA to PATH:
   ```bash
   # Linux/Mac
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   
   # Windows
   set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;%PATH%
   ```

### Issue 2: "DLL load failed" (Windows)

**Cause**: Missing CUDA DLLs

**Solution**:
1. Install Visual C++ Redistributable:
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Verify CUDA DLLs are in PATH:
   ```cmd
   where cudart64_*.dll
   where cublas64_*.dll
   ```

### Issue 3: Version Mismatch

**Cause**: CUDA version incompatibility

**Solution**:
1. Check CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install matching onnxruntime-gpu:
   - CUDA 11.x: `pip install onnxruntime-gpu`
   - CUDA 12.x: `pip install onnxruntime-gpu`

### Issue 4: Out of Memory

**Cause**: GPU memory insufficient

**Solution**:
1. Reduce batch size in RTMPose
2. Process fewer frames at once
3. Use smaller model variant

---

## 📊 Performance Comparison

### Current Setup (CPU-only)

| Stage | Device | FPS | Time (250 frames) |
|-------|--------|-----|-------------------|
| Stage 2 (2D Pose) | CPU | 1-2 | 2-4 minutes |
| Stage 3 (3D Pose) | CUDA | 20-30 | 8-12 seconds |
| **Total** | Mixed | - | **~2-4 minutes** |

### With GPU (CUDA-enabled)

| Stage | Device | FPS | Time (250 frames) |
|-------|--------|-----|-------------------|
| Stage 2 (2D Pose) | CUDA | 15-25 | 10-15 seconds |
| Stage 3 (3D Pose) | CUDA | 20-30 | 8-12 seconds |
| **Total** | CUDA | - | **~20-30 seconds** |

**Speedup**: ~5-8x overall, ~10-15x for Stage 2

### Full Dataset (7 cameras × 270 frames = 1890 frames)

| Setup | Stage 2 Time | Stage 3 Time | Total Time |
|-------|--------------|--------------|------------|
| CPU-only | 15-30 min | 1-2 min | **~16-32 min** |
| GPU-enabled | 1-2 min | 1-2 min | **~2-4 min** |

**Speedup**: ~8-10x overall

---

## 🔍 Advanced Configuration

### Custom Provider Options

```python
import onnxruntime as ort

# Configure CUDA provider with options
cuda_provider_options = {
    'device_id': 0,  # GPU device ID
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
}

providers = [
    ('CUDAExecutionProvider', cuda_provider_options),
    'CPUExecutionProvider'
]

session = ort.InferenceSession('model.onnx', providers=providers)
```

### Multi-GPU Setup

```python
# Use specific GPU
cuda_options = {'device_id': 1}  # Use GPU 1
providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
```

### Fallback to CPU

```python
# Automatic fallback if CUDA not available
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('model.onnx', providers=providers)

# Check which provider is actually used
print(f"Using provider: {session.get_providers()[0]}")
```

---

## 📦 Alternative: Docker with GPU Support

If you prefer containerized deployment:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install onnxruntime-gpu torch torchvision

# Copy your code
COPY . /app
WORKDIR /app

# Run validation
CMD ["python3", "football_app/backend/models/run_validation.py"]
```

Run with GPU:
```bash
docker run --gpus all your-image-name
```

---

## 🎯 Quick Checklist

Before running full validation:

- [ ] CUDA Toolkit installed
- [ ] cuDNN installed
- [ ] `nvidia-smi` shows GPU
- [ ] `onnxruntime` uninstalled
- [ ] `onnxruntime-gpu` installed
- [ ] `CUDAExecutionProvider` in available providers
- [ ] Test run on 50 frames successful
- [ ] GPU memory sufficient (check `nvidia-smi`)

---

## 📚 Additional Resources

### Official Documentation
- ONNX Runtime: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- cuDNN: https://developer.nvidia.com/cudnn

### Compatibility Matrix
- ONNX Runtime versions: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
- CUDA compatibility: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

### Community Support
- ONNX Runtime GitHub: https://github.com/microsoft/onnxruntime
- Issues: https://github.com/microsoft/onnxruntime/issues

---

## 💡 Pro Tips

1. **Check GPU utilization during inference**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Profile performance**:
   ```python
   import time
   start = time.time()
   # Run inference
   print(f"Time: {time.time() - start:.2f}s")
   ```

3. **Monitor memory usage**:
   ```python
   import torch
   print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

4. **Warm-up runs**:
   - First inference is slower (model loading)
   - Run 2-3 warm-up frames before timing

5. **Batch processing**:
   - Process multiple frames in batches for better GPU utilization
   - Adjust batch size based on GPU memory

---

**Document Version**: 1.0  
**Last Updated**: February 9, 2026  
**Status**: Ready for Implementation

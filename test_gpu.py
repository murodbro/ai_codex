#!/usr/bin/env python3
"""
GPU Test Script - Check if CUDA and GPU acceleration are working
"""
import torch
import sys

def test_gpu():
    print("🔍 Testing GPU Support...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
        
        # Test tensor operations on GPU
        print("\n🧪 Testing GPU operations...")
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU matrix multiplication test completed in {gpu_time:.4f}s")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
    else:
        print("❌ CUDA not available. GPU acceleration will not work.")
        print("Possible issues:")
        print("- PyTorch CPU-only version installed")
        print("- CUDA drivers not installed")
        print("- Docker not configured for GPU")
        
    return cuda_available

if __name__ == "__main__":
    test_gpu()

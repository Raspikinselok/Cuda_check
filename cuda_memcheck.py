import numpy as np
from numba import cuda
import os
import subprocess

def check_cuda_version():
    cuda_version_cmd = "nvcc --version"
    nvidia_smi_cmd = "nvidia-smi"
    
    # Check CUDA version
    print("CUDA Version:")
    try:
        cuda_version_output = subprocess.check_output(cuda_version_cmd.split(), stderr=subprocess.STDOUT).decode()
        print(cuda_version_output)
    except subprocess.CalledProcessError as e:
        print("Error:", e.output.decode())
    
    # Check NVIDIA-SMI info
    print("\nNVIDIA-SMI:")
    try:
        nvidia_smi_output = subprocess.check_output(nvidia_smi_cmd.split(), stderr=subprocess.STDOUT).decode()
        print(nvidia_smi_output)
    except subprocess.CalledProcessError as e:
        print("Error:", e.output.decode())

# Test the function
check_cuda_version()


@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    c[i] = a[i] + b[i]

N = 100
a = np.arange(N)
b = np.arange(N)
c = np.zeros(N)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

threads_per_block = 32
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

result = d_c.copy_to_host()
print(result)


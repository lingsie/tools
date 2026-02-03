import torch


# PyTorch
print("PyTorch version:", torch.__version__)


# CUDA
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)


# GPU
if cuda_available:
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("No GPU available.")


# addition
a = torch.tensor([1.0, 2.0], device='cuda' if cuda_available else 'cpu')
b = torch.tensor([3.0, 4.0], device='cuda' if cuda_available else 'cpu')
c = a + b
print("Tensor addition result:", c)


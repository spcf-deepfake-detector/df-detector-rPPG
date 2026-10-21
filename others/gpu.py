import torch

print("Pytorch version:", torch.__version__)
print("CUDA: ", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(0))
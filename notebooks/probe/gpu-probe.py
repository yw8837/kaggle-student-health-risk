# GPU/torch 환경 프로브
import torch, sys
print("python", sys.version)
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "n=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
print("arch list:", torch.cuda.get_arch_list())
try:
    x = torch.randn(4, 4, device="cuda")
    print("matmul ok:", (x @ x).sum().item())
except Exception as e:
    print("CUDA COMPUTE FAIL:", e)

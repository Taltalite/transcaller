import torch
import time

# 检查CUDA
if not torch.cuda.is_available():
    print("CUDA不可用！")
    exit()

device = torch.device("cuda")
x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

print("开始热身...")
# 热身
for _ in range(10):
    torch.mm(x, y)

print("开始测试计算速度...")
start = time.time()
for _ in range(100):
    torch.mm(x, y)
torch.cuda.synchronize() # 等待GPU算完
end = time.time()

print(f"100次 5000x5000 矩阵乘法耗时: {end - start:.2f} 秒")
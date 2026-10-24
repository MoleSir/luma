import torch
import torch.nn as nn

assert torch.cuda.is_available()

device = "cuda"

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
).to(device).eval()

# 假设输入 shape 在每次执行时都固定
x = torch.randn(32, 1024, device=device)

# 1. 创建静态输入缓冲区
static_x = torch.empty_like(x)
static_x.copy_(x)

# 2. 在专用 stream 上预热
warmup_stream = torch.cuda.Stream()
warmup_stream.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(warmup_stream):
    for _ in range(3):
        _ = model(static_x)

torch.cuda.current_stream().wait_stream(warmup_stream)

# 3. 捕获 GPU 操作
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    static_y = model(static_x)

# 4. 重复执行
def run(x):
    # 更新输入数据，不改变 static_x 的存储地址
    static_x.copy_(x)

    # 重放之前捕获的 GPU 工作
    graph.replay()

    return static_y

# 测试
y = run(x)
print(y.shape)  # torch.Size([32, 512])
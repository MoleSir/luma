# Makefile —— 手写 CUDA C++ 学习工程（无 cmake，直接 nvcc）
#
# GPU：这台机器 CUDA 枚举里 device0-3 是 RTX 4090 (sm_89)，device4-5 是
# H200 (sm_90)，device6 是 A100 (sm_80)。这里编成 fat binary 同时覆盖
# sm_89/sm_90，任何一张都能跑。用哪张卡运行时选：
#     CUDA_DEVICE=0 ./alloc_demo   # 默认，4090
#     CUDA_DEVICE=4 ./alloc_demo   # H200

CUDA_HOME ?= /home/taojun/opt/cuda-12.2
NVCC      := $(CUDA_HOME)/bin/nvcc
GENCODE   := -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90
CXXFLAGS  := -O2 -std=c++17

TARGET := alloc_demo

all: $(TARGET)

$(TARGET): allocator.h allocator.cu main.cu
	$(NVCC) $(GENCODE) $(CXXFLAGS) allocator.cu main.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

# 按需覆盖:  make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
CUDA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(which nvcc))))
NCCL_HOME ?= $(CUDA_HOME)
NVCC      ?= $(CUDA_HOME)/bin/nvcc

CXXFLAGS := -O2 -std=c++17 -I$(NCCL_HOME)/include
LDFLAGS  := -L$(NCCL_HOME)/lib -L$(NCCL_HOME)/lib64 -L$(CUDA_HOME)/lib64
LDLIBS   := -lnccl -lcudart

TARGET := nccl_multiprocess

$(TARGET): nccl_multiprocess.cpp
	$(NVCC) $(CXXFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

run: $(TARGET)
	./run_multiprocess.sh

clean:
	rm -f $(TARGET)

.PHONY: run clean

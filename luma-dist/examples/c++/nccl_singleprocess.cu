#include <cuda.h>
#include <nccl.h>
#include <cstdio>
#include <cstdint>

int main() {
    const int nDev = 2;
    const size_t count = 1;

    CUdevice devices[nDev];
    CUcontext ctxs[nDev];
    CUstream streams[nDev];
    CUdeviceptr sendbuff[nDev], recvbuff[nDev];
    ncclComm_t comms[nDev];

    // 1. 初始化 CUDA Driver
    cuInit(0);

    // 2. 获取设备，并保留各设备的 Primary Context
    for (int i = 0; i < nDev; i++) {
        cuDeviceGet(&devices[i], i);
        cuDevicePrimaryCtxRetain(&ctxs[i], devices[i]);
    }

    // 3. 初始化 NCCL 通信器
    // ncclCommInitAll 会为指定的 GPU 创建通信器
    ncclCommInitAll(comms, nDev, nullptr);

    // 4. 创建 Stream、分配显存、初始化输入
    for (int i = 0; i < nDev; i++) {
        cuCtxSetCurrent(ctxs[i]);

        cuStreamCreate(&streams[i], CU_STREAM_DEFAULT);

        cuMemAlloc(&sendbuff[i], count * sizeof(float));
        cuMemAlloc(&recvbuff[i], count * sizeof(float));

        float initValue = static_cast<float>(i);
        cuMemcpyHtoD(sendbuff[i], &initValue, sizeof(float));
    }

    // 5. 提交两个 GPU 的 AllReduce
    ncclGroupStart();

    for (int i = 0; i < nDev; i++) {
        cuCtxSetCurrent(ctxs[i]);
        ncclAllReduce(
            reinterpret_cast<void*>(static_cast<uintptr_t>(sendbuff[i])),
            reinterpret_cast<void*>(static_cast<uintptr_t>(recvbuff[i])),
            count,
            ncclFloat,
            ncclSum,
            comms[i],
            streams[i]
        );
    }

    ncclGroupEnd();

    // 6. 同步 Stream，并读取结果
    for (int i = 0; i < nDev; i++) {
        cuCtxSetCurrent(ctxs[i]);

        cuStreamSynchronize(streams[i]);

        float result = 0.0f;
        cuMemcpyDtoH(&result, recvbuff[i], sizeof(float));
        printf("GPU %d: result = %.0f (expected 1)\n", i, result);
    }

    // 7. 清理资源
    for (int i = 0; i < nDev; i++) {
        cuCtxSetCurrent(ctxs[i]);
        cuMemFree(sendbuff[i]);
        cuMemFree(recvbuff[i]);
        cuStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }

    // 8. 释放 Primary Context 的保留引用
    for (int i = 0; i < nDev; i++) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(devices[i]);
    }

    return 0;
}
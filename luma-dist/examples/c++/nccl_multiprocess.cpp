// 多进程 NCCL all-reduce：一个 rank 一个进程。
//
//   阶段0  rank0: ncclGetUniqueId() 生成 128B id（内含 rank0 的 bootstrap 地址）
//   阶段1  rank0 通过 TCP 把 id 发给每个其它进程（= luma-dist 里要实现的 Store）
//   阶段2  全体: ncclCommInitRank(nranks, id, rank) —— 阻塞会合，等齐 nranks 才返回
//   阶段3  ncclAllReduce 并校验
//
// 编译: make
// 运行: ./run_multiprocess.sh          (默认 WORLD_SIZE=2)
//   或手动:
//     WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 ./nccl_multiprocess &
//     WORLD_SIZE=2 RANK=1 LOCAL_RANK=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 ./nccl_multiprocess
//
// 为便于阅读，这里省略了全部错误检查。

#include <nccl.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  环境变量
// ---------------------------------------------------------------------------
static std::string env_str(const char* key, const char* def) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string(def);
}

static int env_int(const char* key, int def) {
    const char* v = std::getenv(key);
    return v ? std::atoi(v) : def;
}

// ---------------------------------------------------------------------------
//  TCP id 交换（luma-dist::Store 的参考实现）
// ---------------------------------------------------------------------------
static void write_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        p += w;
        n -= static_cast<size_t>(w);
    }
}

static void read_all(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t r = ::read(fd, p, n);
        p += r;
        n -= static_cast<size_t>(r);
    }
}

// rank0：监听，接受 world_size-1 个 peer，把 id 发给每个 peer
static void serve_id(int port, int world_size, const ncclUniqueId& id) {
    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ::listen(srv, world_size);

    for (int i = 1; i < world_size; ++i) {
        int fd = ::accept(srv, nullptr, nullptr);
        write_all(fd, &id, sizeof(id));
        ::close(fd);
    }
    ::close(srv);
}

// 其它 rank：连上 rank0 取 id（带重试，容忍 rank0 还没起来）
static ncclUniqueId fetch_id(const std::string& host, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    while (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::usleep(50 * 1000);
    }

    ncclUniqueId id;
    read_all(fd, &id, sizeof(id));
    ::close(fd);
    return id;
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main() {
    const int world_size = env_int("WORLD_SIZE", 2);
    const int rank = env_int("RANK", 0);
    const std::string master_addr = env_str("MASTER_ADDR", "127.0.0.1");
    const int master_port = env_int("MASTER_PORT", 29500);

    // 阶段0/1：rank0 生成 id 并分发；其余 rank 取 id
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
        serve_id(master_port, world_size, id);
    } else {
        id = fetch_id(master_addr, master_port);
    }

    // 一个进程一张卡
    cudaSetDevice(env_int("LOCAL_RANK", rank));

    // 阶段2：全体阻塞会合；会一直等到 world_size 个 rank 都调用了 initRank
    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, id, rank);

    // 阶段3：all-reduce
    const int n = 4;
    float* sendbuf = nullptr;
    float* recvbuf = nullptr;
    cudaMalloc(&sendbuf, n * sizeof(float));
    cudaMalloc(&recvbuf, n * sizeof(float));

    std::vector<float> host(n, static_cast<float>(rank + 1));  // rank r 填 r+1
    cudaMemcpy(sendbuf, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> out(n);
    cudaMemcpy(out.data(), recvbuf, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("rank %d/%d: result = [%.0f, %.0f, %.0f, %.0f]\n", rank, world_size, out[0], out[1], out[2], out[3]);

    cudaStreamDestroy(stream);
    cudaFree(sendbuf);
    cudaFree(recvbuf);
    ncclCommDestroy(comm);
    return 0;
}

# Nccl

## 常见用法

### 单进程

Comm 对象都处于同一个进程当中，直接利用 `ncclCommInitAll`：

```c++
const int nDev = 2;
ncclComm_t comms[nDev];
ncclCommInitAll(comms, nDev, nullptr);
```

得到一组 comms 对象，他们属于同一通信组，可以互相交换数据。使用方法又可以细分单线程/多线程：

- 单线程：在一个线程中调用所有 comms 对象的 `reduce`/`gather` 等 api，此时要在调用 api 前后包含 `ncclGroupStart()` 和 `ncclGroupEnd()`。

- 多线程：分配多个线程，每个线程使用独立的一个 comm，不需要 group api，直接调用 `reduce`/`gather` 等 api 即可，最简单的调用。

### 多进程

多个进程，每个进程一个 Comm，进行通信。这个主要的难点是理解 nccl 库如何知道多个 进程 的 Comm 属于同一组。

nccl 的设计是基于 id + 进程通信。大概流程是：

1. 需要有一个进程负责初始化 id：
    ```c++
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ```
    这个 id 对象标识一组跨进程的 comms 对象；

2. 所有参与 comm 通信的进程需要自己负责共享 id 对象。这个步骤 nccl 并不负责，可以通过本地 tcp 通信实现（初始化 id 的进程作为 server，其他作为 client 发送 id 数据）；

3. 到这里所有进程都有相同的 id 对象。接下来各自调用：
    ```c++
    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, id, rank);
    ```
    其中 world_size 应该都一致，id 是共享得到的 id，rank 来标记这个不同的进程。每个进程应该各不相同。

    这个函数的执行逻辑是：

    1. id 值包含构造 id（`ncclGetUniqueId`）这个进程的进程信息，和用于通信的信息。
    2. 构造 id 的进程在 `ncclCommInitRank` 执行中发现，自己就是构造 id 的，开启一个 bootstrap server，等待其他进程提交信息。
    3. 其他进程发现自己不是构造 id 的进程，通过 id 的信息得知如何和构造 id 进程的 bootstrap 建立链接，提交自己的 rank。
    4. 构造 id 的进程收集足够所有的 rank 进程后，发送某个消息到其他进程，通知他们可以退出 `ncclCommInitRank` 函数了。
    5. 之后大家解除阻塞，comm 对象属于同一组。
# DuetANN

DuetANN 是面向动态向量检索的存算分离系统。存储节点维护 Vamana 图、精确向量和更新状态，
计算节点通过 RDMA 访问索引，使用 OPQ/PQ 导航和原始向量 L2 精排。

项目包含两种查询引擎，共用 schema-16 索引：

| 查询推进模式 | 引擎 | 执行方式 |
|---|---|---|
| `coupled` | HostOrchestrated | CPU 维护 beam/visited、提交和轮询 RDMA，有限生命周期 CUDA kernel 计算距离 |
| `decoupled` | Persistent | GPU 常驻查询状态，通过 DOCA GPUNetIO 读取远端记录，并提前发起下一轮图读取 |
| `manual` | Persistent | 用于单独控制提前读取、批量宽度等实验参数 |

正式 baseline/full 都保留 Persistent GPU、GPUNetIO 和 Storage Owner。baseline 用 `manual`
关闭提前推进等研究机制，full 用 `decoupled`；配置和运行方法见
[SIFT100M 实验说明](experiment/sift100m/README.md)。HostOrchestrated 是显式选择的后端，运行失败不会触发引擎切换。

## 构建

需要 CMake 3.18+、支持 C++20 的编译器（GCC 11+）、AVX2 CPU、RDMA verbs、Boost 和 TBB。
计算节点还需要 CUDA 11.8+ 和 DOCA GPUNetIO，DOCA 默认安装路径为 `/opt/mellanox/doca`。
离线工具需要 CPU Faiss、BLAS、LAPACK 和 OpenMP，METIS 为可选分片依赖。

计算节点构建：

```bash
cmake -S . -B build-compute -DCMAKE_BUILD_TYPE=Release
cmake --build build-compute -j16
```

Persistent GPUNetIO 的最低架构为 SM70。V100 使用仍支持 Volta 的 CUDA 12.x，并显式
构建 SM70 产物；SM70 使用等价的 shuffle warp 归约，SM80+ 保留原生归约指令：

```bash
cmake -S . -B build-v100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j16
```

V100 兼容构建不降低 GPUNetIO 的硬件要求：DOCA 版本、ConnectX 网卡、GPUDirect RDMA
和 GPU doorbell 仍须受支持，并通过启动时的 RDMA read probe。显存容量和每 SM 常驻 CTA
数量按实际设备检查，不沿用 A800 的吞吐或 occupancy 假设。

无 GPU 的存储/索引构建节点：

```bash
cmake -S . -B build-storage -DCMAKE_BUILD_TYPE=Release \
  -DDUETANN_STORAGE_NODE_ONLY=ON
cmake --build build-storage -j16
```

仅运行存储服务时，可再设置 `-DDUETANN_BUILD_OFFLINE_TOOLS=OFF`，无需安装 Faiss、CUDA 或 DOCA。
若 Conda 的 TBB 引入了另一套 Boost 头文件，在配置命令中同时指定
`-DBOOST_ROOT="$CONDA_PREFIX" -DBoost_NO_SYSTEM_PATHS=ON`，保证 Boost 头文件与链接库来自同一安装。
计算和存储构建使用各自的目录；实验脚本通过 `BUILD_DIR`、`STORAGE_BUILD_DIR` 选择对应产物。

`DUETANN_METIS_PARTITION=AUTO` 会检查 METIS/GKlib 的实际链接兼容性，并尝试回退到系统库。
需要 METIS 时设置 `-DDUETANN_METIS_PARTITION=ON`；自定义安装路径使用
`-DDUETANN_METIS_ROOT=/path/to/metis`。没有兼容 METIS 时仍可使用 `balanced` 或 `bfs` 分片。

| 可执行文件 | 用途 |
|---|---|
| `duetann_compute_node` | 计算服务与更新客户端 |
| `duetann_memory_node` | 存储服务 |
| `duetann_breakdown_benchmark` | 查询/更新负载、吞吐、延迟、召回率与分解统计 |
| `vamana_offline_builder` | CPU 构图和物理分片 |
| `vamana_pq_indexer` | OPQ/PQ 训练、编码和 schema-16 产物生成 |
| `vamana_graph_extent_indexer` | 校验图分片并生成读取长度 sidecar |
| `vamana_metis_repartitioner` | 保留逻辑图，将现有索引重新做 METIS 物理分片 |
| `vamana_legacy_index_converter` | 将支持的 schema-15 旧索引转换为 schema 16 |

## 索引与部署

运行索引使用 schema 16、L2、`plain` 精确向量记录和 incarnation-tagged 紧凑图。
`M` 表示 PQ 子空间数：SIFT100M、Deep100M 使用 32，SPACEV100M 使用 20。

| 文件 | 计算节点 | 存储节点 X | 内容 |
|---|---|---|---|
| `<prefix>.meta.json` | 必需 | 必需 | 格式、分片、远端 offset 和构建指纹 |
| `<prefix>.pqM` | 必需 | 无需 | OPQ 矩阵和 PQ codebook |
| `<prefix>.gextent8` | 正式 profile 必需 | 无需 | 每个基础节点一个字节的图读取长度档 |
| `<prefix>_nodeX_ofN.dat` | 无需 | 必需 | 精确向量和图记录 |
| `<prefix>_nodeX_ofN.idmap` | 无需 | 必需 | 该 authority 的 ID、代际和物理目录 |
| `<prefix>_nodeX_ofN.centroid` | 无需 | 必需 | 物理分片中心状态与启动入口 |
| `<prefix>_nodeX_ofN.pqM.codes` | 无需 | 必需 | 注册到远端内存的基础 PQ 码流 |

`pqM` 是文件名示意，例如 `.pq32` 或 `.pq20`。metadata、分片和 sidecar 必须来自同一次构建，
启动时会校验格式、尺寸、指纹和 checksum。计算 GPU 从存储节点导入基础 PQ 码流；图、
精确向量和 ID 目录保留在存储侧。Persistent 按物理槽位和 incarnation 在 GPU 中缓存动态
PQ 数据；HostOrchestrated 每次读取动态 PQ 到主机 scratch，再复制到对应 CUDA lane。
权威状态均由存储节点持有。

首次运行需要准备匹配的数据和索引。当前四片 SIFT100M 实验的数据路径、节点地址、构建和
运行方法集中在 [SIFT100M 实验说明](experiment/sift100m/README.md)。运行器会校验索引、
输入和配置，在每个 case 前检查 GPU/MN 状态，并保存完整命令、日志和报告。
公共参数位于 `experiment/sift100m/common.conf`，`baseline.conf` 与 `full.conf` 只覆盖研究开关；
运行器会在输出目录保存合成后的完整配置。

## 原生系统 profile

运行时直接支持 `profile = baseline|full|custom` 和 `c1|c2|c3 = auto|on|off`，
在命令行中对应 `--profile`、`--c1`、`--c2`、`--c3`。全关等价于 baseline，
全开等价于 full；单项开关不会改变另外两项贡献。profile 只提供默认值，显式底层
参数最后生效；覆盖后的对应贡献标记为 `custom`，不会强制改回 profile 值。
默认 `custom` 且三个开关为 `auto`，因此旧配置和手动底层参数保持原行为。

- C1：两阶段更新、更新批处理、Stage2 合并、LeaseMerge 和反向更新合并。
- C2：Live/DynaExtent 自适应图传输及其冷启动提示。
- C3：前沿解耦、Stable-Run、提前发起和对应 issue width。

`config/profiles/` 提供 baseline、full 和 C1/C2/C3 独立开关的原生 INI 示例。
每个示例都使用 `[profile]`、`[index]`、`[cluster]`、`[transport]`、`[gpu]`、
`[C1:update]`、`[C2:transfer]`、`[C3:query]` 等标题分组，并列出全部 92 个运行时
字段。每个字段都有中文解释注释。普通分组标题只负责组织，不改变字段名；
`[compute:N]` 和 `[storage:N]` 则提供逐节点覆盖，由集群段的 `node = compute:N`
或 `node = storage:N` 选择，选中节点段的参数最后生效。各节点的 `host`/`port`
是拓扑的唯一来源，`servers`、`storage-peers` 和所需的 `clients` 会自动派生。
公共分组中的重复字段以及同一节点段中的重复字段都会被拒绝。需要完全手动控制时
也可使用 `profile = custom`。
基准输入路径和测量时长仍由 benchmark 参数单独控制。

## 源码导航

- `src/gpu_search/persistent_kernel/`：按 `query/`、`frontier/`、`merge/`、`io/`
  划分 Persistent 查询状态机、前沿证书、Beam 合并和 RDMA I/O；公共上下文与
  kernel 入口保留在根目录。
- `tools/`：按 `breakdown_benchmark/`、`long_insert_recall/`、`probes/` 和
  `vamana_offline/` 组织工具入口与实现。
- `test/`：按 `core/`、`offline/`、`benchmark/`、`runtime/`、`storage_owner/`、
  `gpu/` 组织测试，根目录的同名 CMake 模块负责对应注册。
- `src/memory_node/memory_node.cc`：存储节点生命周期和索引加载；`centroid_routing.cc`
  负责中心路由恢复、发布与成员更新。
- `src/memory_node/peer_rpc/`：Stage1 与 Stage2 RPC handler。
- `src/memory_node/storage_owner_index/`：局部搜索、同步/异步 continuation、图快照与发布。
- `src/memory_node/storage_owner_maintenance/`：维护队列、执行器唤醒/搜索租约调度与
  `Stage2Worker`；`scheduling.cc` 独立实现调度原语，`worker_search.cc`、
  `worker_reconcile.cc`、`worker_prepare.cc`、`worker_finalize.cc` 分别负责搜索、
  反向协调、图准备和收尾。
- `src/common/configuration.hh/.cc`：配置字段、解析与校验。
- `cmake/`：构建选项、工具链、依赖检测、源码清单及各类目标的独立模块；根
  `CMakeLists.txt` 只负责配置顺序和目标编排。
- `experiment/common/`：实验配置合成与报告公共逻辑；集群专用编排仍留在各实验目录。

## 验证与开发

```bash
ctest --test-dir build-storage --output-on-failure
ctest --test-dir build-compute --output-on-failure
```

本地测试覆盖格式、PQ、路由、更新协议、内存预算与查询策略。吞吐、召回率和 GPU/RDMA
运行情况需要在实际存储节点和计算节点上验证，测试通过不能代替硬件实验。

- [论文实现系统](doc/paper_system.md)：背景、动机、三项研究内容、一致性和证据边界。
- [SIFT100M 实验说明](experiment/sift100m/README.md)：baseline/full 配置、集群同步和性能验证。

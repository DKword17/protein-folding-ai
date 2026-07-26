/**
 * kernel/pairwise_potential.cu
 * =============================
 *
 * LJ 6-12 对势 CUDA 核 — 用于 Rosetta 能量函数中的非键相互作用计算。
 *
 * 策略：
 *   1. 共享内存 tile：每个 block 加载 TILE_SIZE 个残基坐标/类型到 shared memory，
 *      减少全局内存访问次数（O(N²) → O(N² / TILE_SIZE)）。
 *   2. 10 Å 截断：只计算 r < cutoff 的对，跳过远距离对。
 *   3. 原子累加：每个线程将局部能量累加到 thread-local 变量，
 *      最后用 atomicAdd 写回全局结果。
 *
 * 调用约定：
 *   blockDim.x = TILE_SIZE (128)
 *   gridDim.x  = ceil(N / TILE_SIZE)
 *   gridDim.y  = ceil(N / TILE_SIZE)
 *
 * Author: 王磊 (Wang Lei) — GPU 加速计算
 * Branch: dev/gpu-wanglei
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* ── 编译期常量 ──────────────────────────────────────────────────── */

#ifndef TILE_SIZE
#define TILE_SIZE 128         // 每个 block 处理的残基数（shared memory tile 宽度）
#endif

#ifndef CUTOFF_DIST
#define CUTOFF_DIST 10.0f     // 截断距离（Å）
#endif

/* ── LJ 能量计算 ─────────────────────────────────────────────────── */

/**
 * 在单个距离上计算 LJ 6-12 能量。
 * 
 * @param r        原子间距离（Å）
 * @param eps      LJ 势井深度（kcal/mol）
 * @param sig      LJ 零势能距离（Å）
 * @return LJ 能量值
 */
__device__ inline float _kernel_lj_pair(
    const float r,
    const float eps,
    const float sig
) {
    // 防止 r 太小导致数值溢出（r < 0.5 Å 视为重叠，返回高能量）
    if (r < 0.5f) return 1e6f;

    const float sr  = sig / r;       // sigma / r
    const float sr6 = sr * sr * sr
                    * sr * sr * sr;  // (sigma/r)^6
    const float sr12 = sr6 * sr6;    // (sigma/r)^12
    return eps * (sr12 - 2.0f * sr6);
}

/* ── 主核：LJ 对势 ───────────────────────────────────────────────── */

/**
 * 计算所有残基对之间的 LJ 6-12 能量。
 * 
 * 分块策略（tiling）：
 *   - 外层循环遍历所有 tile 对 (tx, ty)
 *   - 每个线程加载一个残基的坐标+类型到 shared memory
 *   - 同步后，每个线程计算当前 tile 中自己负责的一行 vs 一列
 *   - 累加局部能量
 *
 * @param coords    全局坐标数组 [N][3]（float, 行主序）
 * @param aa_types  残基类型索引 [N]（int, 指向 eps_table/sig_table）
 * @param eps_table LJ epsilon 查找表 [N][N]（对称矩阵，行主序展平）
 * @param sig_table LJ sigma 查找表 [N][N]
 * @param energy_out 输出标量（atomicAdd 累加）
 * @param n         残基总数
 * @param cutoff_sq 截断距离的平方（避免每次 sqrt）
 */
__global__ void _kernel_pairwise_lj(
    const float* __restrict__ coords,
    const int*    __restrict__ aa_types,
    const float*  __restrict__ eps_table,
    const float*  __restrict__ sig_table,
    float*                     energy_out,
    const int                  n,
    const float                cutoff_sq
) {
    // shared memory：每个维度 2 个 tile（当前行和当前列）
    __shared__ float s_coordsA[TILE_SIZE][3];
    __shared__ int    s_typesA[TILE_SIZE];
    __shared__ float s_coordsB[TILE_SIZE][3];
    __shared__ int    s_typesB[TILE_SIZE];

    // 全局 tile 索引
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int tid = threadIdx.x;

    // 全局残基索引
    const int i_global = tx * TILE_SIZE + tid;

    // thread-local 能量累加器（避免频繁 atomicAdd）
    float local_energy = 0.0f;

    // 外层 tile 循环：遍历所有列 tile
    for (int tile_j = 0; tile_j < gridDim.y; ++tile_j) {
        // 加载 tile_j 列的残基到 shared memory (s_coordsB, s_typesB)
        const int j_load = tile_j * TILE_SIZE + tid;
        if (j_load < n) {
            s_coordsB[tid][0] = coords[j_load * 3 + 0];
            s_coordsB[tid][1] = coords[j_load * 3 + 1];
            s_coordsB[tid][2] = coords[j_load * 3 + 2];
            s_typesB[tid] = aa_types[j_load];
        }
        __syncthreads();

        // 加载当前 tile 的 A 侧残基（当前行 tile）
        if (i_global < n) {
            s_coordsA[tid][0] = coords[i_global * 3 + 0];
            s_coordsA[tid][1] = coords[i_global * 3 + 1];
            s_coordsA[tid][2] = coords[i_global * 3 + 2];
            s_typesA[tid] = aa_types[i_global];
        }
        __syncthreads();

        // 内层循环：对当前 tile_j 中的所有残基，计算 i_global 与 j 的 LJ 能量
        if (i_global < n) {
            const float xi = s_coordsA[tid][0];
            const float yi = s_coordsA[tid][1];
            const float zi = s_coordsA[tid][2];
            const int   ti = s_typesA[tid];

            for (int k = 0; k < TILE_SIZE; ++k) {
                const int j = tile_j * TILE_SIZE + k;

                // 跳过 j >= n（padding）和 i == j（自相互作用）
                if (j >= n || j <= i_global) continue;

                // 读取 B tile 中的 j 残基坐标
                const float xj = s_coordsB[k][0];
                const float yj = s_coordsB[k][1];
                const float zj = s_coordsB[k][2];

                // 计算距离平方（避免 sqrt）
                const float dx = xi - xj;
                const float dy = yi - yj;
                const float dz = zi - zj;
                const float r2 = dx*dx + dy*dy + dz*dz;

                // 截断：跳过 > cutoff 的对
                if (r2 > cutoff_sq || r2 < 1e-6f) continue;

                const float r = sqrtf(r2);
                const int   tj = s_typesB[k];

                // 从查找表获取 eps 和 sig
                const float eps = eps_table[ti * n + tj];
                const float sig = sig_table[ti * n + tj];

                local_energy += _kernel_lj_pair(r, eps, sig);
            }
        }
        __syncthreads();
    }

    // 所有 tile 循环结束，用 atomicAdd 写回全局结果
    if (local_energy != 0.0f) {
        atomicAdd(energy_out, local_energy);
    }
}

/* ── 启动配置辅助函数 ────────────────────────────────────────────── */

/**
 * 计算 grid/block 维度。
 * 
 * @param n       残基数
 * @param grid_x  [out] gridDim.x
 * @param grid_y  [out] gridDim.y
 * @param block_x [out] blockDim.x（固定为 TILE_SIZE）
 */
extern "C" __host__ void _kernel_configure_grid(
    const int n,
    int* grid_x,
    int* grid_y,
    int* block_x
) {
    *grid_x  = (n + TILE_SIZE - 1) / TILE_SIZE;
    *grid_y  = (n + TILE_SIZE - 1) / TILE_SIZE;
    *block_x = TILE_SIZE;
}

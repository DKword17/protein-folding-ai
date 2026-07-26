"""
kernel/__init__.py
==================

Python 接口：LJ 6-12 对势的 GPU 加速计算（Numba CUDA wrapper）。

核心入口：compute_pairwise_lj()
  - 优先走 GPU（Numba CUDA JIT），自动 fallback 到 CPU
  - 输入可以是坐标数组或 Conformation 对象
  - 内部通过 eps_table / sig_table 查找参数

用法：
    >>> from kernel import compute_pairwise_lj
    >>> energy = compute_pairwise_lj(coords, aa_types, eps_table, sig_table)

回退机制：
    如果 CUDA 不可用（无 GPU / nvvm 缺失），自动降级到 CPU 实现。

Author: 王磊 (Wang Lei) — GPU 加速计算
Branch: dev/gpu-wanglei
"""

from __future__ import annotations

import math as _math
from typing import Optional as _Optional

import numpy as _np

# ── 常量和默认值 ──────────────────────────────────────────────────

CUTOFF_DEFAULT: float = 10.0  # 截断距离 (Å)
TILE_SIZE: int = 128           # CUDA block 大小
EPS_MIN: float = 1e-6          # 防止除以零

# ── GPU 可用性探测 ────────────────────────────────────────────────

_CAN_JIT: bool = False
try:
    import numba.cuda as _cuda  # type: ignore[import-untyped]
    if len(_cuda.list_devices()) > 0:
        def _nvvm_probe() -> bool:
            """尝试编译一个空 kernel 来验证 nvvm 可用。"""
            try:
                @_cuda.jit("void()")
                def _probe() -> None:
                    pass
                _probe[1, 1]()
                _cuda.synchronize()
                return True
            except Exception:
                return False
        _CAN_JIT = _nvvm_probe()
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════
#  CPU fallback 实现
# ═══════════════════════════════════════════════════════════════════

def _lj_pair_cpu(r: float, eps: float, sig: float) -> float:
    """单对残基的 LJ 6-12 能量（CPU 版）。"""
    if r < 0.5:
        return 1e6
    sr = sig / r
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return eps * (sr12 - 2.0 * sr6)


def compute_pairwise_lj_cpu(
    coords: _np.ndarray,
    aa_types: _np.ndarray,
    eps_table: _np.ndarray,
    sig_table: _np.ndarray,
    cutoff_sq: float,
) -> float:
    """CPU 版 LJ 对势（双重循环）。"""
    n = coords.shape[0]
    total = 0.0
    for i in range(n):
        xi, yi, zi = coords[i]
        ti = aa_types[i]
        for j in range(i + 1, n):
            xj, yj, zj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r2 = dx*dx + dy*dy + dz*dz
            if r2 > cutoff_sq or r2 < EPS_MIN:
                continue
            r = _math.sqrt(r2)
            tj = aa_types[j]
            eps = eps_table[ti, tj]
            sig = sig_table[ti, tj]
            total += _lj_pair_cpu(r, eps, sig)
    return total


# ═══════════════════════════════════════════════════════════════════
#  GPU 实现（Numba CUDA kernel）
# ═══════════════════════════════════════════════════════════════════

if _CAN_JIT:
    import numba.cuda as _cuda

    @_cuda.jit  # type: ignore[misc]
    def _kernel_pairwise_lj_numba(
        coords: _np.ndarray,       # [N, 3] float32
        aa_types: _np.ndarray,     # [N] int32
        eps_table: _np.ndarray,    # [N, N] float32
        sig_table: _np.ndarray,    # [N, N] float32
        energy_out: _np.ndarray,   # [1] float32
        n: int,
        cutoff_sq: float,
    ) -> None:
        """Numba CUDA 版 LJ 核 — 与 .cu 版本等价。

        坑：
          - Numba 的 shared memory 必须用 shared.array 声明
          - atomicAdd 通过 numba.cuda.atomic.add
        """
        s_coordsA = _cuda.shared.array((TILE_SIZE, 3), dtype=_np.float32)
        s_typesA  = _cuda.shared.array((TILE_SIZE,), dtype=_np.int32)
        s_coordsB = _cuda.shared.array((TILE_SIZE, 3), dtype=_np.float32)
        s_typesB  = _cuda.shared.array((TILE_SIZE,), dtype=_np.int32)

        tx = _cuda.blockIdx.x
        tid = _cuda.threadIdx.x
        i_global = tx * TILE_SIZE + tid
        local_energy = 0.0

        for tile_j in range(_cuda.gridDim.y):
            j_load = tile_j * TILE_SIZE + tid
            if j_load < n:
                s_coordsB[tid, 0] = coords[j_load, 0]
                s_coordsB[tid, 1] = coords[j_load, 1]
                s_coordsB[tid, 2] = coords[j_load, 2]
                s_typesB[tid] = aa_types[j_load]
            _cuda.syncthreads()

            if i_global < n:
                s_coordsA[tid, 0] = coords[i_global, 0]
                s_coordsA[tid, 1] = coords[i_global, 1]
                s_coordsA[tid, 2] = coords[i_global, 2]
                s_typesA[tid] = aa_types[i_global]
            _cuda.syncthreads()

            if i_global < n:
                xi = s_coordsA[tid, 0]
                yi = s_coordsA[tid, 1]
                zi = s_coordsA[tid, 2]
                ti = s_typesA[tid]

                for k in range(TILE_SIZE):
                    j = tile_j * TILE_SIZE + k
                    if j >= n or j <= i_global:
                        continue

                    xj = s_coordsB[k, 0]
                    yj = s_coordsB[k, 1]
                    zj = s_coordsB[k, 2]

                    dx = xi - xj
                    dy = yi - yj
                    dz = zi - zj
                    r2 = dx*dx + dy*dy + dz*dz

                    if r2 > cutoff_sq or r2 < EPS_MIN:
                        continue

                    r = _math.sqrt(r2)
                    tj = s_typesB[k]
                    eps = eps_table[ti, tj]
                    sig = sig_table[ti, tj]

                    if r < 0.5:
                        local_energy += 1e6
                    else:
                        sr = sig / r
                        sr6 = sr * sr * sr * sr * sr * sr
                        sr12 = sr6 * sr6
                        local_energy += eps * (sr12 - 2.0 * sr6)

            _cuda.syncthreads()

        if local_energy != 0.0:
            _cuda.atomic.add(energy_out, 0, local_energy)

    def _compute_on_gpu(
        coords: _np.ndarray,
        aa_types: _np.ndarray,
        eps_table: _np.ndarray,
        sig_table: _np.ndarray,
        cutoff_sq: float,
        n: int,
    ) -> float:
        """GPU 调度入口：分配 device 内存 → launch kernel → copy 结果。"""
        d_coords = _cuda.to_device(coords)
        d_types = _cuda.to_device(aa_types)
        d_eps = _cuda.to_device(eps_table)
        d_sig = _cuda.to_device(sig_table)
        d_out = _cuda.device_array((1,), dtype=_np.float32)
        d_out[0] = 0.0

        grid_x = (n + TILE_SIZE - 1) // TILE_SIZE
        grid_y = (n + TILE_SIZE - 1) // TILE_SIZE

        _kernel_pairwise_lj_numba[(grid_x, grid_y), TILE_SIZE](
            d_coords, d_types, d_eps, d_sig, d_out, n, cutoff_sq,
        )
        _cuda.synchronize()
        return float(d_out.copy_to_host()[0])

else:
    def _compute_on_gpu(*args, **kwargs) -> float:
        raise RuntimeError("CUDA unavailable; use CPU fallback")


# ═══════════════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════════════

def compute_pairwise_lj(
    coords_or_conf: object,
    types_or_efunc: object = None,
    eps_table: _Optional[_np.ndarray] = None,
    sig_table: _Optional[_np.ndarray] = None,
    cutoff: float = CUTOFF_DEFAULT,
) -> float:
    """LJ 6-12 对势统一入口 — CPU/GPU 自适应。

    支持两种调用方式：
      方式一（纯数组）：energy = compute_pairwise_lj(coords, types, eps, sig)
      方式二（对象）：   energy = compute_pairwise_lj(conf, efunc)

    Args:
        coords_or_conf: 坐标数组 [N,3] 或 Conformation 对象
        types_or_efunc: (数组模式) 类型索引; (对象模式) 能量函数
        eps_table: LJ epsilon 查找表 [N,N]
        sig_table: LJ sigma 查找表 [N,N]
        cutoff: 截断距离（默认 10 Å）

    Returns:
        总 LJ 能量 (kcal/mol)
    """
    coords: _np.ndarray
    types: _np.ndarray
    eps_tbl: _np.ndarray
    sig_tbl: _np.ndarray
    n: int
    cutoff_sq: float = cutoff * cutoff

    if hasattr(coords_or_conf, "residues"):
        # 对象模式：从 Conformation 提取数据
        conf = coords_or_conf
        efunc = types_or_efunc
        n = len(conf.residues)
        coords = _np.zeros((n, 3), dtype=_np.float32)
        types = _np.zeros(n, dtype=_np.int32)

        aa_list = sorted(set(r.aa_code for r in conf.residues))
        aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

        for idx, res in enumerate(conf.residues):
            coords[idx] = res.CB.astype(_np.float32)
            types[idx] = aa_to_idx[res.aa_code]

        n_types = len(aa_list)
        eps_tbl = _np.zeros((n_types, n_types), dtype=_np.float32)
        sig_tbl = _np.zeros((n_types, n_types), dtype=_np.float32)

        from folding_engine import VDW_RADII

        for aa_i in aa_list:
            for aa_j in aa_list:
                i, j = aa_to_idx[aa_i], aa_to_idx[aa_j]
                eps_tbl[i, j] = efunc.lj_epsilon.get((aa_i, aa_j), 0.15)
                sig_val = (VDW_RADII.get(aa_i, 1.8) + VDW_RADII.get(aa_j, 1.8)) / 2.0
                sig_tbl[i, j] = sig_val

        eps_full = _np.zeros((n, n), dtype=_np.float32)
        sig_full = _np.zeros((n, n), dtype=_np.float32)
        for i in range(n):
            for j in range(n):
                eps_full[i, j] = eps_tbl[types[i], types[j]]
                sig_full[i, j] = sig_tbl[types[i], types[j]]
        eps_tbl = eps_full
        sig_tbl = sig_full
    else:
        # 数组模式
        coords = _np.asarray(coords_or_conf, dtype=_np.float32)
        types = _np.asarray(types_or_efunc, dtype=_np.int32)
        if eps_table is None or sig_table is None:
            raise ValueError("数组模式下必须提供 eps_table 和 sig_table")
        eps_tbl = _np.asarray(eps_table, dtype=_np.float32)
        sig_tbl = _np.asarray(sig_table, dtype=_np.float32)
        n = coords.shape[0]

    if _CAN_JIT:
        try:
            return _compute_on_gpu(coords, types, eps_tbl, sig_tbl, cutoff_sq, n)
        except Exception as exc:
            import warnings
            warnings.warn(f"GPU 计算失败，降级到 CPU: {exc}")

    return compute_pairwise_lj_cpu(coords, types, eps_tbl, sig_tbl, cutoff_sq)


# ═══════════════════════════════════════════════════════════════════
#  性能基准
# ═══════════════════════════════════════════════════════════════════

def benchmark_cpu_vs_gpu(
    n_residues: int = 200,
    n_repeat: int = 5,
) -> dict[str, float]:
    """运行 CPU vs GPU 性能对比基准。"""
    import time

    rng = _np.random.default_rng(42)
    coords = rng.random((n_residues, 3), dtype=_np.float32) * 30.0
    types = rng.integers(0, 20, size=n_residues).astype(_np.int32)
    eps_tbl = rng.random((n_residues, n_residues), dtype=_np.float32) * 0.3 + 0.05
    sig_tbl = rng.random((n_residues, n_residues), dtype=_np.float32) * 0.5 + 1.5
    cutoff_sq = CUTOFF_DEFAULT * CUTOFF_DEFAULT

    # warmup
    _ = compute_pairwise_lj_cpu(coords, types, eps_tbl, sig_tbl, cutoff_sq)
    if _CAN_JIT:
        try:
            _compute_on_gpu(coords, types, eps_tbl, sig_tbl, cutoff_sq, n_residues)
        except Exception:
            pass

    cpu_times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = compute_pairwise_lj_cpu(coords, types, eps_tbl, sig_tbl, cutoff_sq)
        cpu_times.append(time.perf_counter() - t0)

    result: dict[str, float] = {
        "n_residues": n_residues,
        "cpu_mean_ms": float(_np.mean(cpu_times) * 1000),
        "cpu_std_ms": float(_np.std(cpu_times) * 1000),
    }

    gpu_times: list[float] = []
    if _CAN_JIT:
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _compute_on_gpu(coords, types, eps_tbl, sig_tbl, cutoff_sq, n_residues)
            gpu_times.append(time.perf_counter() - t0)
        result["gpu_mean_ms"] = float(_np.mean(gpu_times) * 1000)
        result["gpu_std_ms"] = float(_np.std(gpu_times) * 1000)
        result["speedup_x"] = round(result["cpu_mean_ms"] / max(result["gpu_mean_ms"], 1e-6), 2)
    else:
        result["gpu_mean_ms"] = float("nan")
        result["gpu_std_ms"] = float("nan")
        result["speedup_x"] = 1.0

    return result


__all__: list[str] = [
    "compute_pairwise_lj",
    "compute_pairwise_lj_cpu",
    "benchmark_cpu_vs_gpu",
    "CUTOFF_DEFAULT",
    "TILE_SIZE",
    "_CAN_JIT",
]

# Protein Folding AI — 跨国团队协作简报

## 项目概述

蛋白质从头折叠引擎，基于 Rosetta REF2015 能量函数的片段插入蒙特卡洛方法。
当前已有 `folding_engine.py` 作为基线代码，需要扩展到完整工程平台。

## 团队架构

```
                    ┌── William Thorpe ── 管道/验证/基准
                    │      🇬🇧 英国
                    │
Dmitry Volkov ──────┼── Jean-Luc Mercier ── 能量景观采样
    🇷🇺 俄罗斯      │         🇫🇷 法国
    MD 引擎          │
                    ├── Priya Sharma ── 测试/CI/回归
                    │      🇮🇳 印度
                    │
                    王磊 ── GPU 加速
                        🇨🇳 中国

                    Klaus Weber ── 力场参数校准
                        🇩🇪 德国
```

## 仓库

**URL: https://github.com/DKword17/protein-folding-ai**

分支策略：每个人在自己的分支开发，然后提 PR 合入 main。
代码用 Python，涉及性能的部分用 C++/CUDA。

有问题用 GitHub Issues 讨论。

---

## 🇷🇺 对 Dmitry Volkov

**国籍/语言**: 俄罗斯，母语俄语，英语为工作语言
**写作手法**: K&R 风格 C++，技术性注释用俄语西里尔字母，变量名保持英语。注释直接写在代码里，不冗余。功能函数先写做什么（俄语）、再写怎么做（代码）。风格像 Linux 内核开发者——干净、直接、不废话。

### 角色
骨干开发者 — 分子动力学引擎

### 分支
`dev/engine-dmitry`

### 交付任务

1. 创建 `kinetics/` 目录
2. `kinetics/verlet.cpp` — Velocity Verlet 积分器（SHAKE 约束）
3. `kinetics/langevin.cpp` — Langevin 热浴耦合（GROMACS 风格）
4. Python 绑定（pybind11），在 `kinetics/__init__.py` 暴露

### 用语的例
```cpp
/* кинетика: шаг интегрирования Verlet
 * обновляет положения и скорости с ограничениями SHAKE
 * возвращает полную кинетическую энергию в kcal/mol */
double verlet_step(Atom *atoms, int n, double dt);
```

### 验收
NVE 系综 1000 步 ΔE < 0.01%

---

## 🇫🇷 对 Jean-Luc Mercier

**国籍/语言**: 法国，母语法语，数学符号为第二语言
**写作手法**: Python 代码，变量名用法语（含重音符号），注释用法语写数学公式。docstring 写法语，但关键括号加注英文术语以利协作。风格像巴黎高师的数学物理研究者——精确、优雅、重视可读性。

### 角色
算法架构师 — 能量景观采样与副本交换

### 分支
`dev/sampling-jeanluc`

### 交付任务

1. 创建 `sampling/` 目录
2. `sampling/replica_exchange.py` — 副本交换 MC（温度 + 哈密顿量梯度）
3. `sampling/wham.py` — 加权直方图分析提取自由能
4. `sampling/umbrella_sampling.py` — 伞形采样

### 用语的例
```python
def distance_fidélité(ρ: np.ndarray, σ: np.ndarray) -> float:
    """
    Calcule la distance de fidélité (Fidelity distance) entre deux états.
    
    F(ρ, σ) = Tr(√(√ρ · σ · √ρ))
    
    Paramètres:
        ρ: premier opérateur densité
        σ: second opérateur densité
    Retourne:
        La fidélité quantique (float entre 0 et 1)
    """
```

### 验收
三温度 REMD 在 15 残基多肽上正确产生 Boltzmann 分布

---

## 🇬🇧 对 William Thorpe

**国籍/语言**: 英国（剑桥大学工程系），母语英语，正体英语写作
**写作手法**: 牛津逗号（Oxford comma）严格使用。while 表「当…时」、whilst 表「然而（轻微转折）」。docstring 写完整句子，含引用文献（Cambridge 格式，非 Harvard）。使用「whom」「should」「ought」等英式措辞。风格像 IBM 研究实验室的系统架构师——文档详尽，分章节，理性克制。

### 角色
系统架构师 — 基准测试与验证管道

### 分支
`dev/benchmark-william`

### 交付任务

1. 创建 `benchmark/` 目录
2. `benchmark/casp_scorer.py` — GDT_TS, RMSD, lDDT, TM-score
3. `benchmark/decoy_cluster.py` — SPICKER 聚类
4. 回旋半径 / SASA 计算工具

### 用语的例
```python
def compute_gdt_ts(predicted: np.ndarray, native: np.ndarray,
                  distance_cutoffs: list[float] = None) -> float:
    """
    Compute the GDT_TS (Global Distance Test — Total Score)
    for a predicted protein structure against the native conformation.

    Reference:
        Zemla, A. (2003). "LGA: a method for finding 3D similarities
        in protein structures." Nucleic Acids Research, 31(13), 3370–3374.

    Parameters:
        predicted: Predicted Cα coordinates [N, 3]
        native: Native Cα coordinates [N, 3]
        distance_cutoffs: Distance thresholds for correspondence
                          (default: [0.5, 1.0, 2.0, 4.0] Å)

    Returns:
        GDT_TS score (0–100), averaged across all distance cutoffs.

    Note:
        This implementation follows the LGA definition, not the
        original CASP implementation, which differs in its handling
        of multi-domain benchmarks.
    """
```

### 验收
对 ≤ 200 残基的 PDB 结构，GDT_TS 误差 < 0.001

---

## 🇮🇳 对 Priya Sharma

**国籍/语言**: 印度（班加罗尔），英语为工作语言
**写作手法**: 印度英语，注释极为详尽，每个测试函数都有完整的三段结构（Arrange-Act-Assert）。每个 assert 带错误消息。文件开头有一段很长的模块说明。使用「Please ensure」「kindly verify」等礼貌措辞。风格像 Infosys 或 Wipro 的高级 QA 工程师——不怕啰嗦，每个细节都写到。

### 角色
质量工程师 — 测试、回归、CI

### 分支
`dev/testing-priya`

### 交付任务

1. 创建 `tests/` 目录
2. `tests/test_energy_conservation.py` — NVE 1000 步能量守恒
3. `tests/test_integrators.py` — Velocity Verlet + Langevin 验证
4. `tests/test_benchmark.py` — 评分精度验证
5. `tests/test_sampling.py` — 副本交换收敛性验证
6. CI 徽章和 GitHub Actions 配置

### 用语的例
```python
def test_nve_energy_drift_under_threshold():
    """
    Test 1: Verify that the NVE integrator conserves total energy.

    Please ensure that the Verlet integrator module has been
    compiled and imported successfully before running this test.

    Arrange: Create a 3-atom water molecule initialised at 300 K
             with kinetic energy derived from Maxwell-Boltzmann
             distribution.
    Act:     Run 1000 steps of NVE integration with dt = 0.5 fs.
    Assert:  Total energy drift (|E_final - E_initial| / E_initial)
             must be less than 0.0001 (0.01%) over the full trajectory.
    """
```

### 验收
全部测试通过，覆盖率 > 85%

---

## 🇨🇳 对王磊

**国籍/语言**: 中国（合肥），母语中文，技术英语读写
**写作手法**: 中英混写。变量名、函数名、CUDA 核用英语。注释和技术说明用中文。注释风格务实——不写理论推导，只写「怎么接」「为什么这么写」「坑在哪」。风格像 FAANG 的中国籍底层工程师——落地、务实、「先跑起来再说」。

### 角色
性能工程师 — GPU 加速计算

### 分支
`dev/gpu-wanglei`

### 交付任务

1. 创建 `kernel/` 目录
2. `kernel/pairwise_potential.cu` — 6-12 LJ 对势 CUDA 核
   - 截断 10 Å
   - shared memory 优化
3. Numba/Cython Python 接口暴露
4. 加速比 > 30x

### 用语的例
```cuda
extern "C" __global__
void _kernel_lj_pairwise(
    const float4* coords,   // 原子坐标 (x,y,z mass)
    const float2* params,   // 参数对 (epsilon, sigma)
    float* energy_out,      // 每个原子的能量
    int n_atoms
) {
    // 每个线程处理一个原子对
    // shared memory 里缓存当前 tile 的坐标
    // 截断距离 10.0 Å (再远 LJ 趋近于 0)
    
    extern __shared__ float4 tile_coords[];
    // ... tile 循环处理所有对
}
```

### 验收
200 残基蛋白质单步能量计算 < 1 ms (GPU)

---

## 🇩🇪 对 Klaus Weber

**国籍/语言**: 德国（慕尼黑工业大学），母语德语，技术英语精准
**写作手法**: 德语变量名/函数名（Gewicht, Skalierung, Optimierung, Abweichung）。类型注解全覆盖（Python Final, TypeVar）。常量大写。每个函数给出输出值的置信区间。docstring 用德语，标注「年-月-日」「版本号」。风格像 Bosch 或 Siemens 的硬核测试工程师——严格、可复现、不留模糊空间。

### 角色
参数工程师 — 力场校准与能量项优化

### 分支
`dev/param-klaus`

### 交付任务

1. 创建 `params/` 目录
2. `params/weight_optimizer.py` — 从 PDB 数据库优化 5 个能量项权重
3. `params/force_field_calibrator.py` — 从 QM 数据校准 LJ epsilon
4. 差分进化 + 贝叶斯优化（skopt / scipy）

### 用语的例
```python
GEWICHTE_VORGABE: Final[dict[str, float]] = {
    'lennard_jones': 0.50,
    'wasserstoffbrücke': 0.75,   # Hydrogen bond
    'solvatisierung': 0.60,
    'ramachandran': 0.35,
    'repulsiv': 1.10,
}

def optimiere_gewichte(pdbs: list[PDBStruktur]) -> OptimierungsErgebnis:
    """
    Optimiert die Energie-Gewichte mittels differentieller Evolution.

    Parameter:
        pdbs: Liste von PDB-Strukturen für das Training

    Rückgabe:
        OptimierungsErgebnis mit den optimalen Gewichten und
        95%-Konfidenzintervall

    Ausgabe:
        Jeder Kandidat wird auf RMSD gegen die Röntgenstruktur
        evaluiert. Das beste Set wird zurückgegeben.
    """
```

### 验收
校准后 5 个测试蛋白平均 RMSD < 4 Å（从 extended chain 折叠）

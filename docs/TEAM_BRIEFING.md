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

### 角色
骨干开发者 — 分子动力学引擎

### 分支
`dev/engine-dmitry`

### 交付任务

1. 创建 `kinetics/` 目录
2. `kinetics/verlet.cpp` — Velocity Verlet 积分器（SHAKE 约束）
3. `kinetics/langevin.cpp` — Langevin 热浴耦合（GROMACS 风格）
4. Python 绑定（pybind11），在 `kinetics/__init__.py` 暴露

### 风格
- K&R C++，技术注释俄语，变量名英语
- GROMACS 热浴接口命名惯例
- 所有函数 noexcept

### 验收
NVE 系综 1000 步 ΔE < 0.01%

---

## 🇫🇷 对 Jean-Luc Mercier

### 角色
算法架构师 — 能量景观采样与副本交换

### 分支
`dev/sampling-jeanluc`

### 交付任务

1. 创建 `sampling/` 目录
2. `sampling/replica_exchange.py` — 副本交换 MC（温度 + 哈密顿量梯度）
3. `sampling/wham.py` — 加权直方图分析提取自由能
4. `sampling/umbrella_sampling.py` — 伞形采样

### 风格
- Python，变量名法语（Amplitude, Échantillon, Température, Probabilité）
- 注释中用法语数学公式
- 参考 Frenkel & Smit

### 验收
三温度 REMD 在 15 残基多肽上正确产生 Boltzmann 分布

---

## 🇬🇧 对 William Thorpe

### 角色
系统架构师 — 基准测试与验证管道

### 分支
`dev/benchmark-william`

### 交付任务

1. 创建 `benchmark/` 目录
2. `benchmark/casp_scorer.py` — GDT_TS, RMSD, lDDT, TM-score
3. `benchmark/decoy_cluster.py` — SPICKER 聚类
4. 回旋半径 / SASA 计算工具

### 风格
- 英国学术英语，Oxford comma
- while / whilst 严格区分
- 完整 docstring + 参考文献
- Cambridge 引用格式

### 验收
对 ≤ 200 残基的 PDB 结构，GDT_TS 误差 < 0.001

---

## 🇮🇳 对 Priya Sharma

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

### 风格
- Arrange-Act-Assert 模式
- 注释极度详尽，断言都有消息
- Python unittest 框架

### 验收
全部测试通过，覆盖率 > 85%

---

## 🇨🇳 对王磊

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

### 风格
- 变量名英文，注释中文
- CUDA 核函数前缀 `_kernel_`
- 注释写接口对接方法

### 验收
200 残基蛋白质单步能量计算 < 1 ms (GPU)

---

## 🇩🇪 对 Klaus Weber

### 角色
参数工程师 — 力场校准与能量项优化

### 分支
`dev/param-klaus`

### 交付任务

1. 创建 `params/` 目录
2. `params/weight_optimizer.py` — 从 PDB 数据库优化 5 个能量项权重
3. `params/force_field_calibrator.py` — 从 QM 数据校准 LJ epsilon
4. 差分进化 + 贝叶斯优化（skopt / scipy）

### 风格
- 德语变量名（Gewicht, Skalierung, Parameter, Optimierung）
- Python 类型注解全覆盖
- 常量 Final 大写
- 置信区间输出

### 验收
校准后 5 个测试蛋白平均 RMSD < 4 Å（从 extended chain 折叠）

# quantum-sim-go

二维含时薛定谔方程的数值模拟器（Go 实现）：

```
i ℏ ∂ψ/∂t = [ -ℏ²/(2m) ∇² + V(x,y) ] ψ
```

输入初始复值波函数、外部势能与演化参数，在有限周期计算域上输出各采样时刻的完整复振幅
（实部/虚部），并提供守恒量诊断、基态求解、确定性并行与检查点恢复能力。

## 构建与测试

```sh
go build ./...
go test ./...
```

## 数值方法

- **时间积分**：Strang 分裂算子（半步势能、整步动能、半步势能）。传播子严格幺正，
  因此范数守恒到舍入误差；能量误差有界振荡、**无长期漂移**，适合长时间演化。
- **空间离散**：傅里叶谱方法（周期边界），动能项在波数空间精确对角化。
  网格维度必须是 2 的幂（基-2 FFT）。
- **基态求解**：虚时演化（t → -iτ）+ 每步归一化，能量单调收敛到基态能量。

### 精度语义

| 量 | 误差行为 |
|---|---|
| 波函数（实时间） | 时间方向 O(dt²)，空间方向谱精度（随网格加密指数下降） |
| 范数 | 幺正性保证，漂移 ~1e-15 量级，不随步数增长 |
| 能量期望（实时间） | 有界振荡 O(dt²)，无长期漂移 |
| 基态能量（虚时间） | O(dτ⁴)：态误差 O(dτ²)，但能量是变分 Rayleigh 商，一阶项抵消 |

`groundstate.EstimateOrder(hs, errs)` 可对任意步长序列做对数-对数最小二乘拟合，
给出收敛阶估计；`Result.Residuals` 保存每次迭代的 |ΔE| 残差曲线。

## 守恒量诊断

`solver.Run` 在每个采样点检查两个守恒量，越限时返回**可区分**的状态码，
CLI 同时以非零退出码终止（结果仍写出，但 `summary.json` 中明确标记不可信）：

| 状态 | 含义 | 默认阈值 |
|---|---|---|
| `ok` | 所有守恒量在容差内 | — |
| `norm-drift` | \|norm−1\| 超过 `norm_tol` | 1e-6 |
| `energy-drift` | 相对能量漂移超过 `energy_tol` | 1e-3 |

`solver.Solver.MaxNormDrift` / `MaxEnergyDrift` 记录运行期间观测到的最大漂移。

## 并行与可复现性

- `workers` 控制 FFT 行/列变换的并行度。每行/列由单个 goroutine 完整计算，
  归约顺序固定，因此**任意线程数产生逐比特一致的输出**（帧文件可逐条 diff）。
- 浮点求和顺序全部固定，相同输入必然产生相同输出。

## 检查点与中断恢复

- 每 `checkpoint.every` 步提交一次检查点：先写临时文件、fsync、再原子 rename，
  崩溃不会留下部分写入的检查点（未提交的 `.tmp` 残留会被忽略并清理）。
- `quantumsim evolve -resume` 从最近一次**已提交**的检查点继续，
  不重复、不跳过任何时间片；恢复运行的最终状态与不中断运行逐比特一致
  （见 `TestResumeMatchesUninterrupted`）。

## 使用

```sh
# 含时演化（配置见 examples/harmonic.json）
quantumsim evolve -config examples/harmonic.json

# 中断后恢复
quantumsim evolve -config examples/harmonic.json -resume

# 基态与本征能量（虚时演化）
quantumsim groundstate -config examples/harmonic.json -dtau 0.01 -tol 1e-12
```

### 配置项

| 字段 | 含义 | 默认 |
|---|---|---|
| `grid.Nx, Ny` | 网格点数（须为 2 的幂） | 必填 |
| `grid.Lx, Ly` | 计算域边长（原点居中，周期边界） | 必填 |
| `evolution.dt` | 时间步长 | 必填 |
| `evolution.steps` | 总步数 | 必填 |
| `evolution.sample_every` | 帧采样间隔（步） | 1 |
| `evolution.workers` | 并行线程数（不影响结果） | 1 |
| `evolution.norm_tol` | 范数漂移阈值 | 1e-6 |
| `evolution.energy_tol` | 相对能量漂移阈值 | 1e-3 |
| `potential.kind` | `zero` / `harmonic` / `slit` | `zero` |
| `potential.omega_x, omega_y` | 谐振子频率 | 0 |
| `potential.barrier_*`, `slit_*` | 双缝势垒参数 | 0 |
| `initial.kind` | `gaussian`（中心/宽度/波矢可配） | `gaussian` |
| `checkpoint.dir`, `every` | 检查点目录与提交间隔 | 每步 |
| `output.dir` | 帧与摘要输出目录 | 必填 |

### 输出

- `frame-XXXXXXXX.csv`：采样帧，`x,y,re,im` 四列，**保留完整复振幅**
  （不仅是概率密度），可逐条比较。
- `summary.json`：状态码、最终步数/时刻、最大范数漂移与能量漂移。
- `residuals.csv`（groundstate）：每次迭代的能量残差曲线。
- 所有输出文件均通过临时文件 + 原子 rename 写入。

## 测试覆盖

| 场景 | 测试 |
|---|---|
| 自由波包扩散（解析解对比、网格细化） | `TestFreeWavepacketSpreading`, `TestFreeWavepacketGridRefinement` |
| 步长收敛阶（O(dt²)） | `TestHarmonicOscillatorDtConvergence` |
| 长时间稳定性（无漂移发散） | `TestLongTimeStability` |
| 叠加态干涉与相位振荡 | `TestSuperpositionPhaseAndInterference` |
| 并行逐比特一致 | `TestParallelConsistency`, `TestFFT2DParallelDeterministic` |
| 束缚态基态能量与收敛阶 | `internal/groundstate` 全部测试 |
| 中断恢复（逐比特一致、无部分写入） | `internal/checkpoint` 全部测试 |
| 守恒量诊断状态可区分 | `TestDriftDiagnostics` |
| CLI 端到端（演化/恢复/基态） | `cmd/quantumsim` 测试 |

# quantum-sim-go

二维含时薛定谔方程的数值模拟：

$$i\hbar\frac{\partial\psi}{\partial t} = \left(-\frac{\hbar^2}{2m}\nabla^2 + V(x,y)\right)\psi$$

系统在有限周期计算域上演化复值波函数，输出各时刻的完整复振幅（非仅概率密度），
支持基态求解、守恒量诊断、确定性并行与断点恢复。

## 数值方法

- **时间积分**：Strang 分裂算子（split-operator）法
  $\psi(t+\Delta t) = e^{-iV\Delta t/2\hbar}\,\mathcal{F}^{-1} e^{-i\hbar k^2\Delta t/2m}\,\mathcal{F}\, e^{-iV\Delta t/2\hbar}\psi(t)$。
  时间二阶精度（误差 $\sim O(\Delta t^2)$），每个子步都是幺正变换。
- **空间离散**：周期边界 + 谱方法（FFT），光滑波函数下谱精度收敛。
  网格点数 `nx`、`ny` 必须是 2 的幂（≥4）。
- **基态求解**：虚时演化（$\tau = it$）+ 每步归一化，激发态指数衰减，
  能量期望值收敛到基态能量。

## 守恒量语义

每次诊断（快照/检查点/末步）记录 `Diagnostics`：

| 字段 | 含义 |
|---|---|
| `Norm` | $\int|\psi|^2\,dA$，幺正演化下守恒到舍入误差（~1e-13） |
| `NormDrift` | 相对第 0 步范数的绝对漂移 |
| `Energy` | $\langle H\rangle = \langle T\rangle + \langle V\rangle$（动能经谱微分计算） |
| `EnergyDrift` | 相对初始能量的相对漂移 |
| `Status` | `ok` / `norm_drift` / `energy_drift` |

**能量漂移的语义**：分裂算子是辛积分器，能量误差是**有界振荡**而非 secular 增长，
振幅 $\sim O(\Delta t^2)$，不随演化步数累积发散。判定阈值用 `energyDriftTol`（相对）与
`normDriftTol`（绝对，默认见下）。**漂移超阈值时 `Run` 立即停止**，返回
`*sim.DriftError`（可用 `errors.As` 提取，含越界时的完整 `Diagnostics`），
`RunResult.Status` 与错误中状态一一对应，不会静默继续或返回不可信结果。

## 精度控制

- **步长 `dt`**：波函数误差 $O(\Delta t^2)$；基态能量因变分原理超收敛至 $O(\Delta t^4)$。
  `EstimateOrder(e1, e2, e3)` 用 dt、dt/2、dt/4 三个能量估计收敛阶。
- **网格 `nx/ny`**：谱精度。注意波函数必须在周期接缝处可忽略（`sigma << L`），
  否则接缝不连续会把收敛降为代数阶。
- **基态残差曲线**：`GroundStateResult.Residuals` 记录每步 $|\Delta E|$，
  虚时中几何衰减；`Converged` 表示末残差低于 `tol`。
- 自动化测试验证了：自由波包解析扩散率、谐振子基态能量（$\hbar\omega$）、
  误差随 dt/网格细化的单调下降、长时演化（2 万步）守恒量有界。

## 并行与确定性

`workers` 控制 FFT 与逐点算子的并行度。所有归约（范数、能量等）按**固定串行顺序**，
逐点算子按行块划分且互不依赖，因此**在固定输入下，改变线程数不改变任何输出**：
快照文件逐字节一致，诊断历史逐项一致（`TestParallelConsistency` 验证）。

## 断点恢复

- 检查点与快照均**原子提交**：先写临时文件、fsync、再 rename；崩溃只会留下
  旧版本或新版本，绝不会留下半写状态。残留的 `.tmp` 文件在下次加载时被忽略并清理。
- 恢复语义：`Run` 启动时若发现已提交检查点且配置哈希匹配，则从
  `checkpoint.Step + 1` 继续——不重复、不跳过任何时间片。由于演化完全确定，
  崩溃前已写出的快照若被重写，内容逐字节相同。
- 配置哈希只覆盖影响态演化的字段（网格、`mass`、`hbar`、`dt`）；
  `steps`、`workers`、输出节奏等可在恢复时改变。物理参数不一致会被拒绝并报错。
- 损坏的已提交检查点会被检测并报错，不会被静默忽略。

## 配置项（JSON）

```json
{
  "nx": 64, "ny": 64, "lx": 20, "ly": 20,
  "mass": 1, "hbar": 1,
  "dt": 0.01, "steps": 1000,
  "workers": 4,
  "outputDir": "out/run",
  "snapshotEvery": 50,
  "checkpointEvery": 100,
  "normDriftTol": 1e-9,
  "energyDriftTol": 1e-4,
  "potential": {"type": "harmonic", "omega": 1, "centerX": 0, "centerY": 0},
  "initial":  {"type": "gaussian", "x0": 2, "sigma": 1, "kx": 0.5},
  "ground":   {"tol": 1e-10, "maxSteps": 100000}
}
```

| 键 | 默认 | 说明 |
|---|---|---|
| `nx`, `ny` | — | 网格点数，2 的幂且 ≥4 |
| `lx`, `ly` | — | 计算域边长（周期边界，域为 [-L/2, L/2)） |
| `mass`, `hbar` | — | 粒子质量与约化普朗克常数，>0 |
| `dt` | — | 时间步长，>0 |
| `steps` | 0 | 演化步数 |
| `workers` | CPU 数 | 并行线程数，不影响结果 |
| `outputDir` | "" | 输出目录；启用快照/检查点时必填 |
| `snapshotEvery` | 0=关闭 | 快照间隔（步） |
| `checkpointEvery` | 0=关闭 | 检查点间隔（步） |
| `normDriftTol` | 1e-9 | 范数漂移阈值（绝对） |
| `energyDriftTol` | 1e-4 | 能量漂移阈值（相对） |
| `potential.type` | zero | `zero` / `harmonic`(omega, centerX/Y) / `barrier`(height, x0, x1) |
| `initial.type` | gaussian | `gaussian`(x0,y0,sigma,kx,ky) / `plane`(kx,ky) / `superposition`(kx,ky,k2x,k2y) |
| `ground.tol` | 1e-10 | 基态能量残差收敛阈值 |
| `ground.maxSteps` | 100000 | 虚时演化最大步数 |

## CLI

```sh
go build ./cmd/quantumsim
./quantumsim run    -config examples/run_harmonic.json    # 实时演化（可恢复）
./quantumsim ground -config examples/ground_harmonic.json # 基态求解
```

退出码：`0` 成功；`3` 守恒量漂移（stderr 含诊断）；`4` 基态未收敛；`1`/`2` 其他错误。

## 库用法

```go
cfg := sim.Config{Nx: 64, Ny: 64, Lx: 20, Ly: 20, Mass: 1, Hbar: 1,
    Dt: 0.01, Steps: 1000, Workers: 4, OutputDir: "out",
    SnapshotEvery: 50, CheckpointEvery: 100}
g := sim.NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
V := sim.HarmonicPotential(g, cfg.Mass, 1.0, 0, 0)
psi0 := sim.GaussianPacket(g, 2, 0, 1, 0.5, 0)
res, err := sim.Run(cfg, V, psi0)          // 实时演化
gs, err := sim.GroundState(cfg, V, psi0, 1e-10, 100000) // 基态
```

## 测试

```sh
go test ./...
```

覆盖：FFT 正确性与并行位级一致、自由波包扩散（解析宽度）、误差随步长/网格
细化下降（二阶/谱精度）、长时演化守恒有界、谐振子基态能量与收敛阶、
定态性、双平面波干涉（完整复振幅逐点比对）、叠加态振荡（$\langle x\rangle(t)$）、
并行一致性（workers=1/2/4/8 输出逐字节一致）、检查点恢复（不重复/不跳过时间片、
半成品清理、配置不匹配拒绝、损坏检测）、漂移诊断状态可区分。

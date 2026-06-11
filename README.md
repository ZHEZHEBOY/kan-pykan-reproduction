# KAN 原论文函数拟合实验复现

本目录给出课程报告第二部分的代码复现。复现对象为 **KAN: Kolmogorov-Arnold Networks** 中官方 `pykan` 入门示例使用的二维函数拟合实验，而不是复现整篇论文的所有实验。

与早期轻量版本不同，当前版本调用官方 `pykan` 实现，包含 KAN 的关键机制：三阶 B 样条边函数、LBFGS 优化、稀疏正则、剪枝和符号化恢复。MLP 只作为课程报告中的对比基线保留。

## 1. 复现论文

Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Y. Hou, Max Tegmark. **KAN: Kolmogorov-Arnold Networks**. International Conference on Learning Representations (ICLR), 2025; arXiv:2404.19756, 2024.

官方代码库：`KindXiaoming/pykan`。

论文链接与版权说明：

- arXiv 页面：https://arxiv.org/abs/2404.19756
- ICLR/OpenReview 页面：https://openreview.net/forum?id=Ozo7qJ5vZi
- OpenReview 页面标注该论文为 `CC BY 4.0`。本仓库保留论文 PDF 仅作为课程复现配套材料；如果公开仓库对版权合规有更高要求，可删除 PDF，仅保留上述官方链接。

## 2. 复现实验范围

代码复现官方 `pykan` 文档中的函数拟合示例：

```text
f(x, y) = exp(sin(pi*x) + y^2),  x,y in [-1,1]
```

默认设置：

- 数据集：固定随机种子 `2026`，生成 `1000` 个训练样本和 `2000` 个测试样本。
- KAN：`width=[2,5,1]`，`grid=5`，`k=3`，即官方 pykan 的三阶 B 样条边函数设置。
- KAN 训练流程：先带稀疏正则训练，再剪枝，再继续无正则 LBFGS 训练，最后执行自动符号化和符号化后训练。
- 对比基线：`MLP-16` 与 `MLP-24`，优化器为 AdamW。
- 评价指标：参数量、训练 RMSE、测试 RMSE、运行时间。
- 运行设备：默认 `--device auto`，优先使用 CUDA GPU；没有 CUDA 时自动回退到 CPU。

说明：这里的“完整复现”指完整复现本课程报告选定的原论文函数拟合实验流程；KAN 原文还包含 PDE、科学发现、scaling law 等大量实验，未在本目录中复现。

## 3. 环境安装

推荐 Python 3.10+。

```bash
pip install -r requirements.txt
```

依赖项：

- `torch`
- `pykan==0.2.8`
- `sympy`

## 4. 运行方式

在本目录运行默认完整实验：

```bash
python kan_reproduction.py --clean-ckpt
```

如需快速检查代码流程，可降低样本数和训练步数：

```bash
python kan_reproduction.py --output-dir smoke_test --n-train 20 --n-test 40 --kan-steps 1 --prune-steps 0 --symbolic-steps 0 --mlp-epochs 2 --clean-ckpt
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--seed` | `2026` | 随机种子 |
| `--n-train` | `1000` | 训练样本数 |
| `--n-test` | `2000` | 测试样本数 |
| `--width` | `2 5 1` | KAN 网络宽度 |
| `--grid` | `5` | KAN 网格区间数 |
| `--spline-order` | `3` | B 样条阶数 |
| `--kan-steps` | `20` | 带稀疏正则的初始 LBFGS 步数 |
| `--prune-steps` | `50` | 剪枝后的 LBFGS 步数 |
| `--symbolic-steps` | `50` | 自动符号化后的 LBFGS 步数 |
| `--lamb` | `0.01` | 初始稀疏正则强度 |
| `--lamb-entropy` | `10.0` | 初始熵正则强度 |
| `--mlp-epochs` | `2500` | MLP 的 AdamW 训练轮数 |
| `--device` | `auto` | 自动选择 `cuda` 或 `cpu`，也可手动指定 |
| `--torch-threads` | 未设置 | 可选 CPU 线程限制；默认使用 PyTorch 自身设置 |
| `--output-dir` | 当前目录 | 输出数据和结果的目录 |

## 5. 输出文件

运行后会生成或更新以下文件：

| 文件 | 作用 |
| --- | --- |
| `kan_paper_toy_dataset.csv` | 按原论文函数生成的训练集和测试集 |
| `results.csv` | 各模型参数量、训练 RMSE、测试 RMSE、运行时间 |
| `results.json` | 详细结果，包含 KAN 符号化公式和训练配置 |
| `run_config.json` | 本次运行的随机种子、样本数、训练步数等配置 |
| `prediction_samples.csv` | 测试集前若干样本的真实值、预测值和绝对误差 |

## 6. 当前复现实验结果

当前完整运行结果如下。本次运行由 `--device auto` 自动选择 CUDA，实际设备为 `NVIDIA GeForce RTX 4060 Laptop GPU`，PyTorch 版本为 `2.7.1+cu118`。

| 模型 | 优化器 | 参数量 | 训练 RMSE | 测试 RMSE | 时间 |
| --- | --- | ---: | ---: | ---: | ---: |
| Official-pykan-KAN-width-[2,5,1]-grid-5-k-3 | LBFGS | 42 | 0.000946 | 0.000895 | 177.74s |
| MLP-16-hidden | AdamW | 337 | 0.015026 | 0.015270 | 15.21s |
| MLP-24-hidden | AdamW | 697 | 0.009212 | 0.009044 | 14.94s |

KAN 自动符号化后恢复出的公式近似为：

```text
exp(y^2 + sin(pi*x))
```

这与原论文函数拟合示例的目标函数一致，说明复现流程成功恢复了该函数结构。

## 7. 代码结构

| 位置 | 说明 |
| --- | --- |
| `target_function` | 原论文二维函数数据生成式 |
| `make_dataset` / `save_dataset_csv` | 调用 `pykan.create_dataset` 并保存 CSV |
| `fit_official_kan` | 官方 pykan KAN 训练、剪枝和符号化流程 |
| `MLP` / `train_mlp` | 对比基线模型和训练 |
| `write_outputs` | 输出 CSV、JSON、配置和预测样例 |

## 8. GitHub 上传建议

建议上传以下文件：

```text
复现代码/
├── README.md
├── requirements.txt
├── kan_reproduction.py
├── KAN_Kolmogorov-Arnold_Networks_2024.pdf
├── kan_paper_toy_dataset.csv
├── results.csv
├── results.json
├── run_config.json
├── prediction_samples.csv
└── .gitignore
```

其中 `README.md`、`requirements.txt`、`kan_reproduction.py` 是最核心文件；数据集、结果文件和配置文件用于保留课程报告对应的实验记录，建议一并上传。

原论文 PDF 已放在本目录中，可与代码一并上传作为配套阅读材料。如果仓库公开发布，也可以只在 README 中保留 arXiv/ICLR 链接，避免不必要的版权问题。

不建议上传以下文件：

- `__pycache__/`
- `*.pyc`
- `pykan_checkpoints/`
- `.idea/`
- 临时测试目录，如 `smoke_test/`
- Word 临时锁文件，如 `~$*.docx`
- 中间渲染图、临时脚本和个人机器路径相关文件

代码规范方面，本脚本尽量满足以下要求：

- 使用相对路径和 `--output-dir` 参数，避免本机绝对路径。
- 固定随机种子，并保存 `run_config.json`，便于复现实验。
- 函数职责清晰：数据、官方 KAN、MLP、训练、输出分别封装。
- 明确说明复现范围：完整复现选定函数拟合实验，而不是复现整篇论文全部实验。

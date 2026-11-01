# 仓库贡献指南

## 项目结构与模块组织

本仓库使用 Jupyter Notebook 开展神经网络学习实验。`VAE.ipynb` 包含变分自编码器及其卷积变体；`Reinforcement Learning.ipynb` 包含 Actor-Critic 和 PPO。`README.md` 提供主题索引，`environment.yml` 记录参考 Conda 环境。目前没有独立的源码包或自动化测试目录。

数据集放在 `data/`，图片放在 `img/`，视频放在 `videos/`。这些目录以及 `draft*`、`testing*` 和 `environment-local.yml` 均被 Git 忽略。编写实验时，不要假设其他贡献者已拥有这些本地文件。

## 构建、测试与开发命令

在仓库根目录执行以下命令：

- `conda env create -f environment.yml`：创建参考环境 `D2L`。
- `conda activate D2L`：运行 Notebook 前激活环境。
- `jupyter notebook`：启动 Notebook，进行交互式开发。
- `git diff --stat`：提交前检查修改范围。

本项目无需构建。环境配置包含 Windows 专用包及固定版本的 CUDA 版 PyTorch，在其他平台安装时可能需要调整。将本地调整记录在已被忽略的 `environment-local.yml` 中；新增共享依赖时，应同步更新相关说明。

## 代码风格与命名约定

Python 单元格使用四个空格缩进。函数和变量使用 `snake_case`，模型类使用 `PascalCase`，实验常量使用 `UPPER_SNAKE_CASE`，例如 `BATCH_SIZE`。集中组织导入语句，将模型定义、损失函数、训练和可视化放在职责清晰的单元格中。使用简洁的 Markdown 或注释说明算法与张量形状。目前未配置格式化或静态检查工具。

## 测试指南

目前未配置自动化测试框架或覆盖率要求。修改实验后，应重启内核并按顺序执行相关单元格，检查张量形状、损失值是否有限，以及生成的图表是否合理。初步检查可减少训练轮数，并在提交说明中记录数据集、运行设备和训练设置。检查 Notebook 输出，避免提交无关的输出变化。

## 提交与拉取请求指南

由 Codex 或相关 GPT agent 创建提交时，保留仓库当前配置的用户作为提交作者，并在提交说明末尾空一行后添加以下联合署名：

```text
Co-authored-by: Codex <codex@openai.com>
```

此署名用于记录 agent 的参与，不保证 GitHub 会将该邮箱关联到账号或在 Contributors 中展示。除非用户明确要求，不修改已有提交的署名。

现有提交使用简短、以动作为导向的说明，例如 `add README` 和 `add conda env config reference file`。每次提交应聚焦单一主题，并说明具体修改。拉取请求应介绍实验或修复内容、列出受影响的 Notebook，并概述验证结果。有关联问题时附上链接；修改模型行为或可视化时，提供有代表性的结果图。

# VAE：变分自编码器

对应 Notebook：`VAE.ipynb`。包含全连接 VAE 和卷积 VAE（`ConvVAE`），以 MNIST 图像为例学习潜在表示与图像生成。

## 1. 核心思想

普通自编码器将输入压缩为一个确定的向量；VAE 则让编码器输出潜变量分布的参数。这里采用对角高斯分布：

$$
q_\phi(z\mid x)=\mathcal N\bigl(\mu(x),\operatorname{diag}(\exp(\mathrm{log\_var}(x)))\bigr).
$$

编码器输出均值 `mu` 和对数方差 `log_var`，解码器根据采样得到的 `z` 重构输入。推理时也可以直接从标准正态分布采样并解码，生成新图像。

## 2. 全连接 VAE

`VAE` 的数据流如下：

- 输入：展平的 MNIST 图像，形状 `(B, 784)`。
- 编码器：`784 → 400`，经过 ReLU，再分别映射到 `mu` 和 `log_var`。
- 潜变量：两个分布参数均为 `(B, 30)`，其中 30 来自 `LATENT_DIM`。
- 解码器：`30 → 400 → 784`，最后使用 Sigmoid 将输出限制在 `[0, 1]`。

`forward(x)` 返回 `(x_recon, mu, log_var)`。当前 `train()` 使用的是 `ConvVAE`；若改用 `VAE`，需要将输入展平，并按需将生成结果还原为图像。

## 3. 卷积 VAE（ConvVAE）

Notebook 中的 **CVAE 指 Convolutional VAE，不是 Conditional VAE**。代码没有标签条件输入。

卷积版本保留图像的空间结构：

| 阶段 | 张量形状（不含批次维） |
| --- | --- |
| 输入 | `1 × 28 × 28` |
| 两层步长为 2 的卷积 | `32 × 14 × 14 → 64 × 7 × 7` |
| 展平并输出分布参数 | `3136 → mu / log_var` |
| 潜变量映射回特征图 | `30 → 3136 → 64 × 7 × 7` |
| 两层转置卷积 | `32 × 14 × 14 → 1 × 28 × 28` |

与全连接版本相比，卷积版本利用局部感受野和参数共享，更适合图像输入；这里的空间尺寸设计专门对应 28×28 图像。

## 4. 重参数化技巧

直接采样不便于对分布参数反向传播，因此将随机性移到独立噪声：

$$
\epsilon\sim\mathcal N(0,I),\qquad z=\mu+\exp(\tfrac12\mathrm{log\_var})\odot\epsilon.
$$

代码中的 `reparameterize()` 使用 `torch.randn_like(std)` 生成噪声，使重构损失能通过 `z` 更新编码器。两个模型的 `forward()` 都会采样；仅调用 `eval()` 不会自动改成使用均值。

## 5. 损失与当前训练设置

实际代码采用加权目标，而非未经加权的标准负 ELBO：

$$
L=0.3\,\mathrm{BCE}(\hat x,x)+D_{KL}(q_\phi(z\mid x)\Vert\mathcal N(0,I)),
$$
$$
D_{KL}=-\frac12\sum_j(1+\mathrm{log\_var}_j-\mu_j^2-\exp(\mathrm{log\_var}_j)).
$$

BCE 衡量像素重构误差，KL 项约束潜在分布接近标准正态分布。两项均按元素求和，训练日志再除以数据集大小；调整权重会改变重构质量与潜在空间约束的平衡。

当前训练使用 `ConvVAE`、Adam、批量大小 128、学习率 `1e-3` 和 15 轮训练，每 5 轮从先验采样生成图像。数据下载到 `data/`。生成结果与损失曲线用于观察实验表现，不能仅凭训练损失判断生成质量。

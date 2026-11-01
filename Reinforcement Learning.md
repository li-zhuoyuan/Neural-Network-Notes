# 强化学习：AC、PPO、SAC 与 GRPO

对应 Notebook：`Reinforcement Learning.ipynb`。四种算法均围绕 CartPole 离散动作任务组织，采用“网络定义 → 算法类 → 训练函数”的结构。

## 1. 公共概念与接口

状态 `s` 描述环境，动作 `a` 由策略选择，奖励 `r` 衡量当前行为。折扣因子 `gamma` 控制未来奖励的重要程度。

- `PolicyNet`：状态经过隐藏层和 softmax，输出每个离散动作的概率。
- `ValueNet`：输出状态价值 `V(s)`，用于 AC 和 PPO。
- `QValueNet`：输出各动作的 `Q(s,a)`，用于 SAC。
- `take_action`：AC/SAC 返回动作；PPO/GRPO 返回动作和采样时的对数概率。
- `update`：AC/PPO 接收一条轨迹字典，SAC 接收回放批次，GRPO 接收一组轨迹字典。

## 2. Actor-Critic（AC）

Actor 学习策略，Critic 学习价值。当前实现先采集完整回合，再用单步 TD 目标更新：

$$
y_t=r_t+\gamma(1-d_t)V(s_{t+1}),\qquad \delta_t=y_t-V(s_t).
$$
$$
L_V=\operatorname{MSE}(V(s_t),\operatorname{stopgrad}(y_t)),\qquad
L_\pi=-\mathbb E[\log\pi(a_t\mid s_t)\operatorname{stopgrad}(\delta_t)].
$$

TD 误差为正时提高所选动作的概率，反之降低。`detach()` 避免 Actor 更新流入 Critic，以及目标值参与 Critic 的梯度计算。入口为 `train_actor_critic()`，默认 800 回合。

## 3. PPO：裁剪策略更新

PPO 保存采样时的旧动作对数概率，通过新旧策略概率比限制更新幅度：

$$
\rho_t=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)),
$$
$$
L_\pi=-\mathbb E[\min(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)].
$$

当前代码先根据 TD 误差逆序计算 GAE：`A_t = delta_t + gamma * lambda * A_(t+1)`，再标准化优势。采集一个回合后，对同一批数据执行多轮 Actor 与 Critic 更新。奖励、动作概率和优势均保持一维，避免广播造成形状错误。

入口为 `train_PPO()`，默认 400 回合。它仍是 on-policy 方法，不能直接像 SAC 一样反复使用任意历史回放数据。

## 4. SAC：离散动作与最大熵目标

这里是 **SAC-Discrete**，不是连续高斯策略版本。除奖励外，SAC 通过熵鼓励探索。使用固定温度 `alpha`，未实现自动温度调节。

双 Q 网络取较小值，降低价值高估；目标网络通过软更新平滑变化。由于动作数量有限，可以直接对所有动作求和：

$$
V_{target}(s')=\sum_a\pi(a\mid s')[\min_i Q_{target,i}(s',a)-\alpha\log\pi(a\mid s')],
$$
$$
y=r+\gamma(1-d)V_{target}(s'),
$$
$$
L_\pi=\mathbb E_s\sum_a\pi(a\mid s)[\alpha\log\pi(a\mid s)-\min_iQ_i(s,a)].
$$

`ReplayBuffer` 保存交互数据，预热阶段随机探索，达到采样门槛后每步抽取批次更新。目标网络按 `target ← (1−tau) * target + tau * online` 更新，不直接接受梯度。

入口为 `train_SAC()`，默认 400 回合，可用 `record_video=False` 关闭录像。回放中的 `dones` 仅记录真正终止；时间上限截断结束回合，但仍允许价值 bootstrap。

参考：[Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207)。

## 5. GRPO：组相对策略优化

这里是将 GRPO 思想用于 CartPole 的**教学适配版**，没有语言模型、文本 token 或奖励模型。初始状态对应同一个问题，完整轨迹对应一个回答，动作对应 token。

每组用同一个旧策略从相同初始状态采集多条轨迹。环境 seed 在组内相同，动作采样随机数不重置，因此轨迹仍可不同。组内采样完成后再统一更新：

$$
R_i=\sum_t r_{i,t},\qquad A_i=\frac{R_i-\operatorname{mean}(R)}{\operatorname{std}(R)+10^{-8}}.
$$

不训练 Critic，同一轨迹各步共享组相对优势。使用与 PPO 相似的裁剪代理目标，并添加参考策略 KL 惩罚。参考策略在初始化时复制并冻结；它不同于每组采样时的旧策略。

本实现对离散动作精确计算：

$$
D_{KL}(\pi\Vert\pi_{ref})=\sum_a\pi(a\mid s)[\log\pi(a\mid s)-\log\pi_{ref}(a\mid s)].
$$

每条轨迹先按时间步平均损失，再按组平均，避免长轨迹获得额外权重。组内奖励相同时优势为零，仅 KL 项可能驱动更新；参考策略也相同时没有有效学习信号。

入口为 `train_GRPO()`，默认 100 组、每组 8 条轨迹；不是 100 个回合。此实现使用有限回合总奖励，不做价值 bootstrap。参考：[DeepSeekMath](https://arxiv.org/abs/2402.03300)。

## 6. 对比与实现边界

| 算法 | 数据组织 | 价值网络 | 核心机制 |
| --- | --- | --- | --- |
| AC | 新采集的完整回合 | `V(s)` | TD 误差指导策略 |
| PPO | 新回合，多轮更新 | `V(s)` | GAE 与概率比裁剪 |
| SAC | 历史回放批次 | 双 `Q(s,a)` | 最大熵、目标网络 |
| GRPO | 同一起点的一组新轨迹 | 无 | 组相对优势与参考 KL |

现有 AC/PPO 将环境终止与时间上限截断合并处理，会在两者发生时都屏蔽 bootstrap；这与新增 SAC 的处理不同。PPO 的 GAE 目前按单个完整回合计算，若以后拼接多回合，需要在回合边界重置递推。

录像输出位于被 Git 忽略的 `videos/`。训练函数与调用分别位于代码单元格中；运行全部单元格会依次启动训练。短程运行只用于发现接口或数值问题，不代表策略收敛。

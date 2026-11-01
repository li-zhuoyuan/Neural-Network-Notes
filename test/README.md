# 测试与实验脚本

先激活参考环境：`conda activate D2L`。以下 Python 脚本根据自身位置定位仓库，可从任意工作目录启动；下面命令以仓库根目录为例。它们是直接运行的脚本，不使用 pytest，导入脚本也会执行对应任务。

| 命令 | 作用与副作用 |
| --- | --- |
| `python test/test_rnn_structure.py` | 快速检查 RNN/GRU/LSTM 输出及状态形状、分段递推一致性、梯度有限性和空输入；不训练、不修改 Notebook |
| `python test/test_rl_run.py` | 完整训练 SAC 400 回合、GRPO 100 组，并各评估 20 回合；检查损失、参数和录像数量，保存到 `data/rl_test_*`、`videos/*/test_*`，通常耗时数分钟 |
| `python test/test_word_prediction.py` | 顺序执行 `RNN.ipynb`，训练三种词预测模型并推理；检查数据划分、补齐、OOV、状态缓存和随机采样；会更新 Notebook 输出及 `data/rnn_word_prediction/`、`img/rnn_word_comparison.png` |
| `python test/test_word_plot.py` | 从已有词预测报告重绘曲线并同步 Notebook 图像；不重新训练，需要 `data/rnn_word_prediction/results.json` |

`testing.ipynb` 是保留的 PyTorch 基础练习 Notebook，不是自动化测试套件；可在 Jupyter 中打开并按需运行。旧的二进制记忆任务已从主 Notebook 移除，对应脚本已替换为不训练的结构检查。

完整词预测实验需要先准备 `data/tatoeba/chinese.txt`，并安装 `environment.yml` 中的 jieba、PyTorch 和 Jupyter 依赖。测试数据、模型、图像、录像保持在仓库约定的忽略目录中；Jupyter 检查点与 Python 缓存不提交。

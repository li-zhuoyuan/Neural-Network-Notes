import json
import time
from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parents[1])
from datetime import datetime
import torch
import numpy as np

# 只执行导入、公共网络和 SAC/GRPO 定义，避免启动 AC/PPO。
n = json.loads(Path('Reinforcement Learning.ipynb').read_text(encoding='utf-8'))
scope = {'__name__': 'algorithm_validation'}
for i in (3, 4, 13, 14, 17, 18):
    exec(compile(''.join(n['cells'][i]['source']), f'notebook_cell_{i}', 'exec'), scope)
torch.set_num_threads(1)
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
output = Path('data') / ('rl_test_' + run_id)
output.mkdir(parents=True, exist_ok=True)
original_video = scope['RecordVideo']
video_dirs = []
def record_video(env, video_folder, **kwargs):
    folder = Path(video_folder) / ('test_' + run_id)
    video_dirs.append(folder)
    return original_video(env, video_folder=str(folder), **kwargs)
scope['RecordVideo'] = record_video
results = {'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'seed': 0}
print('DEVICE:', results['device'], 'REPORT:', output, flush=True)
for name, train in [('SAC', scope['train_SAC']), ('GRPO', scope['train_GRPO'])]:
    base = scope[name]
    stats = {'updates': 0}
    class CheckedAgent(base):
        def update(self, data):
            metrics = super().update(data)
            assert all(np.isfinite(v) for v in metrics.values()), metrics
            stats['updates'] += 1
            stats['last_metrics'] = metrics
            return metrics
    scope[name] = CheckedAgent
    start = time.perf_counter()
    agent, rewards = train()
    assert np.isfinite(rewards).all()
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())
    if name == 'SAC':
        assert not any(p.requires_grad for p in agent.target_critic1.parameters())
    else:
        assert not any(p.requires_grad or p.grad is not None for p in agent.reference_actor.parameters())
    # 在不同于训练的初始状态上评估，仍按随机策略采样动作。
    env = scope['gym'].make('CartPole-v1')
    eval_rewards = []
    try:
        for j in range(20):
            state, _ = env.reset(seed=10000 + j)
            total = 0
            while True:
                action = agent.take_action(state)
                if isinstance(action, tuple):
                    action = action[0]
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            eval_rewards.append(total)
    finally:
        env.close()
    videos = list(video_dirs[-1].glob('*.mp4'))
    assert len(videos) == (4 if name == 'SAC' else 8), videos
    assert all(p.stat().st_size > 0 for p in videos)
    results[name] = dict(stats, episodes=len(rewards), first_50_mean=float(np.mean(rewards[:50])),
                         last_50_mean=float(np.mean(rewards[-50:])),
                         eval_mean=float(np.mean(eval_rewards)), eval_min=min(eval_rewards),
                         eval_max=max(eval_rewards), eval_rewards=eval_rewards,
                         elapsed_seconds=round(time.perf_counter() - start, 2),
                         videos=[str(p) for p in videos])
    np.save(output / (name + '_rewards.npy'), rewards)
    torch.save(agent.actor.state_dict(), output / (name + '_actor.pt'))
    (output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RESULT', name, json.dumps(results[name], ensure_ascii=False), flush=True)
print('ALL PASSED', flush=True)

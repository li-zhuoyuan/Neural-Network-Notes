"""不训练、不改写 Notebook，检查三种手写循环网络的形状、递推和梯度。"""
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'RNN.ipynb').read_text(encoding='utf-8'))
scope = {'__name__': 'structure_validation'}
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and '## 示例：' in ''.join(cell['source']):
        break
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'RNN.ipynb cell {i}', 'exec'), scope)

torch.manual_seed(0)
torch.set_num_threads(1)
for name in ('RNN', 'GRU', 'LSTM'):
    model = scope[name](input_dim=3, hidden_dim=8, output_dim=5)
    x = torch.randn(2, 6, 3)
    output, state = model(x)
    assert output.shape == (2, 6, 5)
    states = state if isinstance(state, tuple) else (state,)
    assert all(s.shape == (2, 8) for s in states)
    # 分段传递状态必须与完整序列递推一致。
    first, middle_state = model(x[:, :3])
    second, _ = model(x[:, 3:], middle_state)
    assert torch.allclose(output, torch.cat((first, second), dim=1), atol=1e-6)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    try:
        model(torch.zeros(2, 0, 3))
    except ValueError:
        pass
    else:
        raise AssertionError(f'{name} 未拒绝空序列')
    print(f'PASS {name}: output/state shape, chunked recurrence, finite gradients, empty input')

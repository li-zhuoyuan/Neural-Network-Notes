import base64
import contextlib
import io
import json
import sys
from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parents[1])
import matplotlib
matplotlib.use('Agg')
import nbformat

p = Path('RNN.ipynb')
n = nbformat.read(p, as_version=4)
nbformat.validate(n)
scope = {'__name__': 'notebook_training'}
class Tee:
    def __init__(self): self.buffer = io.StringIO()
    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.buffer.write(s)
    def flush(self): sys.__stdout__.flush()
count = 0
for i, c in enumerate(n.cells):
    if c.cell_type != 'code': continue
    count += 1
    tee = Tee()
    # 与交互 Notebook 相同顺序执行代码，记录真实 stdout/stderr 与图表。
    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
        exec(compile(c.source, f'RNN.ipynb cell {i}', 'exec'), scope)
    c.execution_count = count
    c.outputs = [nbformat.v4.new_output('stream', name='stdout', text=tee.buffer.getvalue())] if tee.buffer.getvalue() else []
    if i == len(n.cells) - 1:
        fig = scope['word_figure']
        Path('img').mkdir(exist_ok=True)
        fig.savefig('img/rnn_word_comparison.png', dpi=150)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        c.outputs.append(nbformat.v4.new_output('display_data', data={
            'image/png': base64.b64encode(buf.getvalue()).decode('ascii'),
            'text/plain': 'RNN / GRU / LSTM word prediction results'}))
    nbformat.write(n, p)

# 验证同一 token 序列不会跨数据集，且边界/pad 对齐正确。
data = scope['word_data']
splits = [set(map(tuple, data['raw'][key])) for key in ('train', 'val', 'test')]
assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
torch = scope['torch']
inputs, targets = scope['collate_word_batch']([torch.tensor([2, 4, 3]), torch.tensor([2, 5, 6, 3])])
assert inputs.tolist() == [[2, 4, 0], [2, 5, 6]]
assert targets.tolist() == [[4, 3, 0], [5, 6, 3]]
inference = {}
for name, result in scope['word_results'].items():
    model = result['model']
    assert all(torch.isfinite(param).all() for param in model.parameters())
    inference[name] = {}
    for prefix in ('我喜欢', '今天', '你为什么', '我们一起'):
        inference[name][prefix] = {
            'next': scope['predict_next_words'](model, prefix, data),
            'greedy': scope['generate_words'](model, prefix, data),
            'sample': scope['generate_words'](model, prefix, data, sample=True, seed=1)}
    unknown = scope['predict_next_words'](model, 'QZXVUNSEENWORD', data)
    assert unknown['unknown_words']
    assert scope['predict_next_words'](model, '', data)['candidates']
    a = scope['generate_words'](model, '我', data, sample=True, seed=123)
    b = scope['generate_words'](model, '我', data, sample=True, seed=123)
    assert a == b
    # 逐步缓存状态与从头计算 logits 一致（右补齐不参与此项检查）。
    ids = torch.tensor([[2, 4, 5]], device=next(model.parameters()).device)
    with torch.no_grad():
        full, _ = model(ids)
        _, state = model(ids[:, :2])
        last, _ = model(ids[:, 2:], state)
    assert torch.allclose(full[:, -1], last[:, -1], atol=1e-5, rtol=1e-4)
Path('data/rnn_word_prediction/inference.json').write_text(json.dumps(inference, ensure_ascii=False, indent=2), encoding='utf-8')
nbformat.validate(n)
print('PASS: complete training, disjoint splits, target shift/padding, finite parameters, OOV/empty prefixes, seeded sampling and cached state')

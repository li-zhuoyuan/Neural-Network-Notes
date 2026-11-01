import json
import base64
import io
from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parents[1])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = Path('RNN.ipynb')
n = json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    source = ''.join(c['source'])
    if 'def plot_word_results(' in source:
        source = source.replace("axes[2].set(title='Test top-1 accuracy', ylabel='Accuracy')", "axes[2].set(title='Test top-1 accuracy', ylabel='Accuracy',\n                ylim=(0, min(1.15, max(r['test']['top1'] for r in results.values()) + .20)))")
        c['source'] = source.splitlines(keepends=True)
        plot_source = source[source.index('def plot_word_results('):]
scope = {'plt': plt}
exec(plot_source, scope)
results = json.loads(Path('data/rnn_word_prediction/results.json').read_text(encoding='utf-8'))['results']
fig = scope['plot_word_results'](results)
fig.savefig('img/rnn_word_comparison.png', dpi=150)
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=120)
for output in n['cells'][-1]['outputs']:
    if output['output_type'] == 'display_data':
        output['data']['image/png'] = base64.b64encode(buf.getvalue()).decode('ascii')
p.write_text(json.dumps(n, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')

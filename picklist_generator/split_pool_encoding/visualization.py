"""Standalone PNG/PDF figures without Tk or pyplot global state."""
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def save(fig, path):
    FigureCanvasAgg(fig)
    for extension in ('png', 'pdf'):
        fig.savefig(Path(path).with_suffix('.'+extension), dpi=140, bbox_inches='tight')
    fig.clear()


def result_figures(d, result, noise, directory):
    directory = Path(directory); p = result['prediction']; target = d[result['assignment']]
    fig = Figure(figsize=(10, 9)); axes = fig.subplots(3, 1)
    for ax, values, title in zip(axes, (target, p, abs(p-target)), ('Target (address order)', 'Reconstruction', 'Absolute error')):
        im = ax.imshow(values, aspect='auto', vmin=0, vmax=1, cmap='viridis'); ax.set_title(title)
        ax.set_xlabel('Analog position'); ax.set_ylabel('Address'); fig.colorbar(im, ax=ax)
    fig.tight_layout(); save(fig, directory/'heatmaps')
    fig = Figure(figsize=(10, 7)); axes = fig.subplots(2, 2).ravel()
    for ax, row in zip(axes, np.linspace(0, len(d)-1, 4, dtype=int)):
        ax.plot(target[row], label='Target'); ax.plot(p[row], label='Reconstruction'); ax.set_title(f'Address {row}'); ax.set_ylim(-.05, 1.05); ax.legend()
    fig.tight_layout(); save(fig, directory/'canvases')
    fig = Figure(figsize=(9, 4)); ax = fig.subplots()
    history = result['history']
    if history:
        ax.plot([h['mse'] for h in history], label='Selected restart')
        ids = [i for i, h in enumerate(history) if h['phase'] == 'assignment']
        ax.scatter(ids, [history[i]['mse'] for i in ids], s=12, label='Hungarian update'); ax.legend()
    ax.set(xlabel='Logged optimization step', ylabel='MSE', title='Convergence'); save(fig, directory/'learning')
    if result['trials']:
        fig = Figure(figsize=(7, 4)); ax = fig.subplots()
        ax.plot([r['restart'] for r in result['trials']], [np.sqrt(r['mse']) for r in result['trials']], 'o')
        ax.set(xlabel='Random initialization', ylabel='Final RMSE', title='Search sensitivity (all restarts)')
        save(fig, directory/'restarts')
    if noise:
        fig = Figure(figsize=(8, 4)); ax = fig.subplots(); counts = sorted({r['samples'] for r in noise})
        for key in ('encoding_rmse', 'sampling_rmse', 'total_rmse'):
            values = [[r[key] for r in noise if r['samples'] == n] for n in counts]
            ax.errorbar(counts, np.mean(values, axis=1), yerr=np.std(values, axis=1), label=key)
        ax.set(xlabel='Origami per barcode', ylabel='RMSE'); ax.legend(); save(fig, directory/'sampling')


def comparison(rows, path):
    fig = Figure(figsize=(12, 5)); ax = fig.subplots()
    datasets = list(dict.fromkeys(r['dataset'] for r in rows))
    for model in ('pca', 'additive', 'on_only', 'on_off'):
        values = [[r['rmse'] for r in rows if r['dataset'] == d and r['model'] == model and r['flexible']] for d in datasets]
        ax.errorbar(range(len(datasets)), [np.mean(v) if v else np.nan for v in values],
                    yerr=[np.std(v) if v else 0 for v in values], marker='o', label=model)
    ax.set_xticks(range(len(datasets)), datasets, rotation=35, ha='right'); ax.set_ylabel('RMSE (mean ± SD across realizations)'); ax.legend()
    save(fig, path)


def schematic(path):
    fig = Figure(figsize=(11, 5)); ax = fig.subplots(); ax.axis('off')
    for r in range(4):
        for a in range(2**r):
            y = 1-(a+.5)/2**r
            ax.text(r, y, format(a, f'0{r}b') if r else 'U = 1', ha='center', bbox=dict(facecolor='lightblue', edgecolor='gray'))
            if r:
                ax.annotate('', (r-.12, y), (r-1+.12, 1-(a//2+.5)/2**(r-1)), arrowprops=dict(arrowstyle='->'))
        if r:
            ax.text(r, 1.08, f'Round {r}: mixtures M{r},0 / M{r},1', ha='center', fontsize=9)
    ax.set(xlim=(-.3, 3.4), ylim=(-.16, 1.18))
    ax.text(1.5, -.12, 'Each round: split → branch-specific mixture → wash → pool. Only U can become ON or OFF.\nON and OFF remain occupied in every later round.', ha='center')
    save(fig, path)

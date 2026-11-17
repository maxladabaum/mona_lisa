"""Export exactly representable targets: python -m split_pool_encoding.exact_benchmark --output DIR."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .datasets import generate, replay_exact_k


def run(output, seed=17):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    rows = []
    for row, k in enumerate((2, 4, 8)):
        target, truth = generate(f'exact_k_{k}', seed=seed)
        np.savez_compressed(output/f'k{k}_recipe.npz', **truth)
        np.savetxt(output/f'k{k}_targets.csv', target, delimiter=',', fmt='%.17g')
        # Test the saved files, including row/address correspondence.
        with np.load(output/f'k{k}_recipe.npz', allow_pickle=False) as saved:
            replay = replay_exact_k(saved)
        target = np.loadtxt(output/f'k{k}_targets.csv', delimiter=',')
        error = replay-target
        rows.append(dict(k=k, targets=len(target), positions=target.shape[1], rounds=int(truth['rounds']),
                         rmse=float(np.sqrt(np.mean(error**2))), max_error=float(np.max(np.abs(error))),
                         minimum=float(target.min()), maximum=float(target.max()),
                         unique_targets=len(np.unique(target, axis=0))))
        for ax, values, title in zip(axes[row], (target, replay, error),
                (f'K={k}: generated targets', 'Replay of saved recipe', f'Error · RMSE {rows[-1]["rmse"]:.2g}')):
            image = ax.imshow(values, aspect='auto', interpolation='nearest',
                              cmap='coolwarm' if title.startswith('Error') else 'viridis',
                              vmin=-1e-14 if title.startswith('Error') else 0,
                              vmax=1e-14 if title.startswith('Error') else 1)
            ax.set_title(title); ax.set_xlabel('Analog site'); ax.set_ylabel('Target')
            fig.colorbar(image, ax=ax, shrink=.7)
        branch = truth['branch_transfers_nl']
        with (output/f'k{k}_branch_volumes.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['round', 'branch']+[f'K{i+1} nL' for i in range(k)])
            for index, volumes in enumerate(branch):
                writer.writerow([index//2+1, index%2, *volumes])
    fig.suptitle('Known feasible 25 nL ON/OFF recipes · 8 split–pool rounds\n'
                 'Expected occupancy; saved-recipe replay, not blind optimizer recovery')
    fig.savefig(output/'exact_k_comparison.png', dpi=150)
    plt.close(fig)
    (output/'metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2))

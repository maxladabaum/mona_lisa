"""Run with python -m split_pool_encoding from picklist_generator."""
import argparse
import json
from pathlib import Path
import numpy as np
from .datasets import FAMILIES, generate, load
from .experiment import DEFAULTS, run_dataset, write_csv, report
from .visualization import comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--input', type=Path, help='Headerless CSV or .npy matrix; dimensions inferred')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--datasets', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--bits', type=int, nargs='+', help='One or more round counts, e.g. 2 3 4 5 6 7 8')
    parser.add_argument('--validate-known', action='store_true', help='Export oracle-start forward/assignment checks')
    parser.add_argument('--strategy', choices=['joint', 'per_site'])
    parser.add_argument('--quick', action='store_true', help='Smoke-test budget, not a capacity experiment')
    args = parser.parse_args(); cfg = {**DEFAULTS, **(json.loads(args.config.read_text()) if args.config else {})}
    if args.strategy:
        cfg["strategy"] = args.strategy
    if args.quick:
        cfg.update(restarts=1, alternations=3, maxiter=35, realizations=1, measurement_repeats=10)
    args.output.mkdir(parents=True, exist_ok=False); rows = []
    for bits in args.bits or [cfg['bits']]:
        for realization in range(cfg['realizations']):
            for family in (['input'] if args.input else args.datasets):
                d, truth = (load(args.input), {}) if args.input else generate(family, bits, cfg['positions'], cfg['seed']+realization)
                label = f'{family}_b{int(np.log2(len(d)))}'
                folder = args.output/f'{label}_rep{realization}'
                rows.extend(run_dataset(d, folder, cfg, label, realization, progress=lambda s: print(s, flush=True)))
                if truth:
                    np.savez_compressed(folder/'generator_truth.npz', **truth)
                write_csv(args.output/'metrics.csv', rows)
        if args.input:
            break
    if args.validate_known:
        from .validation import known_solution_checks
        write_csv(args.output/'known_solution_checks.csv', known_solution_checks(cfg['bits'], cfg['positions'], cfg['seed']))
    if cfg['figures']:
        comparison(rows, args.output/'structure_comparison')
    report(rows, args.output/'report.md', cfg)


if __name__ == '__main__':
    main()

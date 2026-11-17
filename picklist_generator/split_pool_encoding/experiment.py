"""Reproducible experiment exports shared by the GUI and CLI."""
import csv
import json
from pathlib import Path
import numpy as np
from .models import MODELS, validate_data, physical_parameters
from .optimization import fit
from .evaluation import metrics, sampling, transfer_estimate, precision_sweep
from .visualization import result_figures, comparison, schematic

DEFAULTS = dict(bits=8, positions=40, seed=17, restarts=3, alternations=12, maxiter=150,
                realizations=2, sample_counts=[50, 100, 200, 300, 500, 1000], measurement_repeats=100,
                compare_fixed=True, figures=True, strategy="joint", false_positive=0., false_negative=0.)


def write_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_dataset(d, output, config=None, label='input', realization=0, progress=lambda s: None, cancel=None, metadata=None):
    cfg = {**DEFAULTS, **(config or {})}; d, bits = validate_data(d)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps({**cfg, 'actual_bits': bits, 'actual_positions': d.shape[1]}, indent=2))
    if metadata is not None:
        (output/'target_metadata.json').write_text(json.dumps(metadata, indent=2))
    np.save(output/'targets.npy', d); rows = []
    for model in MODELS:
        for flexible in ([True, False] if cfg['compare_fixed'] and model != 'pca' else [True]):
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Analysis cancelled; completed outputs retained.')
            name = model+('' if flexible else '_fixed'); progress(f'{label}: fitting {name}')
            result = fit(d, model, seed=cfg['seed']+realization, restarts=cfg['restarts'], alternations=cfg['alternations'],
                         maxiter=cfg['maxiter'], flexible=flexible, cancel=cancel, strategy=cfg["strategy"])
            folder = output/name; folder.mkdir()
            parameters = physical_parameters(result['parameters'], model) if model != 'pca' else dict(mean=result['pca_mean'], scores=result['pca_scores'])
            np.savez_compressed(folder/'model.npz', prediction=result['prediction'], assignment=result['assignment'],
                                parameters=result['parameters'], **parameters)
            write_csv(folder/'assignment.csv', [dict(address=a, barcode=format(a, f'0{bits}b'), target_row=int(t)) for a, t in enumerate(result['assignment'])])
            target = d[result['assignment']]
            write_csv(folder/'precision.csv', precision_sweep(target, result['parameters'], model, bits))
            per_file = np.sqrt(np.mean((result['prediction']-target)**2, axis=1))
            write_csv(folder/'per_file.csv', [dict(address=a, target_row=int(result['assignment'][a]), rmse=float(e)) for a, e in enumerate(per_file)])
            row = dict(dataset=label, realization=realization, strategy=cfg["strategy"], model=model, flexible=flexible, bits=bits, positions=d.shape[1],
                       **metrics(target, result['prediction']), runtime_seconds=result['runtime'],
                       parameters=result['parameter_count'], estimated_transfers=transfer_estimate(model, bits, d.shape[1]), converged=result['converged'])
            rows.append(row); write_csv(output/'metrics.csv', rows)
            write_csv(folder/'convergence.csv', result['history']); write_csv(folder/'restarts.csv', result['trials'])
            noise = [] if model == 'pca' else sampling(target, result['prediction'], counts=cfg['sample_counts'],
                repeats=cfg['measurement_repeats'], seed=cfg['seed']+realization,
                false_positive=cfg['false_positive'], false_negative=cfg['false_negative'])
            write_csv(folder/'sampling.csv', noise)
            summaries = []
            for n in cfg['sample_counts'] if noise else []:
                subset = [r for r in noise if r['samples'] == n]
                summaries.append(dict(samples=n, **{f'{key}_{stat}': float(function([r[key] for r in subset]))
                    for key in ('encoding_rmse', 'sampling_rmse', 'total_rmse')
                    for stat, function in (('mean', np.mean), ('std', np.std), ('p95', lambda a: np.quantile(a, .95)))}))
            write_csv(folder/'sampling_summary.csv', summaries)
            if cfg['figures']:
                result_figures(d, result, noise, folder)
    if cfg['figures']:
        comparison(rows, output/'comparison'); schematic(output/'synthesis')
    report(rows, output/'report.md', cfg)
    progress(f'Finished: {output}')
    return rows


def report(rows, path, cfg):
    lines = ['# Split–pool experiment', '', 'Measured optimizer results (bounded search budget; not global optima).', '',
             '| Dataset | Model | Flexible | RMSE | Max error | Seconds |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['flexible']} | {r['rmse']:.6g} | {r['max_absolute_error']:.6g} | {r['runtime_seconds']:.2f} |")
    lines += ['', '## Interpretation', '',
        'Compare fixed and flexible rows within each model. Restart losses are exported separately; differences between restarts diagnose search sensitivity. Residual error on a generated model dataset does not establish a physical limitation: its generator proves representability.', '',
        'ON-only obeys −log(1−p) = Σr −log(1−u[r,a_r]). It is additive in hazard space with nonnegative branch hazards; its nonlinear probability output does not provide arbitrary high-order freedom.', '',
        'ON/OFF introduces ordered interactions through products of remaining availability. It contains ON-only (v=0), but neither universality nor superior fitted error follows. At 8 rounds and 40 sites its smooth polynomial image has dimension at most 1280 inside a 10240-dimensional target space. A finite union over row permutations still has measure zero: almost every arbitrary target collection cannot be represented exactly.', '',
        'A constructive expressivity example: in two rounds, branch 0 of round 1 irreversibly fills a site with OFF, and branch 1 of round 2 incorporates ON with probability 0.6. All other transitions are zero. The outputs are [0, 0, 0, 0.6]. Neither an additive cube nor an ON-only hazard-additive cube can have this pattern, even after row permutation. Thus OFF competition adds expressivity, although this does not imply better accuracy on every dataset.', '',
        'The model is not identifiable: final-round OFF incorporation never affects final ON occupancy; earlier saturation also hides subsequent parameters. Parameter count is an upper bound, not information capacity. If each transition has L realizable levels, 1280 log2(L) is only a loose programming-bit upper bound, not a distinguishable-dataset count. Readout uncertainty and output collisions reduce distinguishability. Binomial standard deviation is at most 1/(2√N), about 0.029 at N=300; calibration, classification, correlations and biochemical errors remain additional sources.', '',
        'PCA uses independently fitted continuous coefficients and is not a split–pool implementation. It is not clipped. Sampling is omitted for PCA. Additive predictions are valid probabilities, but the additive transfer estimate assumes an independent positive/negative basis writing mechanism which is not supplied by the irreversible chemistry.', '',
        'Transfer estimates count one dispense per site-specific ON reagent (plus OFF for ON/OFF or additive) into each of 2R branch environments, plus 2R digital address dispenses. They exclude pooling, splitting, washes, cleanup, stock preparation, transfer increments, and premixing. They are not executable picklists or calibrated volumes.', '',
        f"Budget: {cfg['restarts']} restarts, {cfg['alternations']} assignment cycles, {cfg['maxiter']} L-BFGS iterations/cycle. Convergence flags describe numerical termination, not proof of optimality. Figures show mean ± SD across dataset realizations; restart distributions are in restarts.csv. Sampling plots show mean ± SD over repeated measurements.", '',
        'Use the bit sweep to measure scaling and compare low-rank, smooth, correlated and clustered datasets. Low rank alone does not imply a binary-cube distribution. No information-capacity estimate or universal accuracy claim can be justified by these finite experiments.']
    Path(path).write_text('\n'.join(lines)+'\n')

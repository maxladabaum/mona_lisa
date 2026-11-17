"""Search compact reversible bit circuits against single-use bulk synthesis costs.

No payload lookup table is used. Gates act on all possible n-bit strings, and
reversing their order decodes. Search is heuristic and makes no optimality claim.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from .digital_affine import binary_matrix
from .digital_bulk import compile_program, run, plot_run

ARITY = {'not': 1, 'swap': 2, 'cnot': 2, 'toffoli': 3}
NAMES = tuple(ARITY)
HEADER = struct.Struct('>4sHH')


def validate_gates(gates, width):
    for gate in gates:
        if gate['op'] not in ARITY:
            raise ValueError('Unknown reversible gate.')
        sites = gate['sites']
        if len(sites) != ARITY[gate['op']] or len(set(sites)) != len(sites):
            raise ValueError('Gate requires distinct control/target sites.')
        if any(not isinstance(s, (int, np.integer)) or not 0 <= s < width for s in sites):
            raise ValueError('Gate site outside payload width.')


def apply(values, gates, inverse=False):
    x = binary_matrix(values).copy()
    validate_gates(gates, x.shape[1])
    for gate in reversed(gates) if inverse else gates:
        sites = gate['sites']; op = gate['op']
        if op == 'not':
            x[:, sites[0]] ^= 1
        elif op == 'swap':
            x[:, sites] = x[:, sites[::-1]]
        elif op == 'cnot':
            x[:, sites[1]] ^= x[:, sites[0]]
        else:
            x[:, sites[2]] ^= x[:, sites[0]] & x[:, sites[1]]
    return x


def pack_circuit(width, gates):
    if not 1 <= width <= 65535 or len(gates) > 65535:
        raise ValueError('Circuit format supports 1–65535 sites and at most 65535 gates.')
    validate_gates(gates, width)
    index_bits = max(1, (width-1).bit_length())
    bits = []
    for gate in gates:
        bits.extend(int(b) for b in f'{NAMES.index(gate["op"]):02b}')
        for site in gate['sites']:
            bits.extend(int(b) for b in f'{site:0{index_bits}b}')
    return HEADER.pack(b'RBC1', width, len(gates))+np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def unpack_circuit(blob):
    if len(blob) < HEADER.size:
        raise ValueError('Truncated circuit.')
    magic, width, count = HEADER.unpack_from(blob)
    if magic != b'RBC1' or width < 1:
        raise ValueError('Invalid circuit header.')
    bits = np.unpackbits(np.frombuffer(blob[HEADER.size:], dtype=np.uint8))
    cursor = 0; index_bits = max(1, (width-1).bit_length())
    def number(size):
        nonlocal cursor
        if cursor+size > len(bits):
            raise ValueError('Truncated gate.')
        value = 0
        for b in bits[cursor:cursor+size]:
            value = (value << 1)|int(b)
        cursor += size
        return value
    gates = []
    for _ in range(count):
        op = NAMES[number(2)]
        gates.append(dict(op=op, sites=[number(index_bits) for _ in range(ARITY[op])]))
    if len(bits)-cursor >= 8 or bits[cursor:].any():
        raise ValueError('Unexpected circuit padding or trailing bytes.')
    validate_gates(gates, width)
    return width, gates


def cost(report, minimum_yield):
    m = report['requested_files']; baseline = report['payload_bit_transfer_baseline']
    w = report['destination_wells_consumed']; t = report['total_transfers']
    eligible = report['all_files_identifiable'] and report['useful_product_fraction'] >= minimum_yield
    # Lower is better. Never give credit for dropping requested files.
    return (not eligible, not (w < m and t < baseline),
            max(w/m, t/baseline), w/m+t/baseline, -report['useful_product_fraction'])


def evaluate(values, max_states, minimum_yield):
    best = None; best_score = None; failures = []
    for block in sorted(set(min(values.shape[1], n) for n in (1, 2, 4))):
        for policy, limit in [('all', 1), ('bounded', 2), ('exact', 1)]:
            program = compile_program(values, block, policy, limit)
            try:
                report, sim = run(values, program, max_states)
            except ValueError as exc:
                failures.append(dict(block_bits=block, pooling=policy, error=str(exc)))
                continue
            score = cost(report, minimum_yield)
            if best_score is None or score < best_score:
                best_score = score; best = (program, report, sim)
    if best is None:
        raise ValueError('All schedule candidates exceeded limits.')
    return best, failures


def search_encoding(values, evaluations=100, max_gates=12, decoder_bytes=None,
                    seed=17, max_states=4096, minimum_yield=.01, progress=None):
    values = binary_matrix(values)
    if evaluations < 0 or max_gates < 0 or not 0 < minimum_yield <= 1:
        raise ValueError('Nonnegative evaluation/gate budgets and 0 < minimum yield <= 1 required.')
    width = values.shape[1]; original_bytes = (values.size+7)//8
    # Default: decoder occupies at most one quarter of the uncompressed dataset.
    budget = original_bytes//4 if decoder_bytes is None else decoder_bytes
    if budget < HEADER.size:
        raise ValueError(f'Decoder budget must allow the {HEADER.size}-byte header; use a larger dataset or explicit --decoder-bytes.')
    baseline, failures = evaluate(values, max_states, minimum_yield)
    base_report = baseline[1]
    if not base_report['all_files_identifiable'] or base_report['useful_product_fraction'] < minimum_yield:
        raise ValueError('No baseline schedule meets coverage/yield constraints within simulation limits.')
    rng = np.random.default_rng(seed)
    archive = [([], cost(base_report, minimum_yield))]
    seen = {values.tobytes()}
    best_gates = []; best = baseline; best_score = cost(base_report, minimum_yield)
    history = []
    attempted = 0
    while len(history) < evaluations and attempted < max(100, evaluations*30) and max_gates:
        attempted += 1
        # Retain an elite beam, with occasional exploration from the identity.
        parent = archive[int(rng.integers(len(archive)))][0] if rng.random() > .15 else []
        gates = [dict(op=g['op'], sites=list(g['sites'])) for g in parent]
        action = rng.choice(['append', 'insert', 'replace', 'remove']) if gates else 'append'
        ops = [op for op, arity in ARITY.items() if arity <= width]
        op = str(rng.choice(ops))
        gate = dict(op=op, sites=rng.choice(width, ARITY[op], replace=False).tolist())
        if action == 'remove':
            gates.pop(int(rng.integers(len(gates))))
        elif action == 'replace':
            gates[int(rng.integers(len(gates)))] = gate
        else:
            gates.insert(len(gates) if action == 'append' else int(rng.integers(len(gates)+1)), gate)
        if len(gates) > max_gates or len(pack_circuit(width, gates)) > budget:
            continue
        encoded = apply(values, gates)
        key = encoded.tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidate, _ = evaluate(encoded, max_states, minimum_yield)
        report = candidate[1]; score = cost(report, minimum_yield)
        recovered = apply(encoded, gates, inverse=True)
        if not np.array_equal(values, recovered):
            raise AssertionError('Reversible circuit failed exact round trip.')
        # Only publish a transformation that does not worsen either physical cost,
        # and that preserves the minimum useful-product fraction.
        acceptable = (not score[0] and report['destination_wells_consumed'] <= base_report['destination_wells_consumed']
                      and report['total_transfers'] <= base_report['total_transfers'])
        if acceptable and score < best_score:
            best = candidate; best_gates = gates; best_score = score
        archive.append((gates, score))
        archive.sort(key=lambda item: (item[1], len(pack_circuit(width, item[0]))))
        archive = archive[:8]
        history.append(dict(evaluation=len(history)+1, gates=len(gates), decoder_bytes=len(pack_circuit(width, gates)),
                            destination_wells=report['destination_wells_consumed'], transfers=report['total_transfers'],
                            useful_product_fraction=report['useful_product_fraction'], acceptable=acceptable,
                            best_destination_wells=best[1]['destination_wells_consumed'], best_transfers=best[1]['total_transfers']))
        if progress:
            progress(history[-1])
    return dict(baseline=baseline, best=best, gates=best_gates, history=history,
                decoder_budget=budget, original_bytes=original_bytes, seed=seed,
                minimum_yield=minimum_yield, max_states=max_states, baseline_schedule_failures=failures)


def export(values, result, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    blob = pack_circuit(values.shape[1], result['gates'])
    (output/'decoder.rbc').write_bytes(blob)
    width, saved_gates = unpack_circuit((output/'decoder.rbc').read_bytes())
    encoded = apply(values, saved_gates)
    np.savetxt(output/'original.csv', values, fmt='%d', delimiter=',')
    np.savetxt(output/'encoded.csv', encoded, fmt='%d', delimiter=',')
    recovered = apply(np.loadtxt(output/'encoded.csv', delimiter=',', ndmin=2), saved_gates, inverse=True)
    np.testing.assert_array_equal(values, recovered)
    # Decode actual simulated payloads selected by their physically written address.
    program, report, sim = result['best']
    outputs = {}
    for population in sim.wells.values():
        for state in population.states:
            address = ''.join(map(str, state[width:]))
            outputs[address] = state[:width]
    measured = np.array([outputs[row['address']] for row in report['target_address_manifest']], dtype=np.uint8)
    np.testing.assert_array_equal(apply(measured, saved_gates, inverse=True), values)
    for name, item in [('before', result['baseline']), ('after', result['best'])]:
        (output/f'{name}_schedule.json').write_text(json.dumps(item[0], indent=2)+'\n')
        plot_run(item[0], item[1], item[2], output/f'{name}.png')
    summary = dict(original_bits=int(values.size), encoded_bits=int(encoded.size),
        original_packed_bytes=result['original_bytes'], decoder_bytes=len(blob), decoder_budget_bytes=result['decoder_budget'],
        gates=saved_gates, seed=result['seed'], evaluated_circuits=len(result['history']),
        exact_saved_file_round_trip=True, exact_simulated_payload_round_trip=True,
        minimum_useful_fraction=result['minimum_yield'], before=result['baseline'][1], after=report,
        baseline_schedule_failures=result['baseline_schedule_failures'],
        limitations=['Heuristic search, no optimality claim; identity retained if no non-worsening improvement is found.',
            'Same scheduler choices and state limit before/after. Decoder budget includes binary header and padding.',
            'No compression of payload bits. Address/order manifest is reported separately and is not payload lookup data.',
            'Bulk scheduler searches disjoint block-writing programs; it does not optimize all overlapping blocker programs.',
            'Ideal chemistry and modeled transfers only; digital_bulk assumptions also apply.'])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    before = result['baseline'][1]
    for ax, key, title, goal in zip(axes,
            ('destination_wells_consumed', 'total_transfers', 'useful_product_fraction'),
            ('Single-use destination wells', 'Modeled fluidic transfers', 'Useful product fraction'),
            (len(values), values.size, None)):
        bars = ax.bar(['Original', 'Encoded'], [before[key], report[key]])
        ax.bar_label(bars, fmt='%.3g', padding=3)
        if goal is not None:
            ax.axhline(goal, color='gray', linestyle='--', label=f'Goal: below {goal}')
            ax.legend(fontsize=8)
        ax.margins(y=.2); ax.set_title(title)
    fig.suptitle(f'Reversible encoding · {len(saved_gates)} gates · {len(blob)}-byte decoder'
                 '\nExact recovery from simulated products; unchanged payload bit count')
    fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)
    (output/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'search_history.json').write_text(json.dumps(result['history'], indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='Binary CSV or NumPy matrix')
    source.add_argument('--random', action='store_true', help='Generate random files')
    parser.add_argument('--files', type=int, default=32)
    parser.add_argument('--bits', type=int, default=16)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--evaluations', type=int, default=100)
    parser.add_argument('--max-gates', type=int, default=12)
    parser.add_argument('--decoder-bytes', type=int)
    parser.add_argument('--max-states', type=int, default=4096)
    parser.add_argument('--minimum-yield', type=float, default=.01)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory.')
    if args.random:
        values = np.random.default_rng(args.seed).integers(0, 2, (args.files, args.bits), dtype=np.uint8)
    else:
        values = np.load(args.input, allow_pickle=False) if args.input.suffix == '.npy' else np.loadtxt(args.input, delimiter=',', ndmin=2)
    values = binary_matrix(values)
    def progress(item):
        if item['evaluation'] % 10 == 0:
            print(f"{item['evaluation']} circuits: best {item['best_destination_wells']} wells / {item['best_transfers']} transfers", flush=True)
    result = search_encoding(values, args.evaluations, args.max_gates, args.decoder_bytes, args.seed,
                             args.max_states, args.minimum_yield, progress)
    summary = export(values, result, args.output)
    print(json.dumps({k: summary[k] for k in ('decoder_bytes', 'decoder_budget_bytes', 'gates', 'exact_simulated_payload_round_trip')}, indent=2))
    for name in ('before', 'after'):
        r = summary[name]
        print(f"{name}: {r['destination_wells_consumed']} wells; {r['total_transfers']} transfers; {r['useful_product_fraction']:.2%} useful")


if __name__ == '__main__':
    main()

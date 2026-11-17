"""Ideal bulk digital writing: flexible branching, first-write-wins mixtures.

CLI: python -m split_pool_encoding.digital_bulk --demo --output NEW_DIRECTORY
All predicted abundances are exact deterministic population fractions. No binding
kinetics, finite molecule sampling, carryover, volume or cleaning model is implied.
"""
from collections import defaultdict
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .digital_affine import binary_matrix

EMPTY = -1


@dataclass
class Population:
    # Keys contain payload sites followed by separately writable barcode sites.
    states: dict
    slot: int


class Simulator:
    def __init__(self, payload_bits, address_bits=0, max_states=100000):
        if payload_bits < 1 or address_bits < 0 or max_states < 1:
            raise ValueError('Positive payload width/state limit and nonnegative address width required.')
        self.payload_bits = payload_bits
        self.address_bits = address_bits
        self.width = payload_bits + address_bits
        self.max_states = max_states
        self.wells = {'start': Population({(EMPTY,)*self.width: 1.0}, 0)}
        self.library = {}
        self.used_mixtures = set()
        self.preparation_transfers = 0
        self.addition_transfers = 0
        self.routing_transfers = 0
        self.peak_wells = 1
        self.well_loads = 1
        self.used_destinations = {'start'}
        self.trace = []

    def _reserve(self, count):
        # Every new population destination gets a never-used physical slot.
        slots = list(range(self.well_loads, self.well_loads+count))
        self.peak_wells = max(self.peak_wells, len(self.wells)+count)
        self.well_loads += count
        return slots

    def _fresh(self, names):
        if len(names) != len(set(names)) or any(name in self.used_destinations for name in names):
            raise ValueError('Destination wells must be distinct and never previously used; emptied wells cannot be reused.')

    def _limit(self, future):
        if future > self.max_states:
            raise ValueError(f'Exact simulation would exceed {self.max_states} states; increase --max-states or reduce branching.')

    def split(self, source, outputs, fractions=None):
        if not outputs:
            raise ValueError('Split requires destinations.')
        self._fresh(outputs)
        p = self.wells[source]
        f = np.full(len(outputs), 1/len(outputs)) if fractions is None else np.asarray(fractions, float)
        if f.shape != (len(outputs),) or not np.isfinite(f).all() or np.any(f <= 0) or not np.isclose(f.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError('Split fractions must be positive and sum to one.')
        self._limit(sum(len(w.states) for w in self.wells.values())+(len(outputs)-1)*len(p.states))
        slots = self._reserve(len(outputs))
        for name, share, slot in zip(outputs, f, slots):
            self.wells[name] = Population({state: mass*float(share) for state, mass in p.states.items()}, slot)
        self.used_destinations.update(outputs)
        del self.wells[source]
        self.routing_transfers += len(outputs)

    def add(self, well, mixture):
        values = self.library[mixture]
        p = self.wells[well]
        states = defaultdict(float)
        for state, mass in p.states.items():
            result = tuple(v if old == EMPTY else old for old, v in zip(state, values))
            states[result] += mass
        p.states = dict(states)
        if mixture not in self.used_mixtures:
            self.preparation_transfers += sum(v != EMPTY for v in values)
            self.used_mixtures.add(mixture)
        self.addition_transfers += 1

    def pool(self, sources, destination):
        if not sources or len(set(sources)) != len(sources):
            raise ValueError('Pool requires distinct source wells.')
        self._fresh([destination])
        populations = [self.wells[name] for name in sources]
        slot = self._reserve(1)[0]
        states = defaultdict(float)
        for p in populations:
            for state, mass in p.states.items():
                states[state] += mass
        for name in sources:
            del self.wells[name]
        self.used_destinations.add(destination)
        self.wells[destination] = Population(dict(states), slot)
        self.routing_transfers += len(sources)

    def execute(self, program):
        if program['payload_bits'] != self.payload_bits or program.get('address_bits', 0) != self.address_bits:
            raise ValueError('Program geometry does not match simulator.')
        for name, values in program['mixtures'].items():
            if len(values) != self.width or any(v not in (-1, 0, 1) for v in values):
                raise ValueError('Each mixture must specify -1 (untouched), 0 (OFF), or 1 (ON) at every site.')
            self.library[name] = tuple(values)
        for index, op in enumerate(program['operations']):
            kind = op['op']
            if kind == 'split':
                self.split(op['source'], op['outputs'], op.get('fractions'))
            elif kind == 'add':
                self.add(op['well'], op['mixture'])
            elif kind == 'pool':
                self.pool(op['sources'], op['destination'])
            else:
                raise ValueError(f'Unknown operation {kind!r}.')
            self.trace.append(dict(operation=index+1, kind=kind, stage=op.get('stage'),
                occupied_wells=len(self.wells), destination_wells_consumed=self.well_loads, states=sum(len(p.states) for p in self.wells.values()),
                total_mass=sum(sum(p.states.values()) for p in self.wells.values())))
        return self

    def report(self, targets, expected_read_count=10000):
        targets = binary_matrix(targets)
        if targets.shape[1] != self.payload_bits or expected_read_count < 1:
            raise ValueError('Target width must match payload sites and read count must be positive.')
        states = defaultdict(float)
        for p in self.wells.values():
            for state, mass in p.states.items():
                states[state] += mass
        payloads = defaultdict(float); addresses = defaultdict(set)
        target_set = {tuple(row) for row in targets.tolist()}
        best_addresses = {}
        incomplete = 0.0
        for state, mass in states.items():
            payload, address = state[:self.payload_bits], state[self.payload_bits:]
            if EMPTY in payload:
                incomplete += mass
                continue
            payloads[payload] += mass
            if self.address_bits and EMPTY not in address:
                addresses[address].add(payload)
                if payload not in best_addresses or mass > best_addresses[payload][0]:
                    best_addresses[payload] = (mass, address)
        conflict = sum(len(v) > 1 for v in addresses.values())
        covered = target_set & payloads.keys()
        unique_mixtures = [self.library[k] for k in self.used_mixtures]
        bit_stocks = {(site, value) for mix in unique_mixtures for site, value in enumerate(mix) if value != EMPTY}
        transfers = self.preparation_transfers+self.addition_transfers+self.routing_transfers
        lookup = []
        for index, row in enumerate(targets.tolist()):
            key = tuple(row); found = best_addresses.get(key)
            good = found is not None and len(addresses[found[1]]) == 1
            lookup.append(dict(target_row=index, address=''.join(map(str, found[1])) if good else None,
                payload_fraction=payloads.get(key, 0.0),
                chosen_address_fraction=found[0] if good else 0.0))
        probabilities = [payloads.get(tuple(row), 0) for row in target_set]
        # Union bound under independent draws of molecules; no claim about PAINT biases.
        missing_bound = min(1.0, sum((1-p)**expected_read_count for p in probabilities))
        return dict(requested_files=len(targets), unique_requested_files=len(target_set),
            payload_bits=self.payload_bits, address_bits=self.address_bits,
            covered_unique_files=len(covered), all_files_written=len(covered)==len(target_set),
            all_files_identifiable=all(r['address'] is not None for r in lookup),
            address_conflicts=conflict, generated_unique_payloads=len(payloads),
            extra_payloads=len(payloads.keys()-target_set), incomplete_payload_fraction=incomplete,
            useful_product_fraction=sum(payloads[p] for p in covered),
            minimum_requested_fraction=min(probabilities),
            expected_read_count=expected_read_count, probability_any_file_missing_upper_bound=missing_bound,
            peak_destination_wells=self.peak_wells, destination_wells_consumed=self.well_loads,
            destination_well_loads=self.well_loads, final_occupied_wells=len(self.wells),
            mixture_source_wells=len(self.used_mixtures), elementary_bit_stock_wells=len(bit_stocks),
            mixture_preparation_transfers=self.preparation_transfers,
            mixture_addition_transfers=self.addition_transfers, split_pool_transfers=self.routing_transfers,
            total_transfers=transfers, transfers_with_prepared_library=self.addition_transfers+self.routing_transfers,
            payload_bit_transfer_baseline=int(targets.size),
            meets_peak_well_goal=self.peak_wells < len(targets),
            meets_destination_well_goal=self.well_loads < len(targets),
            meets_single_use_well_goal=self.well_loads < len(targets),
            meets_total_transfer_goal=transfers < targets.size,
            total_population_mass=sum(states.values()), target_address_manifest=lookup,
            assumptions=['Ideal complete irreversible first-write-wins binding; OFF is permanent.',
                'One preparation transfer per active site reagent per unique mixture; one addition per destination.',
                'Every split output and pooled source costs one transfer; no free routing.',
                'Barcode sites are separate writable sites; barcode reagents and additions are counted.',
                'Empty payload sites are not silently interpreted as OFF.',
                'Destination wells are single-use: every split output and new pool consumes a fresh well, including the initial well.',
                'Continued additions to the same resident population are allowed; emptied destination wells are never refilled.',
                'Prepared mixture source wells remain reusable; preparation is charged once per unique used mixture.',
                'No physical volume, stock depletion, cross-talk, dead volume or kinetic model.',
                'Buffer, wash and cleaning transfers are excluded; total_transfers is the modeled total, not an Echo-ready protocol.',
                'Equal splitting unless fractions specified; populations are expectations, not finite molecule counts.',
                'Baseline m*n counts payload-only additions, making the comparison conservative about addresses.',
                'Manifest stores target indices and addresses, not target payload content.'])


def compile_program(targets, block_bits=2, pooling='all', max_pools=2):
    """Constructive heuristic: write blocks, then merge equal or unioned suffix sets.

    Exact suffix merging introduces no unwanted payloads. All/bounded merging can
    introduce cross-products. This is a feasible schedule generator, not an optimal
    scheduler or an exhaustive search over overlapping blocking mixtures.
    """
    targets = binary_matrix(targets)
    if block_bits < 1 or pooling not in ('all', 'exact', 'bounded') or max_pools < 1:
        raise ValueError('Positive block size/pool limit and all, exact, or bounded pooling required.')
    n = targets.shape[1]
    # Each stage needs at most block_bits address bits, even with unequal branching.
    spans = [(a, min(a+block_bits, n)) for a in range(0, n, block_bits)]
    address_width = sum(min(b-a, (len(np.unique(targets[:, a:b], axis=0))-1).bit_length()) for a,b in spans)
    mixtures = {}; mixture_ids = {}; operations = []; next_id = 0
    pools = [('start', {tuple(row) for row in targets.tolist()})]
    address_offset = n
    for stage, (a, b) in enumerate(spans, 1):
        max_choices = len(np.unique(targets[:, a:b], axis=0))
        digit_width = min(b-a, (max_choices-1).bit_length())
        edges = []
        for source, suffixes in pools:
            groups = defaultdict(set)
            for suffix in suffixes:
                groups[suffix[:b-a]].add(suffix[b-a:])
            patterns = sorted(groups)
            if len(patterns) == 1:
                destinations = [source]
            else:
                destinations = []
                for _ in patterns:
                    next_id += 1; destinations.append(f'w{next_id}')
                operations.append(dict(op='split', source=source, outputs=destinations, stage=stage))
            for index, (pattern, destination) in enumerate(zip(patterns, destinations)):
                mix = [EMPTY]*(n+address_width)
                mix[a:b] = pattern
                bits = [(index >> shift)&1 for shift in reversed(range(digit_width))]
                mix[address_offset:address_offset+digit_width] = bits
                key = tuple(mix)
                if key not in mixture_ids:
                    mixture_ids[key] = f'K{len(mixtures)+1}'
                    mixtures[mixture_ids[key]] = mix
                operations.append(dict(op='add', well=destination, mixture=mixture_ids[key], stage=stage))
                edges.append(([destination], groups[pattern]))
        address_offset += digit_width
        # Merge identical continuation languages first (safe even in exact mode).
        merged = {}
        for sources, suffixes in edges:
            key = frozenset(suffixes)
            if key not in merged:
                merged[key] = ([], set(suffixes))
            merged[key][0].extend(sources)
        groups = list(merged.values())
        limit = 1 if pooling == 'all' else max_pools if pooling == 'bounded' else len(groups)
        while len(groups) > limit:
            # Prefer merging the most similar continuation languages.
            _, i, j = min((len(x[1]^y[1]), i, j) for i,x in enumerate(groups) for j,y in enumerate(groups) if i<j)
            groups[i] = (groups[i][0]+groups[j][0], groups[i][1]|groups[j][1])
            groups.pop(j)
        pools = []
        for sources, suffixes in groups:
            if len(sources) == 1:
                destination = sources[0]
            else:
                next_id += 1; destination = f'w{next_id}'
                operations.append(dict(op='pool', sources=sources, destination=destination, stage=stage))
            pools.append((destination, suffixes))
    return dict(payload_bits=n, address_bits=address_width, mixtures=mixtures, operations=operations,
                compiler=dict(block_bits=block_bits, pooling=pooling, max_pools=max_pools),
                stages=len(spans))


def run(targets, program, max_states=100000, reads=10000):
    sim = Simulator(program['payload_bits'], program.get('address_bits', 0), max_states).execute(program)
    report = sim.report(targets, reads)
    report['stages'] = program.get('stages')
    report['compiler'] = program.get('compiler', 'custom schedule')
    return report, sim



def search(targets, block_sizes=(1, 2, 4), pool_limits=(1, 2), max_states=100000, reads=10000):
    """Bounded candidate sweep; returned best is not a global optimum.

    Prefer schedules meeting both cost goals, then useful yield, then transfer cost. Destination wells are single-use.
    State-limit failures remain explicit rather than being approximated as successes.
    """
    targets = binary_matrix(targets)
    trials = []; best = None; best_score = None
    for size in sorted(set(min(targets.shape[1], b) for b in block_sizes)):
        for policy, limit in [('all', 1), ('exact', 1)]+[('bounded', p) for p in pool_limits if p > 1]:
            program = compile_program(targets, size, policy, limit)
            try:
                report, sim = run(targets, program, max_states, reads)
            except ValueError as exc:
                trials.append(dict(block_bits=size, pooling=policy, max_pools=limit, error=str(exc)))
                continue
            trials.append({k:v for k,v in report.items() if k not in ('target_address_manifest', 'assumptions')})
            goals = report['meets_destination_well_goal'] and report['meets_total_transfer_goal']
            score = (report['all_files_identifiable'], goals, report['useful_product_fraction'], -report['total_transfers'])
            if best_score is None or score > best_score:
                best_score = score; best = (program, report, sim)
    if best is None:
        raise ValueError('No candidate could be simulated within the state limit.')
    return best, trials


def plot_run(program, report, sim, path):
    """Standalone inspection graphic; the JSON schedule retains every operation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    steps = [0]+[r['operation'] for r in sim.trace]
    counts = [1]+[r['occupied_wells'] for r in sim.trace]
    consumed = [1]+[r['destination_wells_consumed'] for r in sim.trace]
    ax.step(steps, consumed, where='post', label='Single-use wells consumed')
    ax.step(steps, counts, where='post', alpha=.6, label='Currently occupied')
    ax.axhline(report['peak_destination_wells'], color='orange', linestyle=':', label='Peak during transfers')
    ax.axhline(report['requested_files'], color='gray', linestyle='--', label='One well per requested file')
    ax.set(title='Destination wells', xlabel='Operation', ylabel='Wells'); ax.legend(fontsize=8)
    ax = axes[0, 1]
    names = ['Preparation', 'Additions', 'Split / pool', 'Total', 'm × n']
    values = [report[k] for k in ('mixture_preparation_transfers', 'mixture_addition_transfers',
                                  'split_pool_transfers', 'total_transfers', 'payload_bit_transfer_baseline')]
    bars = ax.bar(names, values)
    ax.bar_label(bars, padding=3); ax.margins(y=.2)
    ax.set(title='Fluidic transfers (includes written barcode)', ylabel='Transfers')
    ax = axes[1, 0]
    keys = list(program['mixtures'])
    im = ax.imshow([program['mixtures'][k] for k in keys], aspect='auto', interpolation='nearest',
                   cmap=ListedColormap(['#eeeeee', '#e89943', '#1aaac5']),
                   norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3))
    if program.get('address_bits'):
        ax.axvline(program['payload_bits']-.5, color='black', linewidth=1)
    stride = max(1, len(keys)//12)
    ax.set_yticks(range(0, len(keys), stride), keys[::stride])
    ax.set(title='Reusable multibit mixtures', xlabel='Payload sites | address sites', ylabel='Mixture')
    cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=.7)
    cb.ax.set_yticklabels(['Untouched', 'OFF', 'ON'])
    ax = axes[1, 1]
    fractions = [v['payload_fraction'] for v in report['target_address_manifest']]
    ax.bar(range(len(fractions)), fractions)
    ax.set(title=f"Requested payload abundance · {report['useful_product_fraction']:.1%} useful product",
           xlabel='Requested file row', ylabel='Fraction of all molecules')
    fig.suptitle(f"Ideal bulk writing · {report['covered_unique_files']}/{report['unique_requested_files']} files covered"
                 f" · {report['extra_payloads']} extra payloads\n"
                 'First write wins; full binding; no kinetic or volume model')
    fig.savefig(path, dpi=140); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true')
    source.add_argument('--input', type=Path, help='Binary CSV/.npy, one requested file per row')
    parser.add_argument('--program', type=Path, help='Run an explicit schedule JSON instead of compiling')
    parser.add_argument('--search', action='store_true', help='Sweep block widths 1, 2, 4 and pooling policies')
    parser.add_argument('--block-bits', type=int, default=2)
    parser.add_argument('--pooling', choices=('all', 'exact', 'bounded'), default='bounded')
    parser.add_argument('--max-pools', type=int, default=2)
    parser.add_argument('--max-states', type=int, default=100000)
    parser.add_argument('--reads', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.program and args.search:
        parser.error('--program and --search cannot be combined.')
    if args.demo:
        rng = np.random.default_rng(17)
        codes = np.array([[(i >> j)&1 for j in range(4)] for i in range(16)], dtype=np.uint8)
        datasets = [('structured', np.repeat(codes, 4, axis=1)),
                    ('random', rng.integers(0, 2, (16, 12), dtype=np.uint8))]
    else:
        values = np.load(args.input, allow_pickle=False) if args.input.suffix == '.npy' else np.loadtxt(args.input, delimiter=',', ndmin=2)
        datasets = [('input', binary_matrix(values))]
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, targets in datasets:
        np.savetxt(args.output/f'{name}_targets.csv', targets, fmt='%d', delimiter=',')
        if args.search:
            (program, report, sim), trials = search(targets, max_states=args.max_states, reads=args.reads)
            label = name+'_best'
            (args.output/f'{label}_schedule.json').write_text(json.dumps(program, indent=2)+'\n')
            (args.output/f'{label}_search.json').write_text(json.dumps(trials, indent=2)+'\n')
            (args.output/f'{label}_trace.json').write_text(json.dumps(sim.trace, indent=2)+'\n')
            plot_run(program, report, sim, args.output/f'{label}.png')
            reports[label] = report
            continue
        policies = ('all', 'bounded', 'exact') if args.demo and not args.program else (args.pooling,)
        for policy in policies:
            label = name+'_'+policy
            program = json.loads(args.program.read_text()) if args.program else compile_program(targets, args.block_bits, policy, args.max_pools)
            (args.output/f'{label}_schedule.json').write_text(json.dumps(program, indent=2)+'\n')
            try:
                report, sim = run(targets, program, args.max_states, args.reads)
                (args.output/f'{label}_trace.json').write_text(json.dumps(sim.trace, indent=2)+'\n')
                plot_run(program, report, sim, args.output/f'{label}.png')
            except ValueError as exc:
                report = dict(error=str(exc))
            reports[label] = report
    (args.output/'report.json').write_text(json.dumps(reports, indent=2)+'\n')
    for name, r in reports.items():
        if 'error' in r:
            print(name, r['error']); continue
        print(f"{name}: {r['covered_unique_files']}/{r['unique_requested_files']} covered, "
              f"{r['destination_wells_consumed']} single-use destination wells, {r['total_transfers']} transfers, "
              f"{r['useful_product_fraction']:.1%} useful, goals={r['meets_destination_well_goal'] and r['meets_total_transfer_goal']}")


if __name__ == '__main__':
    main()

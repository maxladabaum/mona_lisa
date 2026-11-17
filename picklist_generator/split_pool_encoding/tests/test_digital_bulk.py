import numpy as np
import pytest
from split_pool_encoding.digital_bulk import Simulator, compile_program, run


def test_first_write_wins_and_permanent_off():
    sim = Simulator(3)
    sim.library = {'a': (0, 1, -1), 'b': (1, 0, 1)}
    sim.add('start', 'a'); sim.add('start', 'b')
    assert sim.wells['start'].states == {(0, 1, 1): 1.0}
    assert sim.preparation_transfers == 5
    assert sim.addition_transfers == 2


def test_blocking_computes_and_across_pooled_histories():
    program = dict(payload_bits=1, address_bits=2, mixtures={
        'early_off': [0, 0, -1], 'early_open': [-1, 1, -1],
        'late_off': [0, -1, 0], 'late_on': [1, -1, 1]}, operations=[
        dict(op='split', source='start', outputs=['a', 'b'], fractions=[.25, .75]),
        dict(op='add', well='a', mixture='early_off'),
        dict(op='add', well='b', mixture='early_open'),
        dict(op='pool', sources=['a', 'b'], destination='p'),
        dict(op='split', source='p', outputs=['c', 'd']),
        dict(op='add', well='c', mixture='late_off'),
        dict(op='add', well='d', mixture='late_on'),
        dict(op='pool', sources=['c', 'd'], destination='final')])
    report, sim = run([[0], [1]], program)
    states = sim.wells['final'].states
    assert states == {(0,0,0): .125, (0,1,0): .375, (0,0,1): .125, (1,1,1): .375}
    assert report['all_files_identifiable']
    assert report['minimum_requested_fraction'] == .375
    assert report['total_transfers'] == 7+4+8
    assert report['peak_destination_wells'] == 3
    assert all(t['total_mass'] == 1 for t in sim.trace)


@pytest.mark.parametrize('policy', ['all', 'bounded', 'exact'])
@pytest.mark.parametrize('block', [1, 2, 3, 6])
def test_compiler_covers_targets_and_writes_unique_addresses(policy, block):
    targets = np.random.default_rng(10).integers(0, 2, (9, 6), dtype=np.uint8)
    program = compile_program(targets, block, policy, 2)
    report, sim = run(targets, program)
    assert report['all_files_written']
    assert report['all_files_identifiable']
    assert report['address_conflicts'] == 0
    assert report['incomplete_payload_fraction'] == 0
    assert report['total_population_mass'] == pytest.approx(1)
    assert all(t['total_mass'] == pytest.approx(1) for t in sim.trace)
    if policy == 'exact':
        assert report['extra_payloads'] == 0
        assert report['useful_product_fraction'] == pytest.approx(1)


def test_extra_products_from_full_pooling():
    targets = [[0, 0], [1, 1]]
    all_report, _ = run(targets, compile_program(targets, 1, 'all'))
    exact_report, _ = run(targets, compile_program(targets, 1, 'exact'))
    assert all_report['extra_payloads'] == 2
    assert all_report['useful_product_fraction'] == .5
    assert exact_report['extra_payloads'] == 0


def test_prepared_mixture_reuse_charged_once():
    sim = Simulator(1)
    sim.library = {'a': (1,)}
    sim.split('start', ['a', 'b'])
    sim.add('a', 'a'); sim.add('b', 'a')
    assert sim.preparation_transfers == 1
    assert sim.addition_transfers == 2


def test_address_collision_and_unwritten_sites_are_not_success():
    sim = Simulator(1, 1)
    sim.library = {'a': (1, 0), 'b': (0, 0)}
    sim.split('start', ['a', 'b'])
    sim.add('a', 'a'); sim.add('b', 'b')
    r = sim.report([[0], [1]])
    assert r['all_files_written'] and not r['all_files_identifiable']
    assert r['address_conflicts'] == 1
    r = Simulator(1).report([[0]])
    assert not r['all_files_written']
    assert r['incomplete_payload_fraction'] == 1


def test_state_limit_and_bad_splits_are_atomic():
    sim = Simulator(1, max_states=2)
    with pytest.raises(ValueError):
        sim.split('start', ['a', 'b'], [.2, .2])
    with pytest.raises(ValueError):
        sim.split('start', ['a', 'b', 'c'])
    assert list(sim.wells) == ['start']
    assert sim.routing_transfers == 0


def test_goals_include_mixture_preparation():
    r, _ = run([[0, 0], [1, 1]], compile_program([[0, 0], [1, 1]], 2))
    assert r['total_transfers'] == r['mixture_preparation_transfers']+r['mixture_addition_transfers']+r['split_pool_transfers']
    assert r['mixture_preparation_transfers'] == 6  # two two-bit payloads plus two address reagents
    assert not r['meets_total_transfer_goal']


def test_search_selects_real_verified_schedule_and_emits_plot(tmp_path):
    from split_pool_encoding.digital_bulk import search, plot_run
    targets = np.array([[int(b) for b in f'{i:04b}'] for i in range(16)], dtype=np.uint8)
    targets = np.repeat(targets, 4, axis=1)
    (program, report, sim), trials = search(targets)
    assert len(trials) == 9
    assert report['all_files_identifiable']
    assert report['meets_destination_well_goal']
    assert report['meets_total_transfer_goal']
    assert report['useful_product_fraction'] == 1
    path = tmp_path/'plot.png'
    plot_run(program, report, sim, path)
    assert path.stat().st_size > 1000


def test_destination_wells_are_never_reused_but_sources_are_reusable():
    sim = Simulator(1)
    sim.library = {'mix': (1,)}
    sim.split('start', ['a', 'b'])
    sim.add('a', 'mix'); sim.add('b', 'mix')
    sim.pool(['a', 'b'], 'c')
    assert sim.wells['c'].slot == 3
    assert sim.well_loads == 4
    assert sim.peak_wells == 3
    with pytest.raises(ValueError, match='never previously used'):
        sim.split('c', ['a', 'd'])
    with pytest.raises(ValueError, match='never previously used'):
        sim.pool(['c'], 'start')
    assert sim.well_loads == 4
    sim.split('c', ['d', 'e'])
    sim.add('d', 'mix'); sim.add('e', 'mix')
    assert {sim.wells['d'].slot, sim.wells['e'].slot} == {4, 5}
    assert sim.preparation_transfers == 1
    assert sim.addition_transfers == 4
    report = sim.report([[1]]*5)
    assert report['meets_peak_well_goal']
    assert not report['meets_destination_well_goal']
    assert report['destination_wells_consumed'] == 6
    assert 'reusable_destination_slots' not in report

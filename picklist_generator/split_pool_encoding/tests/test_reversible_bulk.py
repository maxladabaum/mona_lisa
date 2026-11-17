import json
import numpy as np
import pytest
from split_pool_encoding.reversible_bulk import apply, pack_circuit, unpack_circuit, search_encoding, export, cost


def universe(width):
    return ((np.arange(2**width)[:, None] >> np.arange(width)) & 1).astype(np.uint8)


def test_all_gates_are_bijective_and_inverse_needs_no_lookup_table():
    values = universe(4)
    gates = [dict(op='not', sites=[1]), dict(op='swap', sites=[0, 2]),
             dict(op='cnot', sites=[2, 3]), dict(op='toffoli', sites=[0, 1, 3])]
    encoded = apply(values, gates)
    assert len(np.unique(encoded, axis=0)) == 16
    np.testing.assert_array_equal(apply(encoded, gates, inverse=True), values)
    assert not np.array_equal(encoded, values)
    width, loaded = unpack_circuit(pack_circuit(4, gates))
    assert width == 4 and loaded == gates
    np.testing.assert_array_equal(apply(encoded, loaded, inverse=True), values)


def test_toffoli_truth_table():
    values = universe(3)
    actual = apply(values, [dict(op='toffoli', sites=[0, 1, 2])])
    expected = values.copy()
    expected[:, 2] ^= values[:, 0] & values[:, 1]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('gate', [dict(op='bad', sites=[0]), dict(op='swap', sites=[0,0]),
                                  dict(op='cnot', sites=[0,5]), dict(op='not', sites=[-1])])
def test_bad_gates_rejected(gate):
    with pytest.raises(ValueError):
        apply(universe(3), [gate])


def test_circuit_bytes_include_header_padding_and_reject_corruption():
    gates = [dict(op='swap', sites=[0,12]), dict(op='cnot', sites=[15,8])]
    blob = pack_circuit(16, gates)
    assert len(blob) == 11  # 8-byte header + 20 instruction bits padded to 3 bytes
    assert unpack_circuit(blob) == (16, gates)
    for bad in (blob[:-1], blob+b'\x00', b'BAD!'+blob[4:]):
        with pytest.raises(ValueError):
            unpack_circuit(bad)


def test_search_respects_budget_and_non_worsening_costs(tmp_path):
    values = np.random.default_rng(17).integers(0, 2, (16, 8), dtype=np.uint8)
    result = search_encoding(values, evaluations=12, max_gates=3, decoder_bytes=12, max_states=512)
    again = search_encoding(values, evaluations=12, max_gates=3, decoder_bytes=12, max_states=512)
    assert result['gates'] == again['gates']
    assert len(result['history']) == 12
    assert all(r['decoder_bytes'] <= 12 for r in result['history'])
    assert len(result['gates']) <= 3
    before, after = result['baseline'][1], result['best'][1]
    for key in ('destination_wells_consumed', 'total_transfers'):
        assert after[key] <= before[key]
    assert after['all_files_identifiable'] and after['useful_product_fraction'] >= .01
    summary = export(values, result, tmp_path/'export')
    assert summary['exact_simulated_payload_round_trip']
    assert summary['original_bits'] == summary['encoded_bits'] == values.size
    assert summary['decoder_bytes'] == (tmp_path/'export/decoder.rbc').stat().st_size


def test_identity_control_and_too_small_budget():
    values = universe(4)
    result = search_encoding(values, evaluations=0, decoder_bytes=8)
    assert result['gates'] == [] and result['history'] == []
    with pytest.raises(ValueError, match='header'):
        search_encoding(values, decoder_bytes=7)

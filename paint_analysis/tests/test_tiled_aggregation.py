from copy import deepcopy
from dataclasses import fields
import gc
import weakref

import numpy as np
import pytest

from analysis_session import load_analysis_session, save_analysis_session
from origami_analysis import concatenate_origami_pick_results
from tiled_aggregation import StreamingTileResults, SEQUENCE_PARAMS, _Rows
from test_tile_checkpoint import run_fixture


def equal(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right or (isinstance(left, float) and np.isnan(left) and np.isnan(right))


def test_streamed_results_match_in_memory_merge_and_save_load(tmp_path):
    live, _, payload = run_fixture()
    second = deepcopy(payload)
    for item in second['templates']:
        item['picks'].density_extent_nm = (10., 200., -100., 40.)
        item['picks'].alignment_candidate_images *= 2
    writer = StreamingTileResults(tmp_path / 'cache', multi=True)
    writer.append(payload)
    writer.append(second)
    actual = writer.finish()
    np.testing.assert_equal(actual['counts'], payload['counts'] * 2)
    assert actual['unclassified_count'] == 24
    for index, item in enumerate(actual['templates']):
        expected = concatenate_origami_pick_results([payload['templates'][index]['picks'], second['templates'][index]['picks']])
        for field in fields(expected):
            equal(getattr(expected, field.name), getattr(item['picks'], field.name))
        assert isinstance(item['picks'].alignment_candidate_images, np.memmap)
        assert not item['picks'].alignment_candidate_images.flags.writeable
        for key in SEQUENCE_PARAMS:
            if key in payload['templates'][index]['params']:
                expected_values = tuple(payload['templates'][index]['params'][key]) + tuple(second['templates'][index]['params'][key])
                equal(expected_values, item['params'][key])
    path = tmp_path / 'combined.paintanalysis'
    save_analysis_session(path, actual)
    reopened = load_analysis_session(path, live._analysis_record_types())
    for a, b in zip(actual['templates'], reopened['templates']):
        for field in fields(a['picks']):
            equal(getattr(a['picks'], field.name), getattr(b['picks'], field.name))
        equal(a['params'], b['params'])


def test_append_releases_tile_arrays_and_preserves_nested_metadata(tmp_path):
    _, _, original = run_fixture()
    payload = deepcopy(original)
    nested = np.arange(400, dtype=float).reshape(200, 2)
    payload['templates'][0]['params']['nested_example'] = {'values': nested}
    payload['templates'][0]['params']['object_example'] = np.array([{'label': 'open'}, None], dtype=object)
    refs = [weakref.ref(nested)]
    for item in payload['templates']:
        refs.extend(weakref.ref(getattr(item['picks'], name)) for name in ('alignment_candidate_images', 'site_counts') if hasattr(item['picks'], name))
        refs.extend(weakref.ref(region) for region in item['picks'].regions)
    writer = StreamingTileResults(tmp_path, multi=True)
    writer.append(payload)
    del payload, nested, item
    gc.collect()
    assert all(ref() is None for ref in refs), 'Writer retained arrays from an earlier tile'
    restored = writer.finish()['templates'][0]
    np.testing.assert_equal(restored['params']['nested_example']['values'], np.arange(400).reshape(200, 2))
    np.testing.assert_equal(restored['params']['object_example'], np.array([{'label': 'open'}, None], dtype=object))
    assert not restored['picks'].regions[0].flags.writeable


def test_worker_disk_failure_preserves_checkpoints_and_can_rebuild(tmp_path, monkeypatch):
    from unittest.mock import Mock
    live, run, payload = run_fixture()
    live._identify_origami_worker = Mock(return_value=('origami_multi_picks', deepcopy(payload)))
    live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    before = {p.name: p.read_bytes() for p in tmp_path.glob('*.paintanalysis')}
    with monkeypatch.context() as patch:
        patch.setattr(_Rows, 'append', Mock(side_effect=OSError(28, 'disk full')))
        with pytest.raises(OSError, match='disk full'):
            live._checkpointed_tiled_worker(tmp_path, run, completed_only=True)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob('*.paintanalysis')}
    live._identify_origami_worker = Mock(side_effect=AssertionError('Recomputed saved tile'))
    _, result = live._checkpointed_tiled_worker(tmp_path, run, completed_only=True)
    np.testing.assert_equal(result['counts'], [24, 24])
    assert isinstance(result['points_nm'], np.memmap)


def test_disk_backed_results_render_all_classification_views(tmp_path, monkeypatch):
    from test_analysis_view_restore import matrix
    live, _, payload = run_fixture()
    writer = StreamingTileResults(tmp_path, multi=True)
    writer.append(payload)
    combined = writer.finish()
    live.origami_multi_template_results = {
        item['name']: {'picks': item['picks'], 'params': item['params']}
        for item in combined['templates']
    }
    live.origami_multi_template_overlay_results = {}
    # Exercise view builders against read-only, disk-backed arrays (including
    # density overlays, galleries and per-particle diagnostics for every class).
    snapshots = matrix(live, monkeypatch)
    assert snapshots


def test_small_array_writes_are_batched_per_tile(tmp_path, monkeypatch):
    from pathlib import Path
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    original_open = Path.open
    opens = []

    def track_open(path, *args, **kwargs):
        if path == store.path:
            opens.append(args)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', track_open)
    values = [np.arange(i % 9, dtype=np.int32) for i in range(500)]
    values += [np.arange(24, dtype=float).reshape(4, 6).T, np.array(3.5), np.empty((3, 0))]
    with store.writing():
        saved = store.put(values, {})
    assert store.writer is None
    assert len(opens) == 1  # Hundreds of small arrays use one buffered stream.
    for expected, actual in zip(values, store.get(saved)):
        np.testing.assert_array_equal(actual, expected)


def test_array_writer_closes_after_failed_tile(tmp_path):
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    with pytest.raises(RuntimeError, match='interrupted'):
        with store.writing():
            stream = store.writer
            store.put(np.arange(4), {})
            raise RuntimeError('interrupted')
    assert stream.closed
    assert store.writer is None


@pytest.mark.parametrize('limit', [2, 8])
def test_result_writers_reuse_handles_and_bound_open_files(tmp_path, monkeypatch, limit):
    from pathlib import Path
    from tiled_aggregation import _Rows, _WriterPool
    pool = _WriterPool()
    pool.limit = limit
    original_open = Path.open
    opens = []

    def track_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.suffix == '.bin':
            opens.append(handle)
        return handle

    monkeypatch.setattr(Path, 'open', track_open)
    rows = [_Rows(tmp_path / f'{i}.bin', pool) for i in range(3)]
    for tile in range(3):
        for row in rows:
            row.append(np.array([tile, tile + 10]))
            assert len(pool.handles) <= limit
    for row in rows:
        np.testing.assert_array_equal(row.finish(), [0, 10, 1, 11, 2, 12])
    assert len(opens) == (3 if limit == 8 else 9)
    assert all(handle.closed for handle in opens)
    assert not pool.handles


def test_streaming_keeps_orientation_diagnostics_aligned_with_legacy_tiles(tmp_path):
    live, _, payload = run_fixture()
    second = deepcopy(payload)
    for item in second['templates']:
        count = len(item['picks'].regions)
        item['picks'].orientation_scores = np.tile([.2, .9], (count, 1))
        item['picks'].orientation_correlations = np.tile([.1, .8], (count, 1))
        item['picks'].orientation_flipped = np.ones(count, dtype=bool)
        item['picks'].orientation_ambiguous = np.zeros(count, dtype=bool)
    writer = StreamingTileResults(tmp_path, multi=True)
    writer.append(payload)
    writer.append(second)
    merged = writer.finish()
    path = tmp_path / 'orientation.paintanalysis'
    save_analysis_session(path, merged)
    restored = load_analysis_session(path, live._analysis_record_types())
    for index, item in enumerate(restored['templates']):
        count = len(payload['templates'][index]['picks'].regions)
        picks = item['picks']
        assert picks.orientation_scores.shape == (2 * count, 2)
        assert np.isnan(picks.orientation_scores[:count]).all()
        np.testing.assert_equal(picks.orientation_scores[count:], np.tile([.2, .9], (count, 1)))
        assert not picks.orientation_flipped[:count].any()
        assert picks.orientation_flipped[count:].all()


def test_shared_alignment_written_once_for_120_templates(tmp_path):
    from dataclasses import replace
    _, _, payload = run_fixture()
    original = payload['templates'][0]
    payload['templates'] = [dict(original, name=str(i), picks=replace(original['picks'])) for i in range(120)]
    payload['counts'] = np.ones(120, dtype=int)
    with StreamingTileResults(tmp_path, multi=True) as writer:
        writer.append(payload)
        # No per-template row files are opened while processing tiles.
        assert not list(writer.directory.glob('*/*.bin'))
        merged = writer.finish()
        images = [item['picks'].alignment_candidate_images for item in merged['templates']]
        assert all(image is images[0] for image in images)
        assert len(list(writer.directory.glob('*/alignment_candidate_images.bin'))) == 1
        np.testing.assert_equal(images[0], original['picks'].alignment_candidate_images)


def test_shared_rows_can_diverge_then_share_again_without_losing_history(tmp_path):
    from tiled_aggregation import _ArrayStore, _SharedRows
    store = _ArrayStore(tmp_path)
    finished = {}
    a, b = [_SharedRows(tmp_path / f'{name}.bin', store, finished) for name in ('a', 'b')]
    for left, right in [(np.array([1., 2.]), None), (np.array([3.]), np.array([4.])), (np.array([5.]), None)]:
        with store.writing():
            memo = {}
            a.append(left, memo)
            b.append(left if right is None else right, memo)
    np.testing.assert_equal(a.finish(), [1, 2, 3, 5])
    np.testing.assert_equal(b.finish(), [1, 2, 4, 5])
    assert a.finish() is not b.finish()


def test_temporary_arrays_kept_alive_only_during_tile_write(tmp_path):
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    with store.writing():
        value = np.arange(10)
        ref = weakref.ref(value)
        saved = store.put(value, {})
        del value
        gc.collect()
        assert ref() is not None
    gc.collect()
    assert ref() is None
    np.testing.assert_equal(store.get(saved), np.arange(10))


def test_shared_parameter_views_use_same_bytes_without_merging_distinct_views(tmp_path):
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    base = np.arange(20).reshape(4, 5)
    with store.writing():
        memo = {}
        first = store.put(base[0], memo)
        again = store.put(base[0], memo)
        other = store.put(base[1], memo)
        reverse = store.put(base[0, ::-1], memo)
    assert first is again
    assert first is not other and first is not reverse
    np.testing.assert_equal(store.get(first), base[0])
    np.testing.assert_equal(store.get(other), base[1])
    np.testing.assert_equal(store.get(reverse), base[0, ::-1])


def test_source_table_gathers_in_order_without_wide_dataframe_take(tmp_path, monkeypatch):
    import pandas as pd
    table = pd.DataFrame({'x': np.arange(250_010, dtype=float),
                          'frame': np.arange(250_010, dtype=np.int32)})
    table.index = pd.Index(np.arange(len(table)) + 50, name='source_row')
    chunks = [np.arange(220_000, 250_010), np.arange(0, 210_000)]
    expected = table.iloc[np.concatenate(chunks)]
    def fail(*args, **kwargs):
        raise AssertionError('Full DataFrame row gather allocates all columns')
    monkeypatch.setattr(pd.DataFrame, 'take', fail)
    updates = []
    with StreamingTileResults(tmp_path, multi=False) as writer:
        actual = writer.source_table(table, chunks, lambda done, total: updates.append((done, total)))
    pd.testing.assert_frame_equal(actual, expected)
    assert not actual['x'].to_numpy().flags.writeable
    assert updates[-1] == (3, 3)


def test_source_table_preserves_nullable_and_text_columns(tmp_path):
    import pandas as pd
    table = pd.DataFrame({'x': [1., 2., 3.], 'label': ['a', None, 'c'],
                          'frame': pd.array([1, None, 3], dtype='Int64')})
    with StreamingTileResults(tmp_path, multi=False) as writer:
        actual = writer.source_table(table, [np.array([2, 0, 1])])
    pd.testing.assert_frame_equal(actual, table.iloc[[2, 0, 1]])


def test_large_shared_row_copy_reports_before_and_during_writes(tmp_path, monkeypatch):
    from tiled_aggregation import _ArrayStore, _SharedRows
    store = _ArrayStore(tmp_path)
    rows = _SharedRows(tmp_path / 'large.bin', store, {})
    data = np.arange(1_200_000, dtype=np.float64).reshape(-1, 2)
    with store.writing():
        rows.append(data, {})
    updates, sizes = [], []
    original = _Rows.append
    def append(self, value):
        assert updates, 'First disk copy must have a visible status'
        sizes.append(value.nbytes)
        original(self, value)
    monkeypatch.setattr(_Rows, 'append', append)
    actual = rows.finish(progress=lambda done, total: updates.append((done, total)))
    np.testing.assert_equal(actual, data)
    assert max(sizes) <= 4 * 1024 * 1024
    assert len(sizes) >= 3
    assert updates[0] == (0, data.nbytes)
    assert updates[-1] == (data.nbytes, data.nbytes)


def test_first_template_reports_detail_before_completion(tmp_path):
    _, _, payload = run_fixture()
    events = []
    with StreamingTileResults(tmp_path, multi=True) as writer:
        writer.append(payload)
        writer.finish(progress=lambda done, total: events.append(('complete', done)),
                      status=lambda current, total, message: events.append(('detail', current, message)))
    first_complete = next(i for i, event in enumerate(events) if event[0] == 'complete')
    assert any(event[0] == 'detail' and 'Copying' in event[2] for event in events[:first_complete])


@pytest.mark.parametrize('order', ['empty_first', 'empty_last', 'empty_middle', 'all_empty'])
def test_no_detection_tiles_do_not_constrain_candidate_image_shape(tmp_path, order):
    from tiled_aggregation import _Picks
    _, _, populated = run_fixture()
    empty = deepcopy(populated)
    for item in empty['templates']:
        picks = item['picks']
        for f in fields(picks):
            value = getattr(picks, f.name)
            if f.name in _Picks._lists:
                setattr(picks, f.name, [])
            elif isinstance(value, np.ndarray) and f.name not in _Picks._static_arrays | _Picks._density:
                setattr(picks, f.name, value[:0])
        picks.alignment_candidate_images = np.empty((0,), dtype=np.float32)
        picks.alignment_reference_image = np.empty((0, 0))
        item['params'] = {}
    empty['counts'] = np.zeros_like(populated['counts'])
    empty['unclassified_count'] = 0
    empty['unclassified_centers_nm'] = np.empty((0, 2))
    sequences = dict(empty_first=[empty,populated], empty_last=[populated,empty],
                     empty_middle=[populated,empty,populated], all_empty=[empty,empty])
    with StreamingTileResults(tmp_path, multi=True) as writer:
        for payload in sequences[order]: writer.append(payload)
        result = writer.finish()
    repetitions = 2 if order == 'empty_middle' else (0 if order == 'all_empty' else 1)
    for original, actual in zip(populated['templates'], result['templates']):
        a, b = original['picks'], actual['picks']
        assert len(b.regions) == repetitions * len(a.regions)
        if repetitions:
            np.testing.assert_equal(b.alignment_candidate_images,
                                    np.concatenate([a.alignment_candidate_images]*repetitions))
            np.testing.assert_equal(b.alignment_reference_image, a.alignment_reference_image)
        else:
            assert b.alignment_candidate_images.shape == (0,)


def test_nonempty_incompatible_rows_still_fail_with_field_name(tmp_path):
    from tiled_aggregation import _ArrayStore, _SharedRows
    store = _ArrayStore(tmp_path)
    rows = _SharedRows(tmp_path/'alignment_candidate_images.bin', store, {})
    with store.writing():
        rows.append(np.zeros((2, 8, 8), dtype=np.float32), {})
        with pytest.raises(ValueError, match='alignment_candidate_images.*expected rows'):
            rows.append(np.zeros((1, 9, 8), dtype=np.float32), {})


def test_disk_tile_index_preserves_edges_invalid_rows_and_source_order(tmp_path, monkeypatch):
    import pandas as pd
    lattice = [(x0, x1, y0, y1) for y0, y1 in [(0, 2), (2, 3)]
               for x0, x1 in [(0, 2), (2, 5), (5, 6)]]
    rng = np.random.default_rng(30)
    points = np.vstack([rng.uniform(-1, 7, (1000, 2)),
                        [[0, 0], [2, 2], [5, 2], [6, 2], [2, 3], [np.nan, 1], [1, np.inf]]])
    table = pd.DataFrame(points / 2, columns=['x', 'y'])
    original = np.argsort
    def bounded_sort(values, *args, **kwargs):
        assert len(values) <= 37
        return original(values, *args, **kwargs)
    monkeypatch.setattr(np, 'argsort', bounded_sort)
    updates = []
    with StreamingTileResults(tmp_path, multi=False) as writer:
        indices, offsets = writer.tile_index(table, 2, lattice,
            lambda done, total: updates.append((done, total)), chunk_size=37)
    assert isinstance(indices, np.memmap)
    assert not indices.flags.writeable
    for i, (x0, x1, y0, y1) in enumerate(lattice):
        expected = np.flatnonzero((points[:, 0] >= x0) & (points[:, 0] < x1)
                                 & (points[:, 1] >= y0) & (points[:, 1] < y1))
        np.testing.assert_array_equal(indices[offsets[i]:offsets[i + 1]], expected)
    assert updates[-1] == (2 * len(table), 2 * len(table))


def test_disk_tile_index_all_points_outside_lattice(tmp_path):
    import pandas as pd
    with StreamingTileResults(tmp_path, multi=False) as writer:
        indices, offsets = writer.tile_index(pd.DataFrame({'x': [-1., np.nan], 'y': [0., 0.]}),
                                             1, [(0, 1, 0, 1)], chunk_size=1)
    assert len(indices) == 0
    np.testing.assert_array_equal(offsets, [0, 0])


def test_tile_index_only_maps_completed_file_read_only_and_cleans_buckets(tmp_path, monkeypatch):
    import pandas as pd
    original = np.memmap
    modes = []
    def readonly_map(*args, **kwargs):
        modes.append(kwargs.get('mode'))
        assert kwargs.get('mode') == 'r', 'Index construction must not scatter writes through mmap'
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'memmap', readonly_map)
    # Exceed the per-tile buffer so rows cross append boundaries.
    table = pd.DataFrame({'x': np.tile([0.5, 1.5], 100_000), 'y': 0.5})
    with StreamingTileResults(tmp_path, multi=False) as writer:
        indices, offsets = writer.tile_index(table, 1, [(0, 1, 0, 1), (1, 2, 0, 1)])
        assert not list(writer.directory.glob('tile-index-*'))
        assert (writer.directory / 'tile-source-index.bin').stat().st_size == len(table) * 8
    assert modes == ['r']
    np.testing.assert_array_equal(offsets, [0, 100_000, 200_000])
    np.testing.assert_array_equal(indices[:100_000], np.arange(0, len(table), 2))
    np.testing.assert_array_equal(indices[100_000:], np.arange(1, len(table), 2))


def test_shared_nested_metadata_is_stored_and_restored_once(tmp_path):
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    shared = {'probabilities': [[0.2, 0.8], [0.9, 0.1]], 'points': np.arange(20)}
    distinct = {'probabilities': [[0.2, 0.8], [0.9, 0.1]]}
    with store.writing():
        memo = {}
        stored = store.put([shared, shared, distinct], memo)
    assert stored[0] is stored[1]
    assert stored[0] is not stored[2]
    decoded = store.get(stored)
    assert decoded[0] is decoded[1]
    assert decoded[0] is not decoded[2]
    equal(decoded, [shared, shared, distinct])


def test_shared_classification_vectors_stay_shared_across_templates_and_tiles(tmp_path):
    _, _, payload = run_fixture()
    vectors = [[0.1, 0.9], [0.8, 0.2]]
    for item in payload['templates']:
        item['params']['classification_template_probability_vectors'] = vectors
    with StreamingTileResults(tmp_path, multi=True) as writer:
        writer.append(payload)
        # Simulate an independently decoded tile with its own vector values.
        other = deepcopy(payload)
        other['templates'][0]['params']['classification_template_probability_vectors'][0][0] = 0.3
        writer.append(other)
        result = writer.finish()
    a, b = [item['params']['classification_template_probability_vectors'] for item in result['templates']]
    assert a is b
    assert isinstance(a, np.memmap)
    assert not np.shares_memory(a[0], a[2])
    assert a[0][0] == 0.1 and a[2][0] == 0.3


def test_dense_probability_metadata_finalizes_without_copying_vector_contents(tmp_path):
    import tracemalloc
    from tiled_aggregation import _ArrayStore
    store = _ArrayStore(tmp_path)
    vectors = tuple(tuple(float(j) / 120 for j in range(120)) for _ in range(4000))
    with store.writing():
        stored = store.put(vectors, {})
    tracemalloc.start()
    try:
        decoded = store.get(stored)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert decoded == vectors
    assert type(decoded) is tuple
    assert all(a is b for a, b in zip(decoded, vectors))
    # Rebuilding 4000 x 120 tuples takes >4 MB, before the decode memo.
    assert peak < 512_000, f'Finalization copied dense metadata: {peak} bytes'


def test_sequence_histories_share_only_when_every_tile_matches(tmp_path):
    _, _, payload = run_fixture()
    key = 'classification_template_probability_vectors'
    shared = ((.1, .9), (.8, .2))
    for item in payload['templates']:
        item['params'][key] = shared
    with StreamingTileResults(tmp_path, multi=True) as writer:
        writer.append(payload)
        writer.append(payload)
        result = writer.finish()
        a, b = [item['params'][key] for item in result['templates']]
        assert a is b
        np.testing.assert_array_equal(a, shared + shared)

    with StreamingTileResults(tmp_path, multi=True) as writer:
        writer.append(payload)
        other = deepcopy(payload)
        other['templates'][1]['params'][key] = ((.3, .7), (.8, .2))
        writer.append(other)
        writer.append(payload)
        result = writer.finish()
        a, b = [item['params'][key] for item in result['templates']]
        assert a is not b
        np.testing.assert_array_equal(a, shared * 3)
        np.testing.assert_array_equal(b, shared + ((.3, .7), (.8, .2)) + shared)


def test_equal_independent_probability_matrices_are_stored_once(tmp_path):
    from dataclasses import replace
    _, _, payload = run_fixture()
    original = payload['templates'][0]
    key = 'classification_template_probability_vectors'
    def matrix():
        return tuple(tuple(float(j) / 120 for j in range(120)) for _ in range(50))
    payload['templates'] = [dict(name=str(i), picks=replace(original['picks']),
                                params={key: matrix()}) for i in range(120)]
    payload['counts'] = np.ones(120, dtype=int)
    with StreamingTileResults(tmp_path, multi=True) as writer:
        writer.append(payload)
        refs = [t['numeric_sequences'][key].chunks[0] for t in writer.templates]
        assert all(ref is refs[0] for ref in refs)
        result = writer.finish()
        matrices = [t['params'][key] for t in result['templates']]
        assert all(a is matrices[0] for a in matrices)
        assert isinstance(matrices[0], np.memmap)
        assert not matrices[0].flags.writeable
        np.testing.assert_array_equal(matrices[0], matrix())
        path = tmp_path / 'roundtrip.paintanalysis'
        save_analysis_session(path, {'a': matrices[0], 'b': matrices[-1]})
        loaded = load_analysis_session(path, {}, array_cache_dir=tmp_path / 'cache')
        assert loaded['a'] is loaded['b']
        np.testing.assert_array_equal(loaded['a'], matrix())


def test_partial_tile_index_writes_only_requested_source_rows(tmp_path):
    import pandas as pd
    table = pd.DataFrame({'x': [0.5, 1.5, 2.5, 1.0, np.nan, -1.0],
                          'y': [0.5] * 6})
    lattice = [(0, 1, 0, 1), (1, 2, 0, 1), (2, 3, 0, 1)]
    with StreamingTileResults(tmp_path, multi=False) as writer:
        indices, offsets = writer.tile_index(table, 1, lattice, chunk_size=2, included_tiles=[1])
        np.testing.assert_array_equal(indices, [1, 3])
        np.testing.assert_array_equal(offsets, [0, 0, 2, 2])
        assert (writer.directory / 'tile-source-index.bin').stat().st_size == 2 * 8

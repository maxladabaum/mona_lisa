from copy import deepcopy
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image
from split_pool_selection import selected_design, highlight_image, draw_selected_origami, draw_evolution
from split_pool_synthesis import draw_synthesis
from split_pool_encoding.quantization import ConcentrationLibrary, RecipeSettings
from split_pool_encoding.models import transitions
from test_split_pool_image import make_image_fixture


def fitted():
    lib = ConcentrationLibrary('on_off', 3, 3, 2, RecipeSettings(20000, 200000, 10000, 1000, 2000, 1000, 50))
    recipe = lib.recipe(lib.initial(np.random.default_rng(8)))
    indices = np.array([1, 4, 7, 2]); assignment = np.array([2, 0, 1, 3])
    return dict(model='on_off', recipe=recipe, address_bits=3, assignment=assignment,
                address_indices=indices, prediction=transitions(recipe['u'], recipe['v'])[1][indices])


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_highlight_tracks_rotated_gaussian_raster_without_mutation(rotation):
    from image_encoding import geometry
    designs, _ = make_image_fixture(3)
    design = selected_design(designs, 'Green', 2)
    design['origami_rotation_degrees_clockwise'] = rotation
    before = deepcopy(design)
    fig = Figure(); ax = fig.subplots()
    image = Image.new('RGB', (100, 200) if rotation in (90, 270) else (200, 100))
    highlight_image(ax, design, image)
    assert len(ax.patches) == 3
    info = geometry(design['encoding_scheme'], rotation)
    site = design['image_sites'][0]
    x = (site['x_nm']+info['pixel_pitch_x_nm']/2)/info['field_width_nm']*200-.5
    y = (site['y_nm']+info['pixel_pitch_y_nm']/2)/info['field_height_nm']*100-.5
    expected = {0:(x,y),90:(99-y,x),180:(199-x,99-y),270:(y,199-x)}[rotation]
    box = ax.patches[0]
    np.testing.assert_allclose((box.get_x()+box.get_width()/2, box.get_y()+box.get_height()/2), expected)
    assert design == before


def test_selected_origami_uses_inverse_assignment_and_channel():
    designs, _ = make_image_fixture(3)
    design = selected_design(designs, 'Blue', 0)
    result = fitted(); fig = Figure()
    draw_selected_origami(fig, design, result, 0)
    assert 'Blue_0' in fig.axes[0].get_title()
    assert '100' in fig.axes[1].get_title()
    np.testing.assert_array_equal(fig.axes[1].images[0].get_array().ravel(), result['prediction'][1])


def test_selected_route_and_evolution_follow_assignment_and_show_digital_code(tmp_path):
    result = fitted(); fig = Figure()
    draw_synthesis(fig, {'RGB': result}, result, 0)
    ax = fig.axes[0]
    assert 'selected route 100' in ax.get_title()
    chosen = [patch for patch in ax.patches if patch.get_linewidth() == 2.5]
    assert len(chosen) == 3
    assert chosen[0].get_y() < 0  # first branch is 1
    assert chosen[1].get_y() > 0 and chosen[2].get_y() > 0
    labels = [text.get_text() for text in ax.texts]
    assert any('Digital D1=1 (ON)' in text and 'Code: 1' in text for text in labels)
    assert any('Digital D3=0 (OFF)' in text and 'Code: 100' in text for text in labels)
    fig = Figure(figsize=(12, 10), layout='constrained'); draw_evolution(fig, result, 0)
    axes = [ax for ax in fig.axes if ax.get_title().replace('\n', ' ') == 'Accumulated pattern after capture']
    assert len(axes) == 3
    np.testing.assert_allclose(axes[-1]._mixed_components[0], result['prediction'][1])
    assert any('+ K' in ax.get_title() for ax in fig.axes)
    fig.savefig(tmp_path/'selected-route-evolution.png')


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_fitted_canvas_matches_origami_frame_and_site_positions(rotation):
    from image_encoding import ROWS, rotate_point
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    designs, _ = make_image_fixture(3)
    design = selected_design(designs, 'Red', 0)
    design['origami_rotation_degrees_clockwise'] = rotation
    for index, site in enumerate(design['image_sites']):
        site.update(row=ROWS[index], column='06')
    result = fitted(); figure = Figure(figsize=(10, 4), layout='constrained')
    canvas = FigureCanvasAgg(figure)
    draw_selected_origami(figure, design, result, 0); canvas.draw()
    original, fitted_ax = figure.axes
    np.testing.assert_allclose(fitted_ax.get_xlim(), original.get_xlim())
    np.testing.assert_allclose(fitted_ax.get_ylim(), original.get_ylim())
    assert fitted_ax.bbox.width/fitted_ax.bbox.height == pytest.approx(original.bbox.width/original.bbox.height)
    collection = fitted_ax.collections[0]
    np.testing.assert_array_equal(collection.get_array(), result['prediction'][1])
    bounds = collection.get_paths()[0].get_extents()
    np.testing.assert_allclose((bounds.x0+bounds.width/2, bounds.y0+bounds.height/2),
                              rotate_point(69.4, 0, 118.9, 35.7, rotation))


def test_combined_synthesis_aligns_each_round_with_all_route_panels(tmp_path):
    from split_pool_synthesis import draw_combined_synthesis
    result = fitted(); fig = Figure(figsize=(16, 10), layout='constrained')
    draw_combined_synthesis(fig, {'RGB': result}, result, 0)
    diagrams = [ax for ax in fig.axes if (ax.get_title() or ax.get_title(loc='left')).startswith('Round ')]
    # Contribution panels also have round titles; flow diagrams contain patches.
    diagrams = [ax for ax in diagrams if ax.patches]
    captures = [ax for ax in fig.axes if ax.get_title().replace('\n', ' ') == 'Accumulated pattern after capture']
    assert len(diagrams) == len(captures) == 3
    for r, (diagram, capture) in enumerate(zip(diagrams, captures)):
        assert diagram.get_subplotspec().get_topmost_subplotspec().rowspan.start == r
        assert capture.get_subplotspec().get_topmost_subplotspec().rowspan.start == r
        assert sum(p.get_linewidth() == 2.5 for p in diagram.patches) == 1
    from matplotlib.patches import ConnectionPatch
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    connectors = [artist for artist in fig.artists if isinstance(artist, ConnectionPatch)]
    assert len(connectors) == 2
    for i, connector in enumerate(connectors):
        assert connector.axesA is diagrams[i]
        assert connector.axesB is diagrams[i+1]
        start = connector.axesA.transData.transform(connector.xy1)
        end = connector.axesB.transData.transform(connector.xy2)
        assert start[0] == pytest.approx(end[0])
        assert start[1] > end[1]  # pooling always flows down into the next resplit
    np.testing.assert_allclose(captures[-1]._mixed_components[0], result['prediction'][1])
    assert any('+ K1' in ax.get_title() for ax in fig.axes)
    assert any('+ K2' in ax.get_title() for ax in fig.axes)
    fig.savefig(tmp_path/'complete-synthesis.png')


@pytest.mark.parametrize('independent', [False, True])
def test_combined_histogram_uses_selected_destinations_and_all_k(independent):
    from split_pool_synthesis import draw_combined_synthesis
    result = fitted()
    if independent:
        result['synthesis_mode'] = 'independent_wells'
    fig = Figure()
    # An empty canvas page must not hide any of the volume additions.
    draw_combined_synthesis(fig, {'RGB': result}, result, 0, vector_index=9)
    histogram, = [ax for ax in fig.axes if ax.get_title().replace('\n', ' ') in ('Prepared K-vector additions at each destination well', 'K additions')]
    branches = [1] if independent else [1, 2, 4]  # assignment target 0 -> address 100
    amounts = result['recipe']['branch_transfers_nl'][branches]
    active = np.flatnonzero(amounts.sum(axis=0) > 0)
    for bars, k in zip(histogram.containers, active):
        assert bars.get_label() == f'K{k+1}'
        np.testing.assert_allclose([patch.get_height() if independent else patch.get_width() for patch in bars], amounts[:, k])
    totals = np.sum([[p.get_height() if independent else p.get_width() for p in bars] for bars in histogram.containers], axis=0)
    np.testing.assert_allclose(totals, result['recipe']['settings']['branch_volume_nl'])


def test_each_premixed_k_transfer_is_drawn_once_with_both_components():
    from split_pool_canvas_build import canvas_build_steps
    result = fitted(); plan = canvas_build_steps(result, 0)
    figure = Figure(); draw_evolution(figure, result, 0)
    additions = [event for event in plan['events'] if event['phase'] == 'addition']
    panels = [ax for ax in figure.axes if 'Premixed ON + OFF' in ax.get_title()]
    assert len(panels) == len(additions)
    for panel, event in zip(panels, additions):
        np.testing.assert_allclose(panel._mixed_components[0], event['added'][:, 0])
        np.testing.assert_allclose(panel._mixed_components[1], event['added'][:, 1])
    assert not any('After capture •' in ax.get_title() for ax in figure.axes)


def test_volume_timeline_runs_top_to_bottom_beside_rounds():
    from split_pool_synthesis import draw_combined_synthesis
    result = fitted(); figure = Figure()
    draw_combined_synthesis(figure, {'RGB': result}, result, 0)
    chart, = [ax for ax in figure.axes if ax.get_title() == 'K additions']
    spec = chart.get_subplotspec()
    assert spec.colspan.start == 2
    assert spec.rowspan.start == 0 and spec.rowspan.stop == 3
    assert chart.yaxis_inverted()
    assert [label.get_text() for label in chart.get_yticklabels()] == [
        'R1 · 1', 'R2 · 0', 'R3 · 0']
    for bars in chart.containers:
        np.testing.assert_allclose([bar.get_y()+bar.get_height()/2 for bar in bars], [0, 1, 2])


def test_barcode_builds_one_mapped_bit_per_round_without_changing_analog_fit():
    from split_pool_selection import draw_barcode_progress
    from image_encoding import ROWS
    design = {'barcode_sites': [[['H01', '12']], [['H03', '11']], [['H05', '10']]],
              'image_sites': [{'row': 'H01', 'column': '06'}]}
    figure = Figure(); axes = figure.subplots(3, 1)
    for completed, ax in enumerate(axes, 1):
        assert draw_barcode_progress(ax, design, '101', completed)
        assert [a.get_gid() for a in ax.collections] == [
            f'barcode-bit-{i+1}-{bit if i < completed else "pending"}' for i, bit in enumerate('101')]
    assert axes[1].collections[1].get_facecolors()[0, 0] == 1  # OFF is explicitly marked.


def test_barcode_overlay_requires_compatible_site_mapping():
    from split_pool_selection import draw_barcode_progress
    figure = Figure(); ax = figure.subplots()
    design = {'barcode_sites': [[['H01', '06']]], 'image_sites': [{'row': 'H01', 'column': '06'}]}
    assert not draw_barcode_progress(ax, design, '10', 1)
    assert not draw_barcode_progress(ax, design, '1', 1)
    assert not ax.collections


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_dashboard_goal_result_use_selected_assignment_and_shared_geometry(rotation):
    from split_pool_selection import draw_selection_comparison
    from image_encoding import ROWS
    designs, _ = make_image_fixture(3)
    design = selected_design(designs, 'Blue', 0)
    design['origami_rotation_degrees_clockwise'] = rotation
    for index, site in enumerate(design['image_sites']):
        site.update(row=ROWS[index], column='06')
    result = fitted(); before = result['prediction'].copy()
    figure = Figure()
    draw_selection_comparison(figure, Image.new('RGB', (200, 100)), design, result, 0)
    overview, goal, actual, chart = figure.axes
    assert len(overview.patches) == 3
    np.testing.assert_allclose(goal.collections[0].get_array(), [.7, .7, .7])
    np.testing.assert_array_equal(actual.collections[0].get_array(), result['prediction'][1])
    np.testing.assert_array_equal(chart.lines[1].get_ydata(), result['prediction'][1])
    np.testing.assert_array_equal(chart.lines[0].get_ydata(), [.7, .7, .7])
    np.testing.assert_allclose(goal.get_xlim(), actual.get_xlim())
    np.testing.assert_allclose(goal.get_ylim(), actual.get_ylim())
    assert goal.collections[0].get_clim() == actual.collections[0].get_clim() == (0, 1)
    np.testing.assert_array_equal(result['prediction'], before)


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_goal_and_result_include_original_and_assigned_barcode_sites(rotation):
    from split_pool_selection import draw_selection_comparison
    from image_encoding import ROWS, rotate_point
    designs, _ = make_image_fixture(3)
    design = selected_design(designs, 'Red', 0)
    design.update(bits=[0, 1, 1], barcode_sites=[[['H01', '12']], [['H03', '11']], [['H05', '10']]],
                  origami_rotation_degrees_clockwise=rotation)
    for index, site in enumerate(design['image_sites']):
        site.update(row=ROWS[index], column='06')
    result = fitted(); before = deepcopy(design)
    figure = Figure()
    draw_selection_comparison(figure, Image.new('RGB', (200, 100)), design, result, 0)
    goal, actual = figure.axes[1:3]
    for ax, code in ((goal, '011'), (actual, '100')):
        markers = [artist for artist in ax.collections if artist.get_gid()]
        assert [artist.get_gid() for artist in markers] == [f'barcode-bit-{i+1}-{bit}' for i, bit in enumerate(code)]
        np.testing.assert_allclose(markers[0].get_offsets()[0], rotate_point(0, 0, 118.9, 35.7, rotation))
    assert any('Original barcode: 011' in text.get_text() for text in goal.texts)
    assert any('Planned barcode (not simulated): 100' in text.get_text() for text in actual.texts)
    assert design == before


def test_compact_layout_reserves_canvas_space_and_keeps_barcode_caption_below():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from split_pool_synthesis import draw_combined_synthesis
    from image_encoding import ROWS
    result = fitted()
    sites = [{'row': ROWS[i], 'column': '06'} for i in range(3)]
    design = {'image_sites': sites, 'barcode_sites': [[['H01', '12']], [['H03', '11']], [['H05', '10']]]}
    figure = Figure(figsize=(10, 10), layout='constrained'); canvas = FigureCanvasAgg(figure)
    draw_combined_synthesis(figure, {'RGB': result}, result, 0, sites, design=design)
    canvas.draw(); renderer = canvas.get_renderer()
    chart, = [ax for ax in figure.axes if ax.get_title() == 'K additions']
    assert chart.get_position().width < .16
    assert all(p.get_height() <= .15 for bars in chart.containers for p in bars)
    for ax in figure.axes:
        for text in ax.texts:
            if text.get_text().startswith('Planned barcode:'):
                assert text.get_window_extent(renderer).y1 < ax.get_window_extent(renderer).y0
    assert figure._suptitle.get_window_extent(renderer).y1 <= figure.bbox.y1


def test_dashboard_includes_whole_result_with_matching_selection_highlights():
    from split_pool_selection import draw_selection_comparison
    designs, _ = make_image_fixture(3)
    design = selected_design(designs, 'Green', 2)
    target_image = Image.new('RGB', (200, 100), (30, 60, 90))
    result_image = Image.new('RGB', (200, 100), (10, 20, 30))
    result = fitted(); figure = Figure()
    draw_selection_comparison(figure, target_image, design, result, 2, result_image=result_image)
    target, simulated, goal, actual, chart, transfers = figure.axes
    np.testing.assert_array_equal(target.images[0].get_array(), np.asarray(target_image))
    np.testing.assert_array_equal(simulated.images[0].get_array(), np.asarray(result_image))
    assert 'Whole-image split–pool result' in simulated.get_title()
    assert len(target.patches) == len(simulated.patches) == 3
    for target_patch, simulated_patch in zip(target.patches, simulated.patches):
        np.testing.assert_allclose(target_patch.get_bbox().bounds, simulated_patch.get_bbox().bounds)
    assert target.get_shared_x_axes().joined(target, simulated)
    assert target.get_shared_y_axes().joined(target, simulated)
    np.testing.assert_array_equal(chart.lines[1].get_ydata(), result['prediction'][0])
    assert 'Goal' in goal.get_title() and 'result' in actual.get_title()


@pytest.mark.parametrize('width', [8, 10, 14])
def test_round_labels_stay_clear_of_vertical_connectors(width):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import ConnectionPatch
    from split_pool_synthesis import draw_combined_synthesis
    result = fitted(); figure = Figure(figsize=(width, 10), layout='constrained')
    canvas = FigureCanvasAgg(figure)
    draw_combined_synthesis(figure, {'RGB': result}, result, 0)
    canvas.draw(); renderer = canvas.get_renderer()
    for connector in figure.artists:
        if isinstance(connector, ConnectionPatch):
            title_bounds = connector.axesB._left_title.get_window_extent(renderer)
            arrow_x = connector.axesB.transData.transform(connector.xy2)[0]
            assert title_bounds.x0 > arrow_x+3


@pytest.mark.parametrize('vector_index', [1, 65, 128])
def test_large_library_displays_active_vectors_regardless_of_library_page(vector_index):
    result = fitted()
    recipe = result['recipe']
    vectors = np.zeros((128, 3, 2))
    vectors[[25, 40]] = recipe['vector_concentrations_nm']
    transfers = np.zeros((6, 128))
    transfers[1, 25] = 500  # selected route 100: round 1 uses K26
    transfers[2, 40] = 250  # round 2 uses K41; round 3 has no additions
    result['recipe'] = {**recipe, 'vector_concentrations_nm': vectors, 'branch_transfers_nl': transfers}
    figure = Figure()
    draw_evolution(figure, result, 0, vector_index=vector_index)
    additions = [ax for ax in figure.axes if 'Premixed ON + OFF' in ax.get_title()]
    assert len(additions) == 2
    assert '+ K26\n500 nL' in additions[0].get_title()
    assert '+ K41\n250 nL' in additions[1].get_title()
    no_additions = [(ax, text) for ax in figure.axes for text in ax.texts if 'No K additions' in text.get_text()]
    assert len(no_additions) == 1
    assert no_additions[0][0].get_subplotspec().get_topmost_subplotspec().rowspan.start == 2
    assert '0 nL' in no_additions[0][1].get_text()


def test_many_active_additions_wrap_without_omitting_vectors():
    result = fitted(); recipe = result['recipe']
    result['recipe'] = {**recipe, 'vector_concentrations_nm': np.ones((12, 3, 2)),
                        'branch_transfers_nl': np.ones((6, 12))*25}
    figure = Figure(); draw_evolution(figure, result, 0)
    additions = [ax for ax in figure.axes if 'Premixed ON + OFF' in ax.get_title()]
    assert len(additions) == 36
    for k in range(1, 13):
        assert sum(f'+ K{k}\n' in ax.get_title() for ax in additions) == 3


def test_selected_transfer_list_uses_inverse_assignment_and_nonzero_volumes():
    from split_pool_selection import selected_transfer_lines
    result = fitted()
    result['recipe']['branch_transfers_nl'] = np.zeros((6, 128))
    result['recipe']['branch_transfers_nl'][1, 25] = 50000
    result['recipe']['branch_transfers_nl'][2, 40] = 41900
    lines = selected_transfer_lines(result, 0)  # target 0 is assigned address 100
    assert lines == ['Round 1 · branch 1 — K26: 50,000 nL',
                     'Round 2 · branch 0 — K41: 41,900 nL',
                     'Round 3 · branch 0 — No K additions (0 nL)']
    result['synthesis_mode'] = 'independent_wells'
    assert selected_transfer_lines(result, 0) == ['Destination well — K26: 50,000 nL']
    assert 'No physical' in selected_transfer_lines({'model': 'pca'}, 0)[0]

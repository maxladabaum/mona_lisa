import numpy as np
import pytest
from split_pool_vectors import vector_panels
from split_pool_encoding.library import BasisLibrary


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off'])
def test_every_learned_vector_is_visible(model):
    lib = BasisLibrary(model, 3, 5, 8)
    params = lib.describe(lib.initial(np.random.default_rng(7)))
    view = vector_panels(dict(model=model, library_parameters=params))
    for _, values in view['panels']:
        assert values.shape == (8, 5)
    if model == 'on_off':
        np.testing.assert_allclose(sum(v for _, v in view['panels']), 1)
    else:
        np.testing.assert_array_equal(view['panels'][0][1], params['basis_vectors'])


def test_quantized_view_shows_actual_diluted_stocks_not_ideal_library():
    concentrations = np.arange(24).reshape(4, 3, 2)*50.
    result = dict(model='on_off', library_parameters={'basis_vectors': np.zeros((4, 3, 3))},
                  recipe={'vector_concentrations_nm': concentrations})
    view = vector_panels(result)
    assert view['unit'] == 'Concentration (nM)'
    np.testing.assert_array_equal(view['panels'][0][1], concentrations[..., 0])
    np.testing.assert_array_equal(view['panels'][1][1], concentrations[..., 1])


def test_pca_components_and_unrestricted_are_not_mislabeled_as_k_library():
    components = np.eye(3)
    view = vector_panels(dict(model='pca', parameters=components))
    np.testing.assert_array_equal(view['panels'][0][1], components)
    assert 'independent of K' in view['note']
    assert vector_panels(dict(model='on_off')) is None


def test_site_canvas_uses_actual_layout_not_flat_order():
    from split_pool_vectors import site_canvas
    sites = [dict(row=r, column=c) for c in range(5) for r in range(8)]
    values = np.array([100*s['row']+s['column'] for s in sites])
    canvas, rows, columns, note = site_canvas(values, sites)
    assert canvas.shape == (8, 5)
    np.testing.assert_array_equal(canvas, np.arange(8)[:, None]*100+np.arange(5))
    assert rows == list(range(8)) and columns == list(range(5))
    assert note == 'Origami site layout'


def test_site_canvas_handles_other_geometries_and_labels_synthetic_layout():
    from split_pool_vectors import site_canvas
    canvas, _, _, note = site_canvas([.2, .5, .8], [dict(row=1, column=3), dict(row=2, column=4), dict(row=1, column=4)])
    np.testing.assert_allclose(canvas, [[.2, .8], [np.nan, .5]], equal_nan=True)
    assert 'Origami' in note
    canvas, _, _, note = site_canvas(np.arange(40))
    assert canvas.shape == (8, 5)
    assert 'no spatial metadata' in note


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_spatial_canvases_share_physical_frame_and_reuse_artists(rotation):
    from matplotlib.figure import Figure
    from split_pool_vectors import draw_site_canvas, update_site_canvas
    from image_encoding import rotate_point
    sites = [{'row': 'H01', 'column': '06'}, {'row': 'H15', 'column': '02'}]
    figure = Figure(); axes = figure.subplots(1, 3)
    for ax, values in zip(axes, ([.1, .8], [.3, .6], [.5, .4])):
        artist = draw_site_canvas(ax, values, sites, rotation, vmin=0, vmax=1)
        center = artist.get_paths()[0].get_extents()
        np.testing.assert_allclose((center.x0+center.width/2, center.y0+center.height/2),
                                  rotate_point(69.4, 0, 118.9, 35.7, rotation))
        assert update_site_canvas(ax, [.7, .2]) is artist
        np.testing.assert_array_equal(artist.get_array(), [.7, .2])
        assert ax.get_aspect() == 1
    for ax in axes[1:]:
        np.testing.assert_array_equal(ax.get_xlim(), axes[0].get_xlim())
        np.testing.assert_array_equal(ax.get_ylim(), axes[0].get_ylim())
    width, height = (35.7, 118.9) if rotation in (90, 270) else (118.9, 35.7)
    np.testing.assert_allclose(axes[0].get_xlim(), [-.2*width, 1.2*width])
    np.testing.assert_allclose(axes[0].get_ylim(), [1.25*height, -.25*height])

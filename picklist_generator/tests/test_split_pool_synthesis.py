import numpy as np
from matplotlib.figure import Figure
from split_pool_synthesis import draw_synthesis
from split_pool_encoding.shared_pool import color_views


def shared_fixture():
    recipe = dict(branch_transfers_nl=np.array([[25, 0], [0, 50], [75, 0], [0, 100]]),
                  branch_remaining_nl=np.array([975, 950, 925, 900]),
                  source_transfers_nl=np.ones((2, 3, 2))*25,
                  vector_buffer_nl=np.array([850, 850]), settings={'branch_volume_nl': 1000})
    fit = dict(recipe=recipe, address_bits=2, assignment=np.arange(3),
               address_indices=np.arange(3), prediction=np.zeros((3, 3)))
    datasets = {color: np.zeros((1, 3)) for color in ('Red', 'Green', 'Blue')}
    return fit, color_views(fit, datasets)


def test_all_colors_share_branches_with_named_stock_additions():
    shared, views = shared_fixture()
    fig = Figure(); draw_synthesis(fig, views)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert '3 assigned designs in one pool' in ax.get_title()
    assert '4 shared branch wells' in ax.get_title()
    labels = [text.get_text() for text in ax.texts]
    for round_number in (1, 2):
        for bit in (0, 1):
            assert any(f'R{round_number} · bit {bit}' in label for label in labels)
    assert 'Pool 1' in labels and 'Final\npool' in labels
    assert any('R2 · bit 1' in label and 'K2: 100 nL' in label for label in labels)
    assert any('R1 · bit 0' in label and 'K1: 25 nL' in label for label in labels)
    assert 'K2: 100 nL' in ax.format_coord(3, -.85)
    assert 'not specified' in ax.format_coord(2, 0)


def test_separate_cached_color_fits_require_refitting():
    shared, _ = shared_fixture()
    fig = Figure(); draw_synthesis(fig, {'Red': shared, 'Green': shared})
    assert 'Click Visualize analysis' in fig.axes[0].texts[0].get_text()


def test_no_recipe_does_not_invent_liquid_volumes():
    fig = Figure(); draw_synthesis(fig, {'Red': {'model': 'pca'}})
    assert '25 nL recipes enabled' in fig.axes[0].texts[0].get_text()


def test_independent_wells_still_have_separate_destination_maps():
    shared, _ = shared_fixture()
    shared['synthesis_mode'] = 'independent_wells'
    fig = Figure(); draw_synthesis(fig, {'Red': shared, 'Green': shared})
    assert len([ax for ax in fig.axes if ax.images]) == 2

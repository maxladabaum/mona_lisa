import numpy as np
import pytest
from split_pool_encoding.shared_pool import combined_targets, color_views
from split_pool_encoding.optimization import fit


def test_joint_fit_assigns_all_120_targets_once_with_one_library():
    datasets = {color: np.random.default_rng(i).random((40, 3))
                for i, color in enumerate(('Red', 'Green', 'Blue'))}
    metadata = {color: {'sites': ['a', 'b', 'c']} for color in datasets}
    targets = combined_targets(datasets, metadata)
    assert targets.shape == (120, 3)
    shared = fit(targets, 'on_off', address_bits=7, allow_partial=True,
                 basis_k=2, restarts=1, alternations=1, maxiter=3)
    views = color_views(shared, datasets)
    all_addresses = np.concatenate([view['address_indices'] for view in views.values()])
    assert len(np.unique(all_addresses)) == 120
    for color, view in views.items():
        np.testing.assert_array_equal(np.sort(view['assignment']), np.arange(40))
        assert view['shared_pool_fit'] is shared
        assert view['library_parameters'] is shared['library_parameters']
        assert view['basis_k'] == 2
        for row, address in enumerate(view['address_indices']):
            global_row = np.flatnonzero(shared['address_indices'] == address)[0]
            np.testing.assert_array_equal(view['prediction'][row], shared['prediction'][global_row])
        assert view['mse'] == pytest.approx(np.mean((view['prediction']-datasets[color][view['assignment']])**2))


def test_mismatched_sites_cannot_be_pooled_silently():
    with pytest.raises(ValueError, match='same analog site layout'):
        combined_targets({'R': np.zeros((1, 2)), 'G': np.zeros((1, 2))},
                         {'R': {'sites': ['a', 'b']}, 'G': {'sites': ['b', 'a']}})


def test_color_views_share_exactly_one_physical_recipe_and_budget():
    from split_pool_encoding.quantization import RecipeSettings
    datasets = {color: np.random.default_rng(i).random((4, 3))
                for i, color in enumerate(('Red', 'Green', 'Blue'))}
    settings = RecipeSettings(on_stock_nm=1000, off_stock_nm=1000,
                              vector_volume_nl=10000, vector_max_nm=100,
                              branch_volume_nl=1000, branch_budget_nl=500,
                              capture_c50_nm=50)
    shared = fit(np.concatenate(list(datasets.values())), 'on_off',
                 address_bits=4, allow_partial=True, basis_k=2,
                 recipe_settings=settings, recipe_probes=0,
                 restarts=1, alternations=1, maxiter=3)
    views = color_views(shared, datasets)
    recipe = shared['recipe']
    assert recipe['branch_transfers_nl'].shape == (8, 2)
    assert np.all(recipe['branch_transfers_nl'].sum(axis=1) <= 500)
    for view in views.values():
        assert view['recipe'] is recipe
        assert view['shared_pool_fit'] is shared

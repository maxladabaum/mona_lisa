from copy import deepcopy
import numpy as np
import pytest
from split_pool_image import image_cases, render_image_example


def make_image_fixture(positions=40):
    designs = []; fits = {}; pi = np.array([2, 0, 3, 1])
    for channel, color in enumerate(('Red', 'Green', 'Blue')):
        target = np.full((4, positions), .2+.25*channel)
        for row in range(4):
            sites = [dict(name=str(j), replacement_fraction=float(target[row, j]),
                          x_nm=3+(j%6)*6+row, y_nm=3+(j//6)*6) for j in range(positions)]
            designs.append(dict(name=f'{color}_{row}', color=color, channel=channel, barcode_code=row,
                encoding_scheme='fixed_4x4', image_sites=sites,
                site_fractions={**{s['name']: s['replacement_fraction'] for s in sites}, 'barcode': 1.}))
        fits[color] = dict(model='additive', assignment=pi, prediction=target[pi])
    return designs, fits


def test_inverse_assignment_channels_and_no_mutation():
    designs, fits = make_image_fixture(3); original = deepcopy(designs)
    expected = np.array([[0, 1, .1], [1, 0, .3], [.2, .4, .6], [.9, .7, .5]])
    fits['Red']['prediction'] = expected[fits['Red']['assignment']]
    (target, mean, sampled), errors = image_cases(designs, fits, 300, 42)
    np.testing.assert_allclose([[s['replacement_fraction'] for s in d['image_sites']] for d in mean[:4]], expected)
    assert sampled[0]['image_sites'][0]['replacement_fraction'] == 0
    assert sampled[0]['image_sites'][1]['replacement_fraction'] == 1
    assert all(d['site_fractions']['barcode'] == 1 for d in sampled)
    assert designs == original
    assert errors['encoding_rmse'] > 0
    assert image_cases(designs, fits, 300, 42)[0] == (target, mean, sampled)
    assert image_cases(designs, fits, 300, 43)[0][2] != sampled


def test_count_changes_noise_without_refitting_expected_image():
    designs, fits = make_image_fixture()
    low, low_errors = image_cases(designs, fits, 30, 8)
    high, high_errors = image_cases(designs, fits, 3000, 8)
    assert low[1] == high[1]
    assert low_errors['encoding_rmse'] == 0
    assert high_errors['sampling_rmse'] < low_errors['sampling_rmse']/5


def test_gaussian_rgb_preview_and_invalid_inputs():
    designs, fits = make_image_fixture(3)
    images, errors = render_image_example(designs, fits, 300, 1, dict(fwhm_nm=7.5, exposure=1))
    assert len(images) == 3
    assert len({i.size for i in images}) == 1
    np.testing.assert_array_equal(images[0], images[1])
    assert not np.array_equal(images[1], images[2])
    for count in (0, -1, 2.5):
        with pytest.raises(ValueError):
            image_cases(designs, fits, count)
    with pytest.raises(ValueError):
        image_cases(designs, {'Red': fits['Red']})
    fits['Red']['assignment'] = np.array([0, 0, 2, 3])
    with pytest.raises(ValueError):
        image_cases(designs, fits)


def test_pca_preview_clips_without_changing_fit_and_reports_raw_error():
    designs, fits = make_image_fixture(3)
    expected = np.array([[-.25, 1.3, .4], [.1, .2, .3], [.6, .7, .8], [.3, .4, .5]])
    for result in fits.values():
        result['model'] = 'pca'
        result['prediction'] = expected[result['assignment']].copy()
    original = deepcopy(fits)
    (_, mean, sampled), errors = image_cases(designs, fits, 300, 5)
    np.testing.assert_allclose([[s['replacement_fraction'] for s in d['image_sites']] for d in mean[:4]], np.clip(expected, 0, 1))
    assert sampled[0]['image_sites'][0]['replacement_fraction'] == 0
    assert sampled[0]['image_sites'][1]['replacement_fraction'] == 1
    assert errors['clipped_fraction'] == pytest.approx(2/12)
    truth = np.array([[s['replacement_fraction'] for s in d['image_sites']] for d in designs])
    assert errors['raw_encoding_rmse'] == pytest.approx(np.sqrt(np.mean((np.tile(expected, (3, 1))-truth)**2)))
    assert errors['encoding_rmse'] < errors['raw_encoding_rmse']
    for color in fits:
        np.testing.assert_array_equal(fits[color]['prediction'], original[color]['prediction'])
    fits['Red']['prediction'][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        image_cases(designs, fits)


def test_physical_models_still_reject_invalid_probability():
    designs, fits = make_image_fixture(3)
    fits['Red']['prediction'][0, 0] = 1.3
    with pytest.raises(ValueError, match='probabilities'):
        image_cases(designs, fits)

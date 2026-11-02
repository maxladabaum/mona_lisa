"""Typed barcode capacities work through encoding, templates, and RGB output."""
import pytest

from barcode_template_schema import from_template, to_template
from image_encoding import (CHANNELS, BARCODE_SCHEMA_KEYS, barcode_bits,
                            compile_designs, default_barcode_schema, encoding_scheme,
                            geometry, reconstruct, schemes_for_bits)


@pytest.mark.parametrize('count', [2, 3, 8, 10, 48])
def test_generated_schemes_fit_capacity(count):
    choices = schemes_for_bits(count)
    assert choices
    for key, scheme in choices.items():
        codes = [tuple(barcode_bits(color, x, y, key)) for color in CHANNELS
                 for y in range(scheme.y_states) for x in range(scheme.x_states)]
        assert all(len(code) == count for code in codes)
        assert len(set(codes)) == scheme.design_count
        assert scheme.design_count <= 2 ** count
    assert max(s.design_count for s in choices.values()) == 3 * min(128, 2 ** count // 3)


@pytest.mark.parametrize('count', [2, 8, 10, 48])
@pytest.mark.parametrize('mode', ['fixed', 'joint'])
@pytest.mark.parametrize('layout', ['tiles', 'shifts'])
def test_new_counts_roundtrip_images_and_templates(count, mode, layout):
    grid = '1x1' if count == 2 else '2x2'
    key = f'bits_{count}_{mode}_{layout}_{grid}'
    assert encoding_scheme(key).bit_count == count
    schema = default_barcode_schema(count)
    restored, restored_count = from_template(to_template(schema, count))
    assert restored_count == count
    assert [set(g['sites']) for g in restored[BARCODE_SCHEMA_KEYS[count]]] == [
        set(g['sites']) for g in schema[BARCODE_SCHEMA_KEYS[count]]]
    info = geometry(key, barcode_schema=restored)
    pixels = [[[255, 128, 0] for _ in range(info['columns'])] for _ in range(info['rows'])]
    designs = compile_designs(pixels, 50, {c: 1 for c in CHANNELS}, key, barcode_schema=restored)
    assert reconstruct(designs) == pixels
    assert len(info['barcode_bits']) == count
    assert info['reserved_code_count'] == 2 ** count - len(designs)


@pytest.mark.parametrize('count', [0, 1, 49, -1, 3.5, '8', True])
def test_invalid_bit_counts(count):
    with pytest.raises(ValueError, match='whole number from 2 to 48'):
        schemes_for_bits(count)


def test_typed_bits_update_controls_and_reject_invalid_text(tmp_path, monkeypatch):
    from unittest.mock import patch
    import picklist_app
    for name in ('CONFIG_PATH', 'PLATE_STATE_PATH', 'STORAGE_STATE_PATH'):
        monkeypatch.setattr(picklist_app, name, tmp_path / (name + '.json'))
    app = picklist_app.PicklistApp()
    app.withdraw()
    try:
        view = app.rgb_image_view
        view.barcode_bit_count.set('8')
        view._bit_count_changed()
        assert encoding_scheme(view._scheme_id()).bit_count == 8
        assert all(encoding_scheme(key).bit_count == 8 for key in view._display_scheme_ids)
        for invalid in ('', 'abc', '8.5', '1', '49'):
            with patch('rgb_image_view.messagebox.showerror') as error:
                view.barcode_bit_count.set(invalid)
                view._bit_count_changed()
                error.assert_called_once()
                assert view.barcode_bit_count.get() == '8'
    finally:
        app.destroy()

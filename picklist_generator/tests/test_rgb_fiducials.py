from copy import deepcopy
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from image_encoding import (CHANNELS, IMAGE_COLUMNS, default_barcode_schema, validate_barcode_schema,
    compile_designs, geometry, site_name, fiducial_sites)
from gaussian_reconstruction import render_gaussian_overlay
from gaussian_optimization import optimize_designs
from alignment_reconstruction import AlignmentModel, AlignmentSettings


def schema():
    value = default_barcode_schema()
    value['always_on_sites'] = ['H07-12', 'H09-12', 'H01-01']
    return value


def assert_fiducials(designs):
    for design in designs:
        assert len(design['always_on_sites']) == 3
        assert all(design['site_fractions'][site_name(*site)] == 1.0 for site in design['always_on_sites'])
        assert not {site_name(*site) for site in design['always_on_sites']} & {s['name'] for s in design['image_sites']}


@pytest.mark.parametrize('scheme', ['fixed_4x4', 'tiles_2x2', 'joint_7x3', 'nine_fixed_16x8'])
def test_fiducials_always_on_for_every_color_and_code_even_with_black_image(scheme):
    columns = IMAGE_COLUMNS[:-1]
    info = geometry(scheme, active_image_columns=columns, barcode_schema=schema())
    pixels = [[[0,0,0] for _ in range(info['columns'])] for _ in range(info['rows'])]
    designs = compile_designs(pixels,25,{c:1 for c in CHANNELS},scheme,
                             active_image_columns=columns,barcode_schema=schema())
    assert_fiducials(designs)
    assert all(s['replacement_fraction']==0 for d in designs for s in d['image_sites'])
    assert info['always_on_sites'] == [['H07','12'],['H09','12'],['H01','01']]


def test_optimizers_cannot_turn_off_fiducials():
    designs = compile_designs([[[255,255,255] for _ in range(5)] for _ in range(8)],
        50,{c:1 for c in CHANNELS},'tiles_1x1',active_image_columns=IMAGE_COLUMNS[:-1],barcode_schema=schema())
    before = deepcopy(designs)
    overlay = render_gaussian_overlay(designs,10,.5)
    optimized, _ = optimize_designs(designs,np.zeros_like(overlay.density),10,.5)
    model = AlignmentModel(designs,AlignmentSettings(),10,.5)
    aligned, _ = model.optimize(designs,np.zeros_like(model.expected(designs).density))
    assert designs == before
    assert_fiducials(optimized)
    assert_fiducials(aligned)


@pytest.mark.parametrize('sites', [['H01-11'], ['H01-12'], ['H07-12','H07-12'], ['H02-12'], 'H07-12'])
def test_invalid_or_overlapping_fiducials_rejected(sites):
    value = default_barcode_schema(); value['always_on_sites'] = sites
    with pytest.raises(ValueError): validate_barcode_schema(value, 9 if sites == ["H01-12"] else 6)


def test_active_analog_overlap_rejected():
    with pytest.raises(ValueError,match='active analog canvas'):
        fiducial_sites(schema(), IMAGE_COLUMNS)
    assert len(fiducial_sites(schema(), IMAGE_COLUMNS[:-1])) == 3


def test_ui_picklists_metadata_and_canvas_guard(tmp_path, monkeypatch):
    import picklist_app
    from picklist_core import parse_source2
    from exhaustive_pdf import ReportSettings, generate_views
    for name in ('CONFIG_PATH','PLATE_STATE_PATH','STORAGE_STATE_PATH'):
        monkeypatch.setattr(picklist_app,name,tmp_path/(name+'.json'))
    app = picklist_app.PicklistApp(); app.withdraw()
    try:
        view = app.rgb_image_view
        view.base_plate_picker.current(1)
        view.final_volume.set('20')
        view.base_target.set('25')
        for source in view.color_views.values():
            source.desired_concentration_nm.set('25')
        view.tile_count.set('1'); view._tile_count_changed()
        previous = deepcopy(view.barcode_schema)
        with pytest.raises(ValueError,match='active analog canvas'):
            view._apply_barcode_schema(schema())
        assert view.barcode_schema == previous
        view.active_row_vars['01'].set(False); view._active_canvas_changed('01')
        view._apply_barcode_schema(schema())
        view.active_row_vars['01'].set(True); view._active_canvas_changed('01')
        assert not view.active_row_vars['01'].get()
        path = tmp_path/'black.png'; Image.new('RGB',(5,8),'black').save(path)
        with patch('rgb_image_view.messagebox.showerror') as error:
            view.image_path.set(str(path)); view._load(); view._prepare()
            assert len(view.runs) == 3
            assert_fiducials(view.designs)
            assert len(view.design_canvas.find_withtag('fiducial_sites')) == 3
            rows, _ = parse_source2(picklist_app.SHEET_DIR/'PAINT_R1_replace.csv')
            wells = {r['Name']:r['Well'] for r in rows}
            permanent_wells = {wells[site_name(*site)] for site in view.designs[0]['always_on_sites']}
            for run in view.runs:
                transfers = {r['Source Well']:r['Transfer Volume'] for r in run['picklist'] if r['Source Plate Name']=='SourcePlate7[7]'}
                assert all(transfers[well] == 25 for well in permanent_wells)
            assert view.snapshot['barcode_schema']['always_on_sites'] == schema()['always_on_sites']
            settings = ReportSettings(str(path),'Stretch',25,{c:1 for c in CHANNELS},10,.5,**view._concentration_options())
            settings.validate()
            report = generate_views(settings,'tiles_1x1',0)
            assert report['geometry']['always_on_sites'] == view.designs[0]['always_on_sites']
            error.assert_not_called()
    finally:
        app.destroy()

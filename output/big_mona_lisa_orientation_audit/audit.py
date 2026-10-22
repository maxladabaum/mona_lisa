"""Read-only audit of the saved batch; writes only diagnostic artifacts here."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'paint_analysis'), str(ROOT/'picklist_generator')]
from paint_analysis_gui import load_custom_template_metadata, logical_bit_model_from_metadata, load_custom_template_image, origami_grid_points
from origami_composite import saved_rgb_batch_mapping
P = ROOT/'picklist_generator/generated_output/big_mona_lisa/2026-10-20_08-35-56'
OUT = Path(__file__).resolve().parent
manifest = json.loads((P/'image_manifest.json').read_text())
templates = json.loads((P/'alignment_templates/manifest.json').read_text())
def seq(s): return ''.join(s.split()).upper()
replacement = list(csv.DictReader((ROOT/'picklist_generator/replacement_sheets/PAINT_R1_replace.csv').open()))
by_name = {r['Name']:seq(r["Sequence (5' > 3')"]) for r in replacement}
checks = Counter()
names=[]
for design, entry in zip(manifest['designs'], templates['templates']):
    assert design['name'] == entry['design_name']
    path = P/'alignment_templates'/entry['png']
    names.append(path.stem)
    meta=load_custom_template_metadata(path)
    model=logical_bit_model_from_metadata(meta)
    assert list(map(int, model['active_bits'])) == design['bits']
    assert int(''.join(map(str,design['bits'])),2)==design['barcode_code']
    expected = {tuple(s) for s in manifest['geometry']['always_on_sites']}
    for bit, group in zip(design['bits'],manifest['geometry']['barcode_bits']):
        if bit: expected.update(map(tuple,group['sites']))
    actual={(f'H{2*s["row"]-1:02}',f'{s["column"]:02}') for s in meta['selected_sites']}
    assert actual==expected
    assert meta['image_mirrored_x'] is True
    assert all(abs(s['x_nm']-(meta['width_nm']-meta['margin_nm']-(s['column']-1)*9.9-meta['logical_model']['column_offsets_nm'][s['column']-1]))<1e-8 for s in meta['selected_sites'])
    raster=load_custom_template_image(path)
    for s in meta['selected_sites']:
        x=round(s['x_nm']/meta['width_nm']*(meta['width_px']-1))
        y=meta['height_px']-1-round(s['y_nm_from_top']/meta['height_nm']*(meta['height_px']-1))
        assert raster[y,x] > raster.max()*.7
        cell=(8-s['row'])*12+(12-s['column'])
        active=set(model['alignment_cells'])
        for enabled,cells in zip(model['active_bits'],model['bit_cells']):
            if enabled: active.update(cells)
        assert cell in active
    rows=list(csv.DictReader((P/design['name']/'picklist.csv').open()))
    actual_volumes=Counter()
    for row in rows:
        if row['Source Plate Name']=='SourcePlate7[7]': actual_volumes[seq(row['Sample Comments'])]+=float(row['Transfer Volume'])
    for name, fraction in design['site_fractions'].items():
        assert abs(actual_volumes[by_name[name]]-fraction*125)<1e-8,(design['name'],name)
        checks['transfer_site_fractions']+=1
    checks['templates_and_codes']+=1
params=dict(rows=8,columns=12,spacing_x_nm=9.9,spacing_y_nm=5.1,column_offsets_nm=meta['column_offsets_nm'])
grid=origami_grid_points(8,12,9.9,5.1,params)
_,_,mappings,defaults=saved_rgb_batch_mapping(manifest,names,grid,params)
assert len(mappings)==120
for name,design in zip(names,manifest['designs']):
    assert mappings[name]['x_shift_nm']==design['x_shift_nm']
    assert mappings[name]['y_shift_nm']==design['y_shift_nm']
checks['reconstruction_mappings']=len(mappings)
combined=list(csv.DictReader((P/'picklist_combined.csv').open()))
copy=list(csv.DictReader((P.parent/'picklist/picklist_combined.csv').open()))
assert combined==copy
checks['combined_csv_copy_matches']=True
schema=manifest['barcode_schema']
def sites(group):return {(int(s[1:3]),int(s[4:])) for s in group}
fid=sites(schema['always_on_sites'])
assert {(16-r,c) for r,c in fid}==fid
assert {(16-r,13-c) for r,c in fid}!=fid
permutation=[]
groups=[sites(g['sites']) for g in schema['seven_bit']]
for g in groups: permutation.append(groups.index({(16-r,c) for r,c in g}))
designs={d['barcode_code']:d for d in manifest['designs']}
reflection=Counter()
normal=np.zeros((64,25,3));wrong=np.zeros_like(normal)
for code,d in designs.items():
    reflected=int(''.join(str(d['bits'][i]) for i in permutation),2)
    target=designs.get(reflected)
    reflection['unchanged' if reflected==code else 'changed']=reflection.get('unchanged' if reflected==code else 'changed',0)+1
    if target is None: reflection['unused_code']+=1
    else:
        reflection['valid_code']+=1
        if target['color']!=d['color']: reflection['changed_color']+=1
    for pixel in d['image_sites']:
        x=d['quadrant_x']*5+6-int(pixel['column']); y=d['quadrant_y']*8+(int(pixel['row'][1:])-1)//2
        channel={'R1 R':0,'R1 G':1,'R1 B':2}[d['color']]
        normal[y,x,channel]=pixel['replacement_fraction']
        if target is not None:
            xx=target['quadrant_x']*5+6-int(pixel['column']); yy=target['quadrant_y']*8+7-(int(pixel['row'][1:])-1)//2
            cc={'R1 R':0,'R1 G':1,'R1 B':2}[target['color']]
            wrong[yy,xx,cc]=pixel['replacement_fraction']
canvas=Image.new('RGB',(640,680),'#202020');draw=ImageDraw.Draw(canvas)
for i,(a,title) in enumerate([(normal,'Verified transfer fractions'),(wrong,'Simulated vertical reflection')]):
    im=Image.fromarray(np.uint8(a*255)).resize((300,640),Image.Resampling.NEAREST)
    canvas.paste(im,(i*320+10,30)); draw.text((i*320+10,8),title,fill='white')
canvas.save(OUT/'reflection_comparison.png')
report=dict(batch=str(P),checks=dict(checks),image_region_defaults=defaults,fiducials_vertical_symmetric=True,fiducials_half_turn_symmetric=False,vertical_reflection_bit_order=[i+1 for i in permutation],vertical_reflection_codes=dict(reflection),limitations='No measured localizations or remote-device session available. Transfer checks compare saved sequences and volumes with repository reference sequences, not physical plate contents.')
(OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

"""Create a reconstruction-only manifest for existing unreflected classifications."""
import json,hashlib,sys
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paint_analysis'))
from origami_composite import saved_rgb_batch_mapping
from paint_analysis_gui import load_custom_template_metadata,origami_grid_points
p=ROOT/'picklist_generator/generated_output/big_mona_lisa/2026-10-20_08-35-56'
source=p/'image_manifest.json';original=json.loads(source.read_text())
by_code={d['barcode_code']:d for d in original['designs']}
permutation=[0,2,1,3,5,4,6]
def reflected(code):
 bits=f'{code:07b}'
 return int(''.join(bits[i] for i in permutation),2)
geometry=deepcopy(original['geometry'])
codebook={e['code']:e for e in geometry['codebook']}
geometry['codebook']=[]
designs=[]
runs=[]
source_runs={r['run_name']:r for r in original['runs']}
for observed in original['designs']:
 code=observed['barcode_code'];physical=by_code[reflected(code)]
 design={k:deepcopy(physical[k]) for k in ['color','x_shift_nm','y_shift_nm','quadrant','quadrant_x','quadrant_y','encoding_scheme']}
 design.update(name=observed['name'],bits=observed['bits'],barcode_code=code,
               physical_design_name=physical['name'],physical_barcode_code=physical['barcode_code'])
 designs.append(design)
 entry=deepcopy(codebook[physical['barcode_code']]);entry.update(code=code,bits=observed['bits'])
 geometry['codebook'].append(entry)
 r=source_runs[physical['name']]
 runs.append(dict(run_name=observed['name'],physical_design_name=physical['name'],
                  destination_plate_name=r['destination_plate_name'],destination_wells=r['destination_wells']))
remapped=dict(format='paint-rgb-reconstruction-manifest-v1',reconstruction_only=True,
              source_manifest=str(source),source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              description='For saved classification made using unreflected templates. Observed template codes and names are preserved; colors and positions refer to the reflected physical design. Not a synthesis/picklist manifest.',
              required_section_mirror='After loading, choose Mirror all sections before XY placement: Top ↔ bottom, then Build composite. Leave whole-image mirrors off. Do not apply this remap to classification already rerun with reflected templates.',
              bit_permutation_one_based=[i+1 for i in permutation],geometry=geometry,designs=designs,runs=runs)
meta=load_custom_template_metadata(p/'alignment_templates/barcodes/design_0001_code_000.png')
params={k:meta[k] for k in ['rows','columns','spacing_x_nm','spacing_y_nm','column_offsets_nm']}
grid=origami_grid_points(8,12,9.9,5.1,params)
names=[f'design_{i+1:04d}_code_{d["barcode_code"]:03d}' for i,d in enumerate(original['designs'])]
_,_,mappings,defaults=saved_rgb_batch_mapping(remapped,names,grid,params)
assert len(mappings)==120 and len({m['code'] for m in mappings.values()})==120
previous=json.loads((ROOT/'output/big_mona_lisa_orientation_audit/vertical_reflection_diagnostic.json').read_text())
for name,m in mappings.items():
 assert int(m['code'],2)==int(name.rsplit('_',1)[1])
 for key in ['color','x_shift_nm','y_shift_nm','rotation_deg','mirror_x','brightness']:
  assert m[key]==previous['mappings'][name][key],(name,key)
 assert reflected(reflected(int(m['code'],2)))==int(m['code'],2)
filename='image_manifest_reflected_classification.json'
encoded=json.dumps(remapped,indent=2)+'\n'
(source.parent/filename).write_text(encoded)
remote=Path('/Volumes/Toshiba_External_Drive/PAINT/big_mona_lisa/big_mona_lisa/2026-10-20_08-35-56')
if remote.is_dir():
 (remote/filename).write_text(encoded)
 assert (remote/filename).read_bytes()==(source.parent/filename).read_bytes()
print('Created',source.parent/filename)
print('Copied to',remote/filename)
print('Validated: all 120 imported mappings; observed labels unchanged; colors and positions match measured-data diagnostic; original manifest unchanged.')
assert hashlib.sha256(source.read_bytes()).hexdigest()==remapped['source_manifest_sha256']

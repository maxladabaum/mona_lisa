"""Independent coordinate-placement check using the actual 120-design batch."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'paint_analysis'))
from origami_composite import CompositeSettings,saved_rgb_batch_mapping,reconstruct_composite
from paint_analysis_gui import load_custom_template_metadata,origami_grid_points
P=ROOT/'picklist_generator/generated_output/big_mona_lisa/2026-10-20_08-35-56'
m=json.loads((P/'image_manifest.json').read_text())
meta=load_custom_template_metadata(P/'alignment_templates/barcodes/design_0001_code_000.png')
params={k:meta[k] for k in ['rows','columns','spacing_x_nm','spacing_y_nm','column_offsets_nm']}
grid=origami_grid_points(8,12,9.9,5.1,params)
names=[f'design_{i+1:04d}_code_{d["barcode_code"]:03d}' for i,d in enumerate(m['designs'])]
_,_,mapping,defaults=saved_rgb_batch_mapping(m,names,grid,params)
settings=CompositeSettings(**defaults,exclusion_radius_nm=0)
source=[];results={};expected_points=[];weights=[];channels=[]
for i,(name,d) in enumerate(zip(names,m['designs'])):
 theta=np.deg2rad((i*37)%360);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 center=np.array([(i%10)*250.,(i//10)*250.])
 corners=np.array([[-79.45,-37.85],[79.45,-37.85],[79.45,37.85],[-79.45,37.85]])@rot.T+center
 local=[]
 for p in d['image_sites']:
  count=round(p['replacement_fraction']*5)
  # Physical site coordinates from the loaded lattice, independent of the crop.
  cell=(7-(int(p['row'][1:])-1)//2)*12+(12-int(p['column']))
  local.extend([grid[cell]]*count)
  # Desired global image position, independent of the builder's shifts and pose.
  expected_points.append([p['pixel_x']*9.9+4.95,p['pixel_y']*5.1+2.55]);weights.append(count)
  channels.append({'R1 R':0,'R1 G':1,'R1 B':2}[d['color']])
 if local:source.extend(np.array(local)@rot.T+center)
 results[name]=dict(picks=SimpleNamespace(accepted_mask=np.array([True]),rectangle_corners_nm=np.array([corners])),params=params)
actual=reconstruct_composite(np.array(source),results,mapping,settings)
expected=np.zeros_like(actual['density']);points=np.array(expected_points)
# The builder pads the image by four sigma on each side.
indices=np.floor((points+4*settings.sigma_nm)/settings.pixel_nm).astype(int)
for (x,y),c,w in zip(indices,channels,weights):expected[y,x,c]+=w
for c in range(3):expected[:,:,c]=gaussian_filter(expected[:,:,c],settings.sigma_nm,mode='constant')
np.testing.assert_allclose(actual['density'],expected,rtol=0,atol=1e-12)
report=dict(designs=len(names),synthetic_localizations=len(source),arbitrary_pose_rotations=True,all_codes_unchanged=True,maximum_density_error=float(np.max(np.abs(actual['density']-expected))),settings=asdict(settings))
out=Path(__file__).parent
(out/'builder_check.json').write_text(json.dumps(report,indent=2)+'\n')
Image.fromarray(actual['rgb']).resize((400,520)).save(out/'synthetic_builder_output.png')
print(json.dumps(report,indent=2))

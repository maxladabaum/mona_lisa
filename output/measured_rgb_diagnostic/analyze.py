"""Read-only diagnostics from saved tile crops, not a full-source reconstruction."""
import sys,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'paint_analysis'))
from analysis_session import load_analysis_session
from origami_analysis import OrigamiPickResult
from origami_composite import pose_basis
P=Path('/Volumes/Toshiba_External_Drive/PAINT/big_mona_lisa/paint-tiles-ozwyoyca')
manifest=json.loads((ROOT/'picklist_generator/generated_output/big_mona_lisa/2026-10-20_08-35-56/image_manifest.json').read_text())
expected=np.zeros((120,8,5));sums=np.zeros_like(expected);counts=np.zeros(120,dtype=int)
grid=np.array([[9.95+x*9.9,17.85-y*5.1] for y in range(8) for x in range(5)])
pose_errors=[];bit_sums=np.zeros((2,7));bit_n=np.zeros((2,7));crop_bounds=[]
files=sorted(P.glob('tile-*.paintanalysis'))
for k,path in enumerate(files):
 payload=load_analysis_session(path,{'OrigamiPickResult':OrigamiPickResult})['payload']
 for item in payload['templates']:
  code=int(item['name'].rsplit('_',1)[1]);picks=item['picks'];params=item['params'];d=manifest['designs'][code]
  expected[code]=np.array([p['replacement_fraction'] for p in d['image_sites']]).reshape(8,5)
  evidence=params.get('digital_group_localization_evidence',())
  for idx in np.flatnonzero(picks.accepted_mask):
   counts[code]+=1
   points=picks.aligned_regions[idx]
   if len(points):
    tree=cKDTree(points)
    sums[code]+=np.array([len(ids) for ids in tree.query_ball_point(grid,3.0)]).reshape(8,5)
    crop_bounds.append([points[:,0].min(),points[:,0].max(),points[:,1].min(),points[:,1].max()])
    center,basis=pose_basis(picks.rectangle_corners_nm[idx]);raw=(picks.regions[idx]-center)@basis
    pose_errors.append(float(cKDTree(raw).query(points)[0].max()))
   if len(evidence)>idx:
    for b,on in enumerate(d['bits']):bit_sums[on,b]+=evidence[idx][b];bit_n[on,b]+=1
 if (k+1)%20==0:print('Read',k+1,'/',len(files),'tiles',flush=True)
mean=sums/np.maximum(counts[:,None,None],1)
valid=counts>0
corr=lambda a,b:float(np.corrcoef(a.ravel(),b.ravel())[0,1])
scores={}
for label,a in [('original',mean),('horizontal',mean[:,:,::-1]),('vertical',mean[:,::-1,:]),('both',mean[:,::-1,::-1])]:
 scores[label]=corr(expected[valid],a[valid])
perm=[0,2,1,3,5,4,6];flipped=np.zeros_like(mean)
for i in range(120):
 bits=f'{i:07b}';other=int(''.join(bits[j] for j in perm),2);flipped[other]=mean[i,::-1,:]
scores['vertical_with_barcode_reassignment']=corr(expected,flipped)
def assemble(a):
 result=np.zeros((64,25,3))
 for code,d in enumerate(manifest['designs']):
  x=d['quadrant_x']*5;y=d['quadrant_y']*8;c={'R1 R':0,'R1 G':1,'R1 B':2}[d['color']]
  result[y:y+8,x:x+5,c]=a[code]
 return result
canvas=Image.new('RGB',(900,690),'#222222');draw=ImageDraw.Draw(canvas)
for i,(a,label) in enumerate([(expected,'Intended transfer fractions'),(mean,'Measured saved crops'),(flipped,'Vertical + code reassignment')]):
 img=assemble(a);img=np.uint8(np.clip(img/max(img.max(),1e-9),0,1)*255)
 canvas.paste(Image.fromarray(img).resize((280,640)),(i*300+10,30));draw.text((i*300+10,8),label,fill='white')
canvas.save(Path(__file__).parent/'comparison.png')
report=dict(tiles=len(files),accepted=int(counts.sum()),species_counts=counts.tolist(),missing_species=int((counts==0).sum()),species_below_5=int((counts<5).sum()),pixel_correlations=scores,brightness_correlation=corr(expected[valid].mean(axis=(1,2)),mean[valid].mean(axis=(1,2))),max_pose_disagreement_nm=max(pose_errors,default=0),median_crop_bounds=np.median(crop_bounds,axis=0).tolist(),mean_bit_evidence_off=(bit_sums[0]/np.maximum(bit_n[0],1)).tolist(),mean_bit_evidence_on=(bit_sums[1]/np.maximum(bit_n[1],1)).tolist(),limitation='Uses saved candidate crops, which may omit signal recovered by the full-source composite. Correlation is descriptive, not proof of cause.')
(Path(__file__).parent/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='species_counts'},indent=2))

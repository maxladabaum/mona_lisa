"""Audit the combined Echo CSV by physical source/destination wells."""
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'picklist_generator/generated_output/big_mona_lisa/2026-10-20_08-35-56'
OUT=Path(__file__).resolve().parent
manifest=json.loads((P/'image_manifest.json').read_text())
def well(s):return s[0].upper()+str(int(s[1:])).zfill(2)
def sequence(s):return ''.join(s.split()).upper()
stock=list(csv.DictReader((ROOT/'picklist_generator/replacement_sheets/PAINT_R1_replace.csv').open()))
by_well={well(r['Well']):r for r in stock}
by_name={r['Name']:r for r in stock}
assert len(by_well)==len(stock)
combined=list(csv.DictReader((P/'picklist_combined.csv').open()))
by_destination=defaultdict(list)
for r in combined:by_destination[(r['Destination Plate Name'],well(r['Destination Well']))].append(r)
runs={r['run_name']:r for r in manifest['runs']}
report={'designs':0,'site_volume_checks':0,'barcode_bits':0,'destination_wells':0,'image_pixel_placements':0}
reverse_image=np.zeros((64,25,3),dtype=float)
all_individual=[]
for d in manifest['designs']:
 run=runs[d['name']]
 own=list(csv.DictReader((P/d['name']/'picklist.csv').open()));all_individual.extend(own)
 for dest in run['destination_wells']:
  rows=by_destination[(run['destination_plate_name'],well(dest))]
  assert rows
  volumes=Counter()
  for row in rows:
   if row['Source Plate Name']!='SourcePlate7[7]':continue
   item=by_well[well(row['Source Well'])]
   assert sequence(row['Sample Comments'])==sequence(item["Sequence (5' > 3')"])
   volumes[item['Name']]+=float(row['Transfer Volume'])
  for name,fraction in d['site_fractions'].items():
   assert abs(volumes[name]-fraction*d['full_replacement_nl'])<1e-8,(d['name'],dest,name)
   report['site_volume_checks']+=1
  assert set(volumes)<=set(d['site_fractions'])
  decoded=[]
  for g in manifest['geometry']['barcode_bits']:
   values=[volumes[f'U-Apt_{row}-{col}_R1']/d['full_replacement_nl'] for row,col in g['sites']]
   assert len(set(values))==1 and values[0] in (0,1)
   decoded.append(int(values[0]))
  assert decoded==d['bits']
  assert int(''.join(map(str,decoded)),2)==d['barcode_code']
  report['barcode_bits']+=len(decoded)
  for pixel in d['image_sites']:
   x=d['quadrant_x']*5+(6-int(pixel['column']))
   y=d['quadrant_y']*8+(int(pixel['row'][1:])-1)//2
   assert x==pixel['pixel_x'] and y==pixel['pixel_y']
   assert abs(pixel['x_nm']-x*9.9)<1e-8 and abs(pixel['y_nm']-y*5.1)<1e-8
   value=volumes[pixel['name']]/d['full_replacement_nl']
   assert abs(value-pixel['replacement_fraction'])<1e-8
   reverse_image[y,x,{'R1 R':0,'R1 G':1,'R1 B':2}[d['color']]]=value
   report['image_pixel_placements']+=1
  report['destination_wells']+=1
 report['designs']+=1
normalize=lambda rows:Counter(tuple(sorted(row.items())) for row in rows)
assert normalize(combined)==normalize(all_individual)
report['combined_equals_all_120_individual_picklists']=True
report['unique_destination_wells']=len(by_destination)
remote=Path('/Volumes/Toshiba_External_Drive/PAINT/big_mona_lisa/big_mona_lisa/2026-10-20_08-35-56')
report['remote_files_identical']={f:hashlib.sha256((P/f).read_bytes()).hexdigest()==hashlib.sha256((remote/f).read_bytes()).hexdigest() for f in ['image_manifest.json','picklist_combined.csv']}
Image.fromarray(np.uint8(np.round(reverse_image*255))).resize((400,640),Image.Resampling.NEAREST).save(OUT/'reconstructed_from_combined_picklist.png')
(OUT/'manifest_picklist_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

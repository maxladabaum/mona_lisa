"""Independent audit and figures for the interleaved base source sheet."""
from pathlib import Path
import csv, json, sys
from collections import Counter
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SHEETS = ROOT/'picklist_generator/replacement_sheets'
sys.path.insert(0,str(ROOT/'picklist_generator'))
from picklist_core import parse_source1

def read(path):
    with path.open(encoding='utf-8-sig') as f:
        next(f)
        return list(csv.DictReader(f))

old_inventory=read(SHEETS/'book_base_20uM.csv')
oldraw=[r for r in old_inventory if r.get('Usage') != 'Supplemental']
inventory=read(SHEETS/'book_base_20uM_interleaved.csv')
new=[r for r in inventory if r.get('Usage') != 'Supplemental']
supplements=[r for r in inventory if r.get('Usage') == 'Supplemental']
biotin=read(SHEETS/'lowconc_D_biotin_additions.csv')
assert len(inventory)==260 and len(supplements)==len(biotin)==8
assert len({r['Well Position'] for r in inventory})==260
assert {r['Well'] for r in biotin}=={f'P{i}' for i in range(17,25)}
by_well={r['Well Position']:r for r in supplements}
for r in biotin:
    stocked=by_well[r['Well']]
    assert stocked['Sequence Name']==r['Name']
    assert stocked['Sequence']==r['Sequence']
    assert stocked['Original Well Position']==r['Replace Well']
    assert float(stocked['Nominal Concentration µM'])==float(r['Nominal Concentration µM'])==20
old_supplements={r['Well Position']:r for r in old_inventory if r.get('Usage')=='Supplemental'}
assert len(old_supplements)==8
for well,r in by_well.items():
    assert all(old_supplements[well][k]==r[k] for k in ['Sequence Name','Sequence','Original Well Position','Nominal Concentration µM'])
BIOTIN_COLOR='#E8B4DE'
old,_=parse_source1(SHEETS/'book_base_20uM.csv','balanced_rigid')
old={r['Replacement Well']:r for r in old}
wb=openpyxl.load_workbook('/Users/maxladabaum/Downloads/IDT order_06012026_Ref DNA Book_new source plate.xlsx',data_only=True)
# Independently spell out the diagram's destination rows and column offsets.
row_maps={1:'ACEGIKMO',2:'ACEGIKMO',3:'BDFHJLNP',4:'BDFHJLNP'}
ordered={}
for plate,sheet in enumerate(list(wb)[:4],1):
    for well,name,seq,_,previous in sheet.iter_rows(min_row=2,max_col=5,values_only=True):
        if not name: continue
        dest=row_maps[plate]['ABCDEFGH'.index(well[0])]+f'{int(well[1:])+(12 if plate in (2,4) else 0):02d}'
        assert name not in ordered
        ordered[name]=(plate,well,seq,previous,dest)
assert len(new)==len(ordered)==252
assert {r['Sequence Name'] for r in new}==set(ordered)
assert len({r['Well Position'] for r in new})==252
assert {r['Original Well Position'] for r in new}==set(old)
audit=[]
for r in new:
    plate,well,seq,previous,expected=ordered[r['Sequence Name']]
    assert r['Well Position']==expected
    assert ''.join(r['Sequence'].split())==''.join(seq.split())
    assert int(r['IDT Source Plate'])==plate and r['IDT Source Well']==well
    assert r['Previous Quadrant Well']==previous
    before=old[r['Original Well Position']]
    change=''.join(before['Sequence'].split())!=''.join(seq.split())
    audit.append(dict(logical_site=r['Original Well Position'],name=r['Sequence Name'],
        idt_plate=plate,idt_well=well,previous_physical_well=before['Well'],
        workbook_quadrant_well=previous,expected_interleaved_well=expected,
        actual_interleaved_well=r['Well Position'],mapping_pass=True,sequence_matches_order=True,
        sequence_changed_from_old_base=change,old_sequence=before['Sequence'],ordered_sequence=seq))
assert all(a['previous_physical_well']==a['workbook_quadrant_well'] for a in audit)
with (OUT/'well_mapping_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
colors={1:'#B7DFED',2:'#F6A09B',3:'#F5E98B',4:'#B9D99B'}
counts=Counter(a['idt_plate'] for a in audit)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

def axes_plate(ax,title):
    ax.set(xlim=(0,24),ylim=(16,0),aspect='equal',title=title)
    ax.set_xticks([i+.5 for i in range(24)],[f'{i+1:02}' for i in range(24)],fontsize=9)
    ax.set_yticks([i+.5 for i in range(16)],list('ABCDEFGHIJKLMNOP'),fontsize=10)
    ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for y in range(16):
        for x in range(24):ax.add_patch(Rectangle((x,y),1,1,facecolor='#f5f5f5',edgecolor='#cecece',lw=.4))

def put(ax,well,color,label,changed=False):
    x,y=int(well[1:])-1,ord(well[0])-65
    ax.add_patch(Rectangle((x,y),1,1,facecolor=color,edgecolor='#303030' if changed else 'white',lw=1.7 if changed else .5))
    ax.text(x+.5,y+.52,label,ha='center',va='center',fontsize=6.4)
    if changed:ax.plot(x+.88,y+.12,'o',color='#222222',ms=2.4)

def put_biotin(ax, logical=False):
    for r in biotin:
        label=('Bio\n'+r['Replace Well']) if logical else r['Name'].replace('D-Biotin_', '').replace('-', '\n')
        put(ax,r['Well'],BIOTIN_COLOR,label)

def legend(fig,y):
    fig.legend(handles=[Patch(facecolor=colors[p],label=f'IDT plate {p}: {counts[p]} staples') for p in colors]+[Patch(facecolor=BIOTIN_COLOR,label='D-Biotin: 8 added, 20 µM'),Patch(facecolor='#f5f5f5',label='Empty')],loc='lower center',bbox_to_anchor=(.5,y),ncol=3,frameon=False)

figures=[]
f,axs=plt.subplots(2,1,figsize=(17,15))
f.suptitle('Ordered staples: quadrant layout → vertically interleaved layout',fontsize=19,y=.985)
for ax,title in zip(axs,['Before: original physical source wells','After: interleaved physical source wells']):axes_plate(ax,title)
for a in audit:
    label=f"P{a['idt_plate']}\n{a['idt_well']}"
    for ax,k in zip(axs,['previous_physical_well','actual_interleaved_well']):put(ax,a[k],colors[a['idt_plate']],label,a['sequence_changed_from_old_base'])
put_biotin(axs[0])
put_biotin(axs[1])
f.text(.5,.048,'Cell text = IDT plate and 96-well position. Axes = 384-well position. Black dot/border = updated sequence in the order.',ha='center')
f.text(.5,.028,'Both LowConc layouts: 252 active body + 8 D-Biotin staples. D-Biotin stays at P17–P24; pink labels = helix / row.',ha='center')
legend(f,.064);f.subplots_adjust(top=.93,bottom=.12,hspace=.28)
figures.append(('01_before_after_plate_maps',f))

f,ax=plt.subplots(figsize=(18,11))
f.suptitle('Where each original source well moved',fontsize=20,y=.97)
axes_plate(ax,'Axes = NEW interleaved well • text inside each occupied well = OLD physical well')
for a in audit:put(ax,a['actual_interleaved_well'],colors[a['idt_plate']],a['previous_physical_well'],a['sequence_changed_from_old_base'])
put_biotin(ax)
legend(f,.045)
f.text(.5,.025,'Example: old D17 → new G17. Its logical replacement site remains H05; physical addresses and replacement keys are different.',ha='center')
f.subplots_adjust(top=.87,bottom=.18)
figures.append(('02_old_wells_at_new_locations',f))

f,axs=plt.subplots(1,2,figsize=(16,10))
f.suptitle('Independent mapping check against the 96-well IDT workbook',fontsize=19)
for ax,plates,title in zip(axs,[(1,3),(2,4)],['Columns 01–12: plates 1 + 3','Columns 13–24: plates 2 + 4']):
    ax.set(xlim=(-.25,1.25),ylim=(16,0),title=title);ax.axis('off')
    ax.text(0,-.6,'OLD row',ha='center',weight='bold');ax.text(1,-.6,'NEW row',ha='center',weight='bold')
    for i,row in enumerate('ABCDEFGHIJKLMNOP'):
        ax.text(0,i+.5,row,ha='center',va='center');ax.text(1,i+.5,row,ha='center',va='center')
        dst=2*(i%8)+(i//8)
        plate=plates[i//8]
        n=sum(a['idt_plate']==plate and a['previous_physical_well'][0]==row for a in audit)
        ax.plot([.07,.93],[i+.5,dst+.5],color=colors[plate],lw=4)
        ax.text(.17,i+.5+(dst-i)*(.10/.86)-.10,str(n),fontsize=8,color='#333333',bbox=dict(facecolor='white',edgecolor='none',pad=.3))
    ax.text(.5,17,'Numbers on lines = occupied wells moved; columns do not change.',ha='center',fontsize=9)
changed=[a for a in audit if a['sequence_changed_from_old_base']]
moved=sum(a['previous_physical_well']!=a['actual_interleaved_well'] for a in audit)
f.text(.5,.065,f'252 / 252 wells match the diagram and workbook • 252 / 252 sequences match the order\n{moved} staples move wells; {252-moved} retain their address • 244 sequences unchanged; 8 updated sequences • 252 logical replacement sites preserved',ha='center',fontsize=12,linespacing=1.8)
f.subplots_adjust(top=.86,bottom=.2,wspace=.2)
f.text(.5,.025,'This row-remapping audit covers the 252 ordered body staples. The 8 D-Biotin additions occupy P17–P24 and are not remapped.',ha='center',fontsize=11)
figures.append(('03_row_mapping_audit',f))
# Compare explicitly with the original 200 µM source, not just its dilute copy.
with (SHEETS/'book_base.csv').open(encoding='utf-8-sig', newline='') as handle:
    original_raw = list(csv.DictReader(handle))
original_rows, original_plate = parse_source1(SHEETS/'book_base.csv', 'balanced_rigid')
original = {r['Replacement Well']: r for r in original_rows}
assert original == old, 'Original 200 µM and noninterleaved 20 µM mappings differ.'
active_wells = {r['Well'] for r in original_rows}
assert len({r['Well Position'] for r in original_raw}) == len(original_raw)
assert {r['Well Position'] for r in original_raw} == {r['Well Position'] for r in oldraw}
lookup = {r['Well Position']: r for r in oldraw}
assert all(''.join(r['Sequence'].split()) == ''.join(lookup[r['Well Position']]['Sequence'].split()) for r in original_raw)

f, ax = plt.subplots(figsize=(18, 12))
f.suptitle('Original base plate — full CSV inventory', fontsize=20, y=.97)
axes_plate(ax, 'book_base.csv • 200 µM • SourcePlate1[1] • axes = physical wells; cell text = logical replacement reference')
inventory_colors = {'active':'#B7DFED', 'rigid':'#B9D99B', 'flexible':'#CBB8E8', 'superseded':'#F5D69F'}
status_counts = Counter()
for r in original_raw:
    physical = r['Well Position']
    logical = r['Original Well Position'] or physical
    kind = ('flexible' if r['Hinge Type']=='original' else
            'rigid' if r['Hinge Type']=='balanced_rigid' else
            'active' if physical in active_wells else 'superseded')
    status_counts[kind] += 1
    put(ax, physical, inventory_colors[kind], logical)
f.legend(handles=[Patch(facecolor=inventory_colors[k], label=f'{label}: {status_counts[k]}') for k,label in
    [('active','Active non-hinge staples'),('rigid','Active rigid hinges'),('flexible','Flexible hinge alternatives'),('superseded','Superseded original entries')]],
    loc='lower center', bbox_to_anchor=(.5,.075), ncol=2, frameon=False)
f.text(.5,.055, '282 stored sequences; 252 active when rigid hinges are selected. The noninterleaved 20 µM sheet also includes 8 D-Biotin additions at P17–P24.', ha='center')
f.text(.5,.035, 'Flexible hinges and superseded entries remain in the original CSV but are not used in the rigid-hinge picklist or the new IDT order.', ha='center')
f.subplots_adjust(top=.87, bottom=.19)
figures.append(('04_original_base_full_inventory',f))

f, axs = plt.subplots(1, 3, figsize=(25, 9))
f.suptitle('Same replacement references across all three base-plate options', fontsize=22, y=.97)
for ax,title in zip(axs, ['Original base: 200 µM\nbook_base.csv • plate 1',
                         'Dilute original layout: 20 µM\nbook_base_20uM.csv • plate 8',
                         'Dilute interleaved: 20 µM\nbook_base_20uM_interleaved.csv • plate 8']):
    axes_plate(ax,title)
for a in audit:
    for ax, physical in zip(axs, [original[a['logical_site']]['Well'],old[a['logical_site']]['Well'],a['actual_interleaved_well']]):
        put(ax,physical,colors[a['idt_plate']],a['logical_site'],a['sequence_changed_from_old_base'])
put_biotin(axs[1], logical=True)
put_biotin(axs[2], logical=True)
f.text(.5,.265, 'Pink = 8 added D-Biotin staples at P17–P24, 20 µM each, on both LowConc layouts; interleaved inventory = 260 wells.', ha='center', fontsize=13)
f.text(.5,.22, 'Cell text = logical reference matched by PAINT “Replace Well”. Axes = actual transfer well. All panels use rigid hinges.', ha='center', fontsize=13)
f.text(.5,.17, 'The first two layouts and all their active sequences are identical. Interleaving moves 225 of 252 active staples; 27 stay in place.', ha='center', fontsize=13)
f.text(.5,.12, 'Black borders mark the 8 sites whose sequences were updated in the IDT order. The other 244 active sequences are unchanged.', ha='center', fontsize=13)
f.text(.5,.07, 'Example: replacement reference H05 → physical D17 in either original layout → physical G17 in the interleaved layout.', ha='center', fontsize=13)
f.subplots_adjust(top=.77, bottom=.30, left=.03, right=.99, wspace=.13)
figures.append(('05_three_base_plate_options', f))

with (OUT/'three_base_plate_comparison.csv').open('w', newline='') as handle:
    fields = ['logical_replacement_reference','original_200uM_well','original_20uM_well','interleaved_20uM_well','sequence_changed_in_order']
    writer = csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader()
    for a in sorted(audit,key=lambda a:a['logical_site']):
        writer.writerow(dict(zip(fields,[a['logical_site'],original[a['logical_site']]['Well'],old[a['logical_site']]['Well'],a['actual_interleaved_well'],a['sequence_changed_from_old_base']])))

# Explicit supplemental inventory: physical loading wells and logical replacement keys.
body_by_key={r['Original Well Position']:r for r in new}
biotin_audit=[]
for r in biotin:
    body=body_by_key[r['Replace Well']]
    biotin_audit.append(dict(name=r['Name'],source_plate='SourcePlate8[8]',source_well=r['Well'],
        stock_uM=20,paint_p1_well=r['PAINT P1 Well'],logical_replacement_key=r['Replace Well'],
        replaced_body_physical_well=body['Well Position'],sequence=r['Sequence']))
with (OUT/'d_biotin_mapping_audit.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(biotin_audit[0]));writer.writeheader();writer.writerows(biotin_audit)
f,ax=plt.subplots(figsize=(17,10))
f.suptitle('D-Biotin additions — LowConc interleaved base plate',fontsize=20,y=.96)
ax.axis('off')
columns=['D-Biotin site','Load into plate 8','Stock','PAINT P1 reference','Logical replace key','Body well replaced']
values=[[r['name'].replace('D-Biotin_',''),r['source_well'],'20 µM',r['paint_p1_well'],
         r['logical_replacement_key'],r['replaced_body_physical_well']] for r in biotin_audit]
table=ax.table(cellText=values,colLabels=columns,cellLoc='center',bbox=[0,.25,1,.68])
table.auto_set_font_size(False);table.set_fontsize(12)
for (row,col),cell in table.get_celld().items():
    cell.set_facecolor(BIOTIN_COLOR if row==0 or col==1 else '#f5f5f5')
    if row==0:cell.set_text_props(weight='bold')
f.text(.5,.20,'SourcePlate8[8] • book_base_20uM_interleaved.csv • 252 body staples + 8 supplemental D-Biotin staples',ha='center',fontsize=13)
f.text(.5,.15,'D-Biotin is selected explicitly: it replaces the corresponding body staple; both remain inventoried on the plate.',ha='center',fontsize=12)
f.text(.5,.10,'PAINT P1 wells identify the original extension sequences. Transfer sources are now plate 8, P17–P24, at 20 µM.',ha='center',fontsize=12)
f.text(.5,.05,'H15-01 and H15-12 retain the PAINT P1 extension sequences; their interleaved body counterparts were updated in the IDT order.',ha='center',fontsize=11)
figures.append(('06_d_biotin_additions',f))

with PdfPages(OUT/'base_plate_mapping_review.pdf') as pdf:
    for name,f in figures:
        f.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');pdf.savefig(f);plt.close(f)
summary={'ordered_staples':len(audit),'supplemental_d_biotin_staples':len(biotin),'total_interleaved_stocked_wells':len(inventory),'d_biotin_stock_uM':20,'d_biotin_mapping':biotin_audit,'plate_counts':dict(counts),'moved':moved,'unchanged_well':252-moved,'changed_sequences':[{k:a[k] for k in ('logical_site','previous_physical_well','actual_interleaved_well','name')} for a in changed],'all_checks_passed':True}
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

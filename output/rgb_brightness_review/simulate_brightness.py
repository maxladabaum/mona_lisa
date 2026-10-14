"""Compare signal errors on saved RGB designs without altering synthesis fractions.

The copied renderer weights represent effective brightness only. Random errors are
fixed per design/site (persistent heterogeneity), not per molecule or photon noise.
"""
from pathlib import Path
import argparse, copy, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'picklist_generator'))
from alignment_reconstruction import AlignmentModel, AlignmentSettings

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'picklist_generator/generated_output/2026-10-13_15-07-35/image_manifest.json')
    parser.add_argument('--from-level',type=float,default=.4)
    parser.add_argument('--to-level',type=float,default=.5)
    parser.add_argument('--random-sd',type=float,default=.1)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    assert all(np.isfinite(x) for x in [args.from_level,args.to_level,args.random_sd])
    assert 0<=args.from_level<=1 and 0<=args.to_level<=1 and args.random_sd>=0
    manifest=json.loads(args.manifest.read_text());designs=manifest['designs']
    meta=manifest['alignment_simulation'];out=Path(__file__).resolve().parent
    model=AlignmentModel(designs,AlignmentSettings(**meta['alignment']),meta['fwhm_nm'],meta['exposure'],
        seed=meta['model_seed'],**manifest['color_processing'])
    rng=np.random.default_rng(args.seed)
    noise=[rng.normal(0,args.random_sd,len(d['image_sites'])) for d in designs]
    cases=[('Nominal',False,False),('Systematic bias',True,False),('Random variation',False,True),('Bias + random variation',True,True)]
    results=[];report={'manifest':str(args.manifest),'systematic_mapping':[args.from_level,args.to_level],
        'random_sd_percentage_points':100*args.random_sd,'seed':args.seed,
        'random_model':'Independent additive Gaussian error, fixed per design/site; nominal zero stays off; effective brightness clipped to [0,1]. Same noise draws in both random cases.',
        'scope':'Image sites only; barcode and fiducial signals excluded by the image renderer. No synthesis or picklist changes.',
        'render_settings':meta,'cases':[]}
    for label,bias,random in cases:
        effective=copy.deepcopy(designs);affected=0;clipped=0
        for di,d in enumerate(effective):
            for si,s in enumerate(d['image_sites']):
                nominal=s['replacement_fraction'];value=nominal
                if bias and np.isclose(nominal,args.from_level):value=args.to_level;affected+=1
                if random and nominal>0:value+=noise[di][si]
                if value<0 or value>1:clipped+=1
                s['replacement_fraction']=float(np.clip(value,0,1))
        result=model.expected(effective);results.append(result)
        result.rgb.save(out/(label.lower().replace(' ','_').replace('+','and')+'.png'))
        nominal_density=results[0].density
        report['cases'].append({'name':label,'systematically_changed_sites':affected,'clipped_site_weights':clipped,
            'integrated_signal_change_percent':100*(result.density.sum()/nominal_density.sum()-1),
            'display_rmse_vs_nominal':float(np.sqrt(np.mean((np.asarray(result.rgb,dtype=float)/255-np.asarray(results[0].rgb,dtype=float)/255)**2)))})
    fig,axs=plt.subplots(2,4,figsize=(16,10))
    vmax=max(float(np.abs(r.density-results[0].density).max()*model.gain) for r in results)
    for col,((label,_,_),result) in enumerate(zip(cases,results)):
        axs[0,col].imshow(result.rgb);axs[0,col].set_title(label);axs[0,col].axis('off')
        change=np.max(np.abs(result.density-results[0].density)*model.gain,axis=2)
        heat=axs[1,col].imshow(change,cmap='magma',vmin=0,vmax=vmax or 1);axs[1,col].axis('off')
        axs[1,col].set_title('Absolute signal change',fontsize=11)
    cax=fig.add_axes([.92,.18,.012,.27])
    fig.colorbar(heat,cax=cax,label='Signal change before display clipping')
    fig.suptitle(f'Brightness sensitivity — saved 12 RGB designs\nSystematic: {args.from_level:.0%} → {args.to_level:.0%}; random SD: {args.random_sd*100:g} percentage points',fontsize=16)
    fig.text(.5,.025,'Shared exposure and alignment model (3 nm X/Y, 3° angle SD). Random error persists per site; zero sites stay off.\nBrightness bounded to 0–100%; clipping can bias the mean near endpoints. Picklists and replacement percentages are unchanged.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.88,bottom=.12,left=.035,right=.89,hspace=.18,wspace=.2)
    fig.savefig(out/'brightness_comparison.png',dpi=180)
    fig.savefig(out/'brightness_comparison.pdf')
    plt.close(fig)
    (out/'simulation_details.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['cases'],indent=2))

if __name__=='__main__':main()

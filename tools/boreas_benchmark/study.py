"""Audited multi-seed aggregation and ranking stability; no GPU jobs."""
from __future__ import annotations
import argparse, copy, csv, json
from itertools import combinations
from pathlib import Path
import numpy as np
try:
    from .evaluate import CLASSES
    from .extended import VERSION, sha256_file
except ImportError:
    from evaluate import CLASSES
    from extended import VERSION, sha256_file


def ranks(values):
    return {name:1+sum(v>value for v in values.values())+(sum(v==value for v in values.values())-1)/2
            for name,value in values.items()}


def kendall_tau_b(first,second):
    common=sorted(set(first)&set(second));c=d=t=u=0
    for a,b in combinations(common,2):
        x,y=np.sign(first[a]-first[b]),np.sign(second[a]-second[b])
        if x==0 and y==0:continue
        if x==0:t+=1
        elif y==0:u+=1
        elif x==y:c+=1
        else:d+=1
    denominator=np.sqrt((c+d+t)*(c+d+u))
    return {'tau_b':float((c-d)/denominator) if denominator else None,'models':common,
            'n_models':len(common),'concordant_pairs':c,'discordant_pairs':d,
            'ties_first_only':t,'ties_second_only':u}


def ranking_analysis(scores):
    out={'rankings':{k:ranks(v) for k,v in scores.items()},'scores':scores,'comparisons':{}}
    for condition,values in scores.items():
        suffix=next(('_'+c for c in CLASSES if condition.endswith('_'+c)),'')
        reference='beam128'+suffix
        if condition!=reference and reference in scores:
            out['comparisons'][reference+'_vs_'+condition]=kendall_tau_b(scores[reference],values)
    return out


def mean_std(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,
            'std':float(a.std(ddof=1)) if len(a)>1 else None}


def read_training_manifest(folder,specified=None):
    files=[Path(specified)] if specified else sorted(folder.glob('run_config_*.json'))
    if not files:raise ValueError(f'Missing training metadata in {folder}; do not infer seed from tag')
    manifests=[json.loads(p.read_text()) for p in files]
    identities={(m.get('seed'),m.get('training_precision'),m.get('batch_size_per_gpu'),
                 m.get('total_gpus'),m.get('epochs'),json.dumps(m.get('effective_config'),sort_keys=True)) for m in manifests}
    if len(identities)!=1:raise ValueError(f'Mixed runs in {folder}; provide explicit metadata CSV path')
    return manifests[-1],files[-1]


def audit_run(root,row,allow_legacy_splits=False):
    model,tag,seed=row['model'],row['tag'],int(row['seed']);folder=root/model/tag
    specified=row.get('metadata','').strip()
    if specified and not Path(specified).is_absolute():specified=str(folder/specified)
    metadata,metadata_path=read_training_manifest(folder,specified or None)
    if metadata.get('seed')!=seed:raise ValueError(f'CSV seed disagrees with training metadata: {model}/{tag}')
    args=metadata.get('arguments',{})
    if args.get('boreas_beams',128)!=128 or args.get('pretrained_model') or args.get('ckpt'):
        raise ValueError('Primary study requires dense training without external pretrained weights')
    config=metadata['effective_config'];data=config['DATA_CONFIG'];dataset=Path(data['DATA_PATH'])
    hashes=metadata.get('split_sha256',{}).copy(); warnings=[]
    for split in ('train','val'):
        current=sha256_file(dataset/'ImageSets'/(split+'.txt'))
        if split not in hashes and allow_legacy_splits:
            hashes[split]=current
            warnings.append(f'{model}/{tag}: {split} hash recorded at analysis time only; historical split identity assumed')
        if hashes.get(split)!=current:
            raise ValueError(f'Training split provenance missing/changed in {metadata_path}; audit legacy run first')
    shared={'precision':metadata['training_precision'],'batch':metadata['batch_size_per_gpu']*metadata['total_gpus'],
            'epochs':metadata['epochs'],'workers':args.get('workers'),'splits':hashes,'classes':config['CLASS_NAMES'],
            'range':data['POINT_CLOUD_RANGE'],'augmentation':data['DATA_AUGMENTOR'],
            'features':data['POINT_FEATURE_ENCODING']}
    training_model=copy.deepcopy(config['MODEL']);training_model.pop('POST_PROCESSING',None)
    training_model.get('DENSE_HEAD',{}).pop('POST_PROCESSING',None)
    training_model['optimizer']=copy.deepcopy(config['OPTIMIZATION'])
    training_model['optimizer'].pop('BATCH_SIZE_PER_GPU',None)
    values={};scores={};density=[];calibration=[]
    truth_hash=inference_common=model_hash=checkpoint_hash=None
    for beam in (128,64,32,16):
        suffix='' if beam==128 else f'_{beam}'
        ap=json.loads((folder/f'range_ap{suffix}.json').read_text())
        extra=json.loads((folder/f'extended_{beam}.json').read_text());inf=extra.get('inference') or {}
        if ap.get('protocol_version')!='local_ap40_v1' or extra.get('protocol_version')!=VERSION:
            raise ValueError('Stale metric protocol; rerun run_metrics.sh')
        if not extra.get('verified_inference') or not inf.get('checkpoint_sha256'):
            raise ValueError('Study requires verified inference and checkpoint provenance')
        if ap.get('prediction_sha256')!=inf.get('prediction_sha256') or ap.get('truth_sha256')!=extra['truth_sha256']:
            raise ValueError('AP/calibration content hashes differ; regenerate primary reports')
        if extra['beam']!=beam or extra['operating_score']!=.1:raise ValueError('Incompatible beam/score')
        if Path(ap['prediction_file']).resolve()!=Path(extra['prediction_file']).resolve():
            raise ValueError('AP and calibration prediction files differ')
        condition=(inf['precision'],inf.get('score_cutoff'),inf.get('nms_pre_maxsize'),inf.get('nms_post_maxsize'),
                   inf.get('tf32_matmul'),inf.get('tf32_cudnn'),inf.get('torch_version'),inf.get('cuda_version'),extra['calibration_bins'])
        if inf.get('batch_size') != 1:raise ValueError('Accuracy inference must use batch one')
        if condition[:4]!=('FP32',.001,4096,500):raise ValueError('Use FP32, cutoff .001, caps 4096/500')
        if truth_hash is not None and (extra['truth_sha256']!=truth_hash or condition!=inference_common
                                      or inf['model_config_sha256']!=model_hash or inf['checkpoint_sha256']!=checkpoint_hash):
            raise ValueError('Beam conditions must share GT, weights and inference config')
        truth_hash,inference_common=extra['truth_sha256'],condition
        model_hash,checkpoint_hash=inf['model_config_sha256'],inf['checkpoint_sha256']
        primary=ap['ranges']['all']
        if any(primary['classes'][c]['AP40'] is None for c in CLASSES):raise ValueError('Missing class support')
        values[f'mAP40_beam{beam}']=primary['mAP40'];scores[f'beam{beam}']=primary['mAP40']
        for name in CLASSES:
            score=primary['classes'][name]['AP40'];values[f'{name}_AP40_beam{beam}']=score;scores[f'beam{beam}_{name}']=score
            entry=extra['classes'][name]
            for metric in ('ECE','Brier'):values[f'{name}_{metric}_beam{beam}']=entry['calibration'][metric]
            calibration.append({'model':model,'seed':seed,'beam':beam,'class':name,
                                **{k:v for k,v in entry['calibration'].items() if k!='bins'},
                                'recall':entry['operating_point']['recall']})
            for mode in ('density_fixed128','density_actual'):
                for band,groups in entry[mode].items():
                    for group,counts in groups.items():
                        density.append({'model':model,'seed':seed,'beam':beam,'class':name,
                                        'grouping':mode,'range':band,'points':group,**counts})
        if beam==128:
            for band,entry in ap['ranges'].items():
                if band!='all':
                    scores[f'range_{band}']=entry['mAP40']
                    for name in CLASSES:scores[f'range_{band}_{name}']=entry['classes'][name]['AP40']
    profile=json.loads((folder/'profile.json').read_text())
    if profile['precision']!='FP32' or Path(profile['data_path']).resolve()!=dataset.resolve():
        raise ValueError('Profile must use dense input and FP32')
    if sha256_file(profile['checkpoint'])!=checkpoint_hash:raise ValueError('Profile/accuracy weights differ')
    if (profile.get('tf32_matmul'),profile.get('tf32_cudnn'))!=inference_common[4:6]:raise ValueError('TF32 mismatch')
    values.update(mAP40=values['mAP40_beam128'],FPS=profile['fps_from_mean_end_to_end'],
                  GPU_MiB=profile['peak_cuda_allocated_mib'],parameters=profile['parameters'])
    for stat in ('mean','p50','p95'):values[f'latency_{stat}_ms']=profile['end_to_end'][stat+'_ms']
    for beam in (64,32,16):
        values[f'drop_128_to_{beam}_pp']=values['mAP40_beam128']-values[f'mAP40_beam{beam}']
        values[f'retention_{beam}_percent']=100*values[f'mAP40_beam{beam}']/values['mAP40_beam128'] if values['mAP40_beam128'] else None
    return {'model':model,'tag':tag,'seed':seed,'values':values,'scores':scores,
            'training_key':json.dumps(shared,sort_keys=True),'model_key':json.dumps(training_model,sort_keys=True),
            'truth_hash':truth_hash,'inference_key':inference_common,
            'profile_key':tuple(profile.get(k) for k in ('gpu','torch_version','cuda_version','frames','warmup_frames','batch_size')),
            'checkpoint_hash':checkpoint_hash,'metadata':str(metadata_path),'warnings':warnings,'density':density,'calibration':calibration}


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def aggregate_long(rows, group_keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, subset in groups.items():
        for metric in metrics:
            output.append({**dict(zip(group_keys,key)), 'metric':metric,
                           **mean_std([row.get(metric) for row in subset])})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True,help='CSV: model,tag,seed,metadata(optional)')
    parser.add_argument('--root',type=Path,default=Path('../output/boreas_models'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--allow-partial',action='store_true',help='Explicit incomplete exploratory study')
    parser.add_argument('--allow-legacy-splits',action='store_true',help='Explicitly assume unchanged splits for older runs with verified seed metadata; warnings retained')
    parser.add_argument('--out',type=Path,default=Path('../output/boreas_summary/multiseed'))
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Duplicate requested seeds')
    with args.runs.open() as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty run manifest')
    runs=[];missing=[];keys=set()
    for row in rows:
        key=(row['model'],int(row['seed']))
        if key in keys:raise ValueError(f'Duplicate model/seed: {key}')
        keys.add(key)
        if key[1] not in args.seeds:raise ValueError('Unrequested seed in CSV')
        try:runs.append(audit_run(args.root,row,args.allow_legacy_splits))
        except FileNotFoundError as error:
            if not args.allow_partial:raise
            missing.append(str(error))
    if not runs:raise ValueError('No complete audited run')
    if len({(r['training_key'],r['truth_hash'],r['inference_key'],r['profile_key']) for r in runs})!=1:
        raise ValueError('Runs differ in training, validation or inference/profile conditions')
    models=sorted({row['model'] for row in rows})
    missing_combinations=[(m,s) for m in models for s in args.seeds if (m,s) not in {(r['model'],r['seed']) for r in runs}]
    if missing_combinations and not args.allow_partial:raise ValueError(f'Incomplete seed matrix: {missing_combinations}')
    for model in models:
        subset=[r for r in runs if r['model']==model]
        if len({r['checkpoint_hash'] for r in subset})!=len(subset):
            raise ValueError(f'Same checkpoint counted under multiple training seeds: {model}')
        if len({r['model_key'] for r in runs if r['model']==model})>1:raise ValueError(f'Changed config across seeds: {model}')
    aggregate=[];mean_rows=[]
    for model in models:
        subset=[r for r in runs if r['model']==model]
        if not subset:continue
        summary={'model':model}
        for metric in sorted(set(k for r in subset for k in r['values'])):
            stat=mean_std([r['values'].get(metric) for r in subset])
            aggregate.append({'model':model,'metric':metric,**stat})
            summary[metric]=stat['mean'];summary[metric+'_std']=stat['std']
        mean_rows.append(summary)
    seed_rankings={}
    for seed in args.seeds:
        scores={}
        for run in runs:
            if run['seed']==seed:
                for condition,value in run['scores'].items():
                    if value is not None:scores.setdefault(condition,{})[run['model']]=value
        if scores:seed_rankings[str(seed)]=ranking_analysis(scores)
    across_seeds={}
    for s1,s2 in combinations(args.seeds,2):
        a=seed_rankings.get(str(s1),{}).get('scores',{}).get('beam128',{})
        b=seed_rankings.get(str(s2),{}).get('scores',{}).get('beam128',{})
        across_seeds[f'{s1}_vs_{s2}']=kendall_tau_b(a,b)
        for name in CLASSES:
            a=seed_rankings.get(str(s1),{}).get('scores',{}).get('beam128_'+name,{})
            b=seed_rankings.get(str(s2),{}).get('scores',{}).get('beam128_'+name,{})
            across_seeds[f'{s1}_vs_{s2}_{name}']=kendall_tau_b(a,b)
    wins={m:0 for m in models}
    for report in seed_rankings.values():
        ranking=report['rankings'].get('beam128',{})
        if ranking:
            for model,rank in ranking.items():
                if rank==min(ranking.values()):wins[model]+=1
    try:
        from .summarize import frontier,svg_plot
    except ImportError:
        from summarize import frontier,svg_plot
    frequencies={m:{'speed':0,'memory':0,'evaluated_seeds':0} for m in models}
    for seed in args.seeds:
        subset=[{'model':r['model'],**r['values']} for r in runs if r['seed']==seed]
        if not subset:continue
        speed=frontier(subset,'FPS',True);memory=frontier(subset,'GPU_MiB',False)
        for row in subset:
            p=frequencies[row['model']];p['evaluated_seeds']+=1
            p['speed']+=int(row['model'] in speed);p['memory']+=int(row['model'] in memory)
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'seed_metrics.csv',[{'model':r['model'],'seed':r['seed'],'tag':r['tag'],**r['values']} for r in runs])
    write_csv(args.out/'mean_std.csv',aggregate)
    write_csv(args.out/'density_recall.csv',[row for r in runs for row in r['density']])
    calibration_rows=[row for r in runs for row in r['calibration']]
    density_rows=[row for r in runs for row in r['density']]
    write_csv(args.out/'calibration.csv',calibration_rows)
    write_csv(args.out/'density_recall_mean_std.csv',aggregate_long(density_rows,
              ['model','beam','class','grouping','range','points'],['recall','gt','TP','FN']))
    write_csv(args.out/'calibration_mean_std.csv',aggregate_long(calibration_rows,
              ['model','beam','class'],['ECE','Brier','recall','detections']))
    report={'complete_seed_matrix':not missing_combinations,'requested_seeds':args.seeds,
            'provenance_warnings':[w for r in runs for w in r['warnings']],'missing':missing,'missing_model_seeds':missing_combinations,'per_seed':seed_rankings,
            'across_seeds':across_seeds,'top_rank_frequency':wins,'pareto_frequency':frequencies,
            'note':'Top-rank ties all count as wins. Tau uses explicit model intersection. Partial studies are exploratory. Mean-frontier is descriptive, not a confidence bound.'}
    (args.out/'ranking_stability.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_csv(args.out/'pareto_mean.csv',mean_rows)
    svg_plot(mean_rows,'FPS','Mean serial FPS across training seeds',True,args.out/'pareto_mean_fps.svg')
    svg_plot(mean_rows,'GPU_MiB','Mean allocated MiB across training seeds',False,args.out/'pareto_mean_memory.svg')
    print(f'Wrote {len(runs)} audited runs; complete seed matrix: {not missing_combinations}')

if __name__=='__main__':main()

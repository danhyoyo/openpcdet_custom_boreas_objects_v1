"""Meaningful CPU regression tests for frozen-checkpoint extended evaluation."""
import json, pickle, tempfile, unittest
from pathlib import Path
import numpy as np
from extended import (count_points_in_boxes, calibration_metrics, match_class, density_recall,
                      density_counts, load_records, truth_fingerprint, sha256_file, analyze)
from study import ranks, kendall_tau_b, mean_std, audit_run, read_training_manifest

BOX=np.array([10,0,0,4,2,2,0.],dtype=float)
def truth_record(names,boxes):
    return {'name':np.array(names),'gt_boxes_lidar':np.asarray(boxes).reshape(-1,7)}
def pred_record(names,boxes,scores,frame='f'):
    return {'name':np.array(names),'boxes_lidar':np.asarray(boxes).reshape(-1,7),'score':np.array(scores),'frame_id':frame}

class GeometryCalibrationTests(unittest.TestCase):
    def test_rotated_boxes_and_faces(self):
        b=BOX.copy();b[6]=np.pi/2
        points=np.array([[10,0,0],[10,1.9,0],[10,2,1],[11.1,0,0],[10,0,1.1],[13,0,0]])
        self.assertEqual(count_points_in_boxes(points,[b]),[3])
        self.assertEqual(count_points_in_boxes(points,[BOX]),[2])
        self.assertEqual(count_points_in_boxes(np.empty((0,3)),[b]),[0])

    def test_duplicates_misc_wrong_class_and_threshold(self):
        other=BOX.copy();other[0]=30
        truth={'f':truth_record(['Car','Misc'],[BOX,other])}
        pred={'f':pred_record(['Car','Car','Car','Pedestrian'],[BOX,BOX,other,BOX],[.9,.8,.99,.7])}
        outcomes,matched,ignored=match_class(pred,truth,'Car',.7)
        self.assertEqual([o['correct'] for o in outcomes],[1,0]);self.assertEqual(ignored,1)
        self.assertEqual(matched,{('f',0):.9})
        ped,_,_=match_class(pred,truth,'Pedestrian',.5)
        self.assertEqual(ped[0]['correct'],0)
        groups=density_recall(truth,matched,{'f':[2,20]},'Car',.95)
        self.assertEqual(groups['1-5']['FN'],1)

    def test_calibration_bin_edges_and_empty(self):
        values=[{'score':0.,'correct':0},{'score':1.,'correct':1}]
        result=calibration_metrics(values,2)
        self.assertEqual(result['ECE'],0);self.assertEqual(result['Brier'],0)
        self.assertEqual([b['count'] for b in result['bins']],[1,1])
        result=calibration_metrics([{'score':.9,'correct':0}],15)
        self.assertAlmostEqual(result['ECE'],.9);self.assertAlmostEqual(result['Brier'],.81)
        self.assertIsNone(calibration_metrics([])['ECE'])
        with self.assertRaises(ValueError):calibration_metrics([],0)

    def test_fixed_density_cohort_does_not_move_after_sparsification(self):
        truth={'f':truth_record(['Car'],[BOX])};pred={'f':pred_record([],[],[])}
        report=analyze(truth,pred,{'f':[25]},{'f':[0]})['Car']
        self.assertEqual(report['density_fixed128']['all']['21-50']['gt'],1)
        self.assertEqual(report['density_actual']['all']['0']['gt'],1)
        self.assertEqual(report['density_fixed128']['all']['21-50']['recall'],0)
        self.assertIsNone(report['calibration']['ECE'])

    def test_cache_invalidation_and_ground_truth_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'128';(data/'points').mkdir(parents=True)
            infos=[{'point_cloud':{'lidar_idx':'f'},'annos':truth_record(['Car'],[BOX])}]
            points=data/'points/f.npy';np.save(points,np.array([[10.,0,0,1]]))
            cache=root/'cache'
            self.assertEqual(density_counts(infos,data,cache)['f'],[1])
            self.assertEqual(density_counts(infos,data,cache)['f'],[1])
            np.save(points,np.array([[10.,0,0,1],[10.,0,0,2]]))
            self.assertEqual(density_counts(infos,data,cache)['f'],[2])
            self.assertEqual(len(list(cache.glob('*.json'))),2)
            with self.assertRaises(ValueError):truth_fingerprint(infos+infos)
            (root/'info.pkl').write_bytes(pickle.dumps(infos))
            bad=pred_record(['Car'],[BOX],[2.0]);(root/'pred.pkl').write_bytes(pickle.dumps([bad]))
            with self.assertRaises(ValueError):load_records(root/'pred.pkl',root/'info.pkl')

class RankingTests(unittest.TestCase):
    def test_rank_ties_and_reversal(self):
        self.assertEqual(ranks({'a':3,'b':3,'c':1}),{'a':1.5,'b':1.5,'c':3})
        self.assertEqual(kendall_tau_b({'a':3,'b':2,'c':1},{'a':1,'b':2,'c':3})['tau_b'],-1)
        self.assertAlmostEqual(kendall_tau_b({'a':3,'b':3,'c':1},{'a':3,'b':2,'c':1})['tau_b'],2/np.sqrt(6))
        self.assertIsNone(kendall_tau_b({'a':1},{'a':2})['tau_b'])
        self.assertIsNone(kendall_tau_b({'a':1,'b':1},{'a':2,'b':2})['tau_b'])
        stat=mean_std([1,3,None]);self.assertEqual(stat['n'],2)
        self.assertEqual(stat['mean'],2);self.assertAlmostEqual(stat['std'],np.sqrt(2))
        self.assertIsNone(mean_std([1])['std'])

    def test_never_infer_legacy_training_seed(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'seed42_beam128';folder.mkdir()
            with self.assertRaises(ValueError):read_training_manifest(folder)
            (folder/'run_config_1.json').write_text(json.dumps({'seed':None}))
            with self.assertRaises(ValueError):audit_run(Path(d),{'model':'','tag':'seed42_beam128','seed':42})


class StudyIntegrationTests(unittest.TestCase):
    def test_complete_three_seed_study_and_checkpoint_reuse_rejection(self):
        import sys
        from unittest.mock import patch
        from study import main
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset=root/'data';(dataset/'ImageSets').mkdir(parents=True)
            for split in ('train','val'):(dataset/'ImageSets'/(split+'.txt')).write_text('f\n')
            csv_path=root/'runs.csv';lines=['model,tag,seed,metadata']
            for model in ('a','b'):
                for seed in (42,43,44):
                    tag=f'seed{seed}_beam128';folder=root/model/tag;folder.mkdir(parents=True)
                    checkpoint=folder/'checkpoint.pth';checkpoint.write_bytes(f'{model}-{seed}'.encode())
                    training={'seed':seed,'training_precision':'AMP','batch_size_per_gpu':4,'total_gpus':1,'epochs':80,
                              'arguments':{'boreas_beams':128},'split_sha256':{s:sha256_file(dataset/'ImageSets'/(s+'.txt')) for s in ('train','val')},
                              'effective_config':{'CLASS_NAMES':['Car','Pedestrian','Cyclist'],'OPTIMIZATION':{'LR':.003},
                                                  'MODEL':{'NAME':model},'DATA_CONFIG':{'DATA_PATH':str(dataset),'POINT_CLOUD_RANGE':[-75.2,-75.2,-3,75.2,75.2,5],
                                                                                     'DATA_AUGMENTOR':{},'POINT_FEATURE_ENCODING':{}}}}
                    (folder/'run_config_1.json').write_text(json.dumps(training))
                    for beam in (128,64,32,16):
                        ap={'protocol_version':'local_ap40_v1','prediction_file':str(folder/f'pred{beam}.pkl'),
                            'prediction_sha256':'predhash','truth_sha256':'truth',
                            'ranges':{'all':{'mAP40':80+seed-42-(model=='b'),
                                             'classes':{c:{'AP40':80+seed-42} for c in ('Car','Pedestrian','Cyclist')}}}}
                        (folder/('range_ap.json' if beam==128 else f'range_ap_{beam}.json')).write_text(json.dumps(ap))
                        inf={'checkpoint_sha256':sha256_file(checkpoint),'prediction_sha256':'predhash','precision':'FP32','batch_size':1,'score_cutoff':.001,
                             'nms_pre_maxsize':4096,'nms_post_maxsize':500,'tf32_matmul':False,'tf32_cudnn':True,
                             'torch_version':'test','cuda_version':'test','model_config_sha256':model}
                        cal={'ECE':.1,'Brier':.2,'detections':1,'bins':[]}
                        extra={'protocol_version':'density_calibration_v1','verified_inference':True,'beam':beam,'operating_score':.1,
                               'prediction_file':ap['prediction_file'],'truth_sha256':'truth','calibration_bins':15,'inference':inf,
                               'classes':{c:{'calibration':cal,'operating_point':{'recall':.8},'density_fixed128':{},'density_actual':{}}
                                          for c in ('Car','Pedestrian','Cyclist')}}
                        (folder/f'extended_{beam}.json').write_text(json.dumps(extra))
                    profile={'precision':'FP32','data_path':str(dataset),'checkpoint':str(checkpoint),'tf32_matmul':False,'tf32_cudnn':True,
                             'fps_from_mean_end_to_end':20,'peak_cuda_allocated_mib':100,'parameters':1000,
                             'end_to_end':{'mean_ms':50,'p50_ms':45,'p95_ms':60},'gpu':'test','torch_version':'test','cuda_version':'test',
                             'frames':200,'warmup_frames':20,'batch_size':1}
                    (folder/'profile.json').write_text(json.dumps(profile));lines.append(f'{model},{tag},{seed},')
            csv_path.write_text('\n'.join(lines)+'\n');out=root/'summary'
            argv=['study.py','--runs',str(csv_path),'--root',str(root),'--out',str(out)]
            with patch.object(sys,'argv',argv):main()
            report=json.loads((out/'ranking_stability.json').read_text())
            self.assertTrue(report['complete_seed_matrix']);self.assertEqual(report['top_rank_frequency']['a'],3)
            self.assertTrue((out/'pareto_mean_fps.svg').exists())
            # Copied weights must not count as a distinct seed.
            first=root/'a/seed42_beam128/checkpoint.pth';second=root/'a/seed43_beam128/checkpoint.pth'
            second.write_bytes(first.read_bytes())
            for beam in (128,64,32,16):
                p=root/'a/seed43_beam128'/f'extended_{beam}.json';e=json.loads(p.read_text())
                e['inference']['checkpoint_sha256']=sha256_file(second);p.write_text(json.dumps(e))
            with patch.object(sys,'argv',argv), self.assertRaisesRegex(ValueError,'Same checkpoint'):main()

if __name__=='__main__':unittest.main()

"""Regression checks for incremental Excel output (no GPU or detector runtime)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('export_excel', Path(__file__).with_name('export_excel.py'))
x = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(x)


class ExcelExportTests(unittest.TestCase):
    def fixture(self, root, model='pointpillar', tag='seed666_beam128', seed=666, legacy=True):
        folder = root/model/tag
        folder.mkdir(parents=True)
        ck = folder/'ckpt/checkpoint_epoch_80.pth'
        ck.parent.mkdir();ck.write_bytes(b'fixture checkpoint')
        ckhash=x.file_hash(ck)
        if legacy:
            (folder/'legacy_run_rename.json').write_text(json.dumps({'historical_training_seed':seed,'seed_basis':'Historical fixed seed recorded in README'}))
            (folder/'train_20261007-000000.log').write_text('INFO Training with a single process\nINFO batch_size 4\nINFO workers 2\nINFO epochs 80\nINFO use_amp True\nINFO fix_random_seed True\nINFO End training\n')
        else:
            (folder/'run_config_1.json').write_text(json.dumps({'seed':seed,'training_precision':'AMP','batch_size_per_gpu':4,'total_gpus':1,'epochs':80,'arguments':{'workers':2}}))
        dense=root/'dense'
        for beam in x.BEAMS:
            r=folder/'eval/epoch_80/val'/f'beam{beam}';r.mkdir(parents=True)
            pred=r/'result.pkl';pred.write_bytes(f'pred{beam}'.encode());ph=x.file_hash(pred)
            inf={'checkpoint':str(ck),'checkpoint_sha256':ckhash,'model_config_sha256':'config','prediction_sha256':ph,'precision':'FP32','batch_size':1,'score_cutoff':.001,'nms_pre_maxsize':4096,'nms_post_maxsize':500,'evaluation_seed':42,'tf32_matmul':False,'tf32_cudnn':True,'torch_version':'fixture','cuda_version':'fixture'}
            op={'score_threshold':.1,'TP':1,'FP':1,'FN':1,'precision':.5,'recall':.5,'F1':.5,'gt':2}
            item={'AP40':float(beam)/2,'gt':2,'predictions':2,'ignored_misc':0,'operating_point':op}
            ap={'protocol_version':'local_ap40_v1','prediction_file':str(pred),'prediction_sha256':ph,'truth_sha256':'truth','ranges':{'all':{'mAP40':float(beam)/2,'classes':{c:item for c in x.CLASSES}}}}
            (r/'range_ap.json').write_text(json.dumps(ap))
            cal={'ECE':0.,'Brier':.25,'detections':2,'mean_confidence':.5,'empirical_precision':.5,'bins':[{'lower':0,'upper':.5,'count':0,'mean_confidence':None,'empirical_precision':None},{'lower':.5,'upper':1,'count':2,'mean_confidence':.5,'empirical_precision':.5}]}
            counts={'all':{'0':{'gt':0,'TP':0,'FN':0,'recall':None},'1-5':{'gt':2,'TP':1,'FN':1,'recall':.5}}}
            entry={'operating_point':op,'calibration':cal,'density_fixed128':counts,'density_actual':counts}
            extra={'protocol_version':'density_calibration_v1','beam':beam,'truth_sha256':'truth','prediction_file':str(pred),'verified_inference':True,'inference':inf,'calibration_bins':2,'operating_score':.1,'reference_data':str(dense),'classes':{c:entry for c in x.CLASSES}}
            (folder/f'extended_{beam}.json').write_text(json.dumps(extra))
        profile={'model':model,'precision':'FP32','batch_size':1,'seed':42,'checkpoint':str(ck),'data_path':str(dense),'tf32_matmul':False,'tf32_cudnn':True,'torch_version':'fixture','cuda_version':'fixture','fps_from_mean_end_to_end':10,'end_to_end':{'mean_ms':100.,'p50_ms':95.,'p95_ms':110.},'preprocess_including_npy_io':{'mean_ms':20.},'host_to_gpu':{'mean_ms':5.},'model_forward_including_postprocess':{'mean_ms':75.},'parameters':123456,'peak_cuda_allocated_mib':300.,'peak_cuda_reserved_mib':500.,'active_voxels_median':1000,'frames':200,'warmup_frames':20,'gpu':'fixture GPU'}
        (folder/'profile.json').write_text(json.dumps(profile))
        return folder

    def test_upsert_preserves_other_runs_and_blank_vs_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'models';folder=self.fixture(root)
            book=Path(tmp)/'common.xlsx'
            key,rows=x.collect_run(root,'pointpillar','seed666_beam128',80)
            self.assertEqual(rows['Runs'][0]['TrainSeed'],666)
            self.assertEqual(rows['Runs'][0]['EvalSeed'],42)
            self.assertEqual(rows['Runs'][0]['TrainSeedEvidence'],'inferred_legacy')
            x.upsert_workbook(book,key,rows)
            self.fixture(root,'pv_rcnn','seed42_beam128',42,False)
            key2,rows2=x.collect_run(root,'pv_rcnn','seed42_beam128',80)
            x.upsert_workbook(book,key2,rows2)
            rows['Runs'][0]['FPS_serial']=12.
            x.upsert_workbook(book,key,rows)
            wb=x.load_excel_library().load_workbook(book)
            self.assertEqual(wb['Runs'].max_row,5)
            ids=[wb['Runs'].cell(i,1).value for i in (4,5)]
            self.assertEqual(set(ids),{key,key2})
            c=x.HEADERS['Runs'].index('FPS_serial')+1
            self.assertEqual(wb['Runs'].cell(5,c).value,12.)
            self.assertEqual(wb['Accuracy'].max_row,3+len(rows['Accuracy'])+len(rows2['Accuracy']))
            ec=x.HEADERS['Calibration'].index('ECE')+1
            self.assertEqual(wb['Calibration'].cell(4,ec).value,0)
            mc=x.HEADERS['Reliability'].index('MeanConfidence')+1
            self.assertIsNone(wb['Reliability'].cell(4,mc).value)
            self.assertEqual(wb['Runs'].freeze_panes,'F4')
            self.assertIsNotNone(wb['Runs'].auto_filter.ref)
            wb.close()

    def test_missing_profile_and_wrong_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root)
            p=folder/'extended_16.json';d=x.read_json(p);d['inference']['checkpoint_sha256']='wrong';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'weights'):x.collect_run(root,'pointpillar','seed666_beam128',80)
            d['inference']['checkpoint_sha256']=x.file_hash(folder/'ckpt/checkpoint_epoch_80.pth');p.write_text(json.dumps(d))
            (folder/'profile.json').unlink()
            with self.assertRaises(FileNotFoundError):x.collect_run(root,'pointpillar','seed666_beam128',80)

    def test_no_seed_inference_from_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root)
            (folder/'legacy_run_rename.json').unlink()
            with self.assertRaisesRegex(ValueError,'metadata'):x.training_details(folder)

    def test_unexpected_existing_schema_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            book=Path(tmp)/'common.xlsx';x.upsert_workbook(book)
            wb=x.load_excel_library().load_workbook(book);wb['Runs']['A3']='Custom header';wb.save(book);wb.close()
            original=book.read_bytes()
            with self.assertRaisesRegex(ValueError,'schema'):x.upsert_workbook(book)
            self.assertEqual(book.read_bytes(),original)


if __name__=='__main__':unittest.main()

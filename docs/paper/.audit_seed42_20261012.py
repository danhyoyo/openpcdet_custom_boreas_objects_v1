from pathlib import Path
import json,pickle,hashlib,sys
import numpy as np
sys.path.insert(0,"tools/boreas_benchmark")
from extended import sha256_file,truth_fingerprint
MODELS=["pointpillar","second","centerpoint","voxelnext","pv_rcnn","pointpillar_lite"]
CONDITIONS=["beam128","beam64","beam32","beam16","elevation_vlp16","elevation_vlp16_dense_fov","elevation_vlp32c","elevation_vlp32c_dense_fov"]
out=Path("docs/paper/results_seed42_20261012"); out.mkdir(parents=True,exist_ok=True)
records={}; truths=set(); common=set(); sources=[]
for model in MODELS:
    base=Path("output/boreas_models")/model/"seed42_beam128"
    trainpath=sorted(base.glob("run_config_*.json"))[-1]
    train=json.loads(trainpath.read_text())
    assert train["seed"]==42 and train["arguments"]["fix_random_seed"]
    assert train["epochs"]==80 and train["batch_size_per_gpu"]*train["total_gpus"]==4
    assert train["training_precision"]=="AMP"
    checkpoint=base/"ckpt/checkpoint_epoch_80.pth"
    checkpoint_hash=sha256_file(checkpoint)
    profile=json.loads((base/"profile.json").read_text())
    assert Path(profile["checkpoint"]).resolve()==checkpoint.resolve()
    assert (profile["seed"],profile["batch_size"],profile["precision"],profile["frames"],profile["warmup_frames"])==(42,1,"FP32",200,20)
    record={"train":train,"training_manifest":str(trainpath),"profile":profile,"checkpoint_sha256":checkpoint_hash,"conditions":{}}
    for condition in CONDITIONS:
        folder=base/"eval/epoch_80/val"/condition
        report=json.loads((folder/"range_ap.json").read_text())
        infer=json.loads((folder/"inference_manifest.json").read_text())
        assert infer["evaluation_seed"]==42 and infer["precision"]=="FP32" and infer["batch_size"]==1
        assert infer["checkpoint_sha256"]==checkpoint_hash
        pred_hash=sha256_file(folder/"result.pkl")
        assert pred_hash==infer["prediction_sha256"]==report["prediction_sha256"]
        assert report["protocol_version"]=="local_ap40_v1"
        data=Path(infer["data_path"])
        infos=pickle.loads((data/"custom_infos_val.pkl").read_bytes())
        assert len(infos)==1326
        truth=truth_fingerprint(infos); assert truth==report["truth_sha256"]; truths.add(truth)
        preds=pickle.loads((folder/"result.pkl").read_bytes())
        assert len(preds)==1326 and {str(p["frame_id"]) for p in preds}=={str(i["point_cloud"]["lidar_idx"]) for i in infos}
        common.add(tuple(infer[k] if k!="point_cloud_range" else tuple(infer[k]) for k in
            ["score_cutoff","nms_pre_maxsize","nms_post_maxsize","precision","batch_size","tf32_matmul","tf32_cudnn","torch_version","cuda_version","evaluation_seed","point_cloud_range"]))
        extpath=base/("extended_"+condition.removeprefix("beam")+".json") if condition.startswith("beam") else base/"elevation/epoch_80"/condition.removeprefix("elevation_")/"extended.json"
        ext=json.loads(extpath.read_text()); assert ext["verified_inference"]
        assert ext["truth_sha256"]==truth and ext["inference"]["prediction_sha256"]==pred_hash
        assert ext["inference"]["checkpoint_sha256"]==checkpoint_hash
        entry={"ap":report,"inference":infer,"extended":ext}
        if condition.startswith("elevation"):
            sample=json.loads((folder/"sampling_manifest.json").read_text())
            assert sample["truth_sha256"]==truth
            assert sha256_file(folder/"sampling_manifest.json")==sha256_file(data/"manifest.json")
            entry["sampling"]=sample
        record["conditions"][condition]=entry
        sources.append({"model":model,"condition":condition,"report":str(folder/"range_ap.json"),
                        "prediction_sha256":pred_hash,"checkpoint_sha256":checkpoint_hash,"truth_sha256":truth})
    records[model]=record
assert len(truths)==1 and len(common)==1
audit={"snapshot_date":"2026-10-12","training_seed":42,"models":records,"verified_conditions":len(sources),
       "truth_sha256":next(iter(truths)),"common_inference":list(next(iter(common))),"sources":sources}
(out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
def map40(m,c="beam128",r="all"):
    return records[m]["conditions"][c]["ap"]["ranges"][r]["mAP40"]
def frontier(cost):
    return [m for m in MODELS if not any(map40(n)>=map40(m) and cost(n)<=cost(m)
              and (map40(n)>map40(m) or cost(n)<cost(m)) for n in MODELS if n!=m)]
print("Verified",len(sources),"conditions, common truth/inference, six true seed42 checkpoints")
for c in CONDITIONS:
    print("ORDER",c,sorted(MODELS,key=lambda m:-map40(m,c)))
print("SPEED_FRONTIER",frontier(lambda m:records[m]["profile"]["end_to_end"]["mean_ms"]))
print("MEMORY_FRONTIER",frontier(lambda m:records[m]["profile"]["peak_cuda_allocated_mib"]))
for m in MODELS:
    b=map40(m); k=map40(m,"beam16"); e=map40(m,"elevation_vlp16"); t=records[m]["profile"]["end_to_end"]["mean_ms"]
    print(m,"dense",b,"mod16",k,"loss",b-k,"retention",100*k/b,"vlp16",e,"vlp16-minus-mod16",e-k,"latency",t)
print("VLP16_CROP_LOSS",[round(map40(m)-map40(m,"elevation_vlp16_dense_fov"),4) for m in MODELS])
print("VLP32C_MINUS_MOD32",[round(map40(m,"elevation_vlp32c")-map40(m,"beam32"),4) for m in MODELS])
print("DENSE_CONTROL_DIFF",[round(map40(m,"elevation_vlp32c_dense_fov")-map40(m),4) for m in MODELS])

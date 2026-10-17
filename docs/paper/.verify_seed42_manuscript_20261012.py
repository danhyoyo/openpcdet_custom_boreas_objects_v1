from pathlib import Path
import re,json
p=Path("/mnt/c/Users/ADMIN/.codex/visualizations/2026/10/10/01a125a0-3129-7080-9bab-5e683b39b2fb/boreas_benchmark_paper.tex")
s=p.read_text(encoding="utf-8")
s=s.replace("This removes the previous need to infer the training seed from historical logs.","Training seed 42 is verified directly from the archived manifests.")
s=s.replace("At modulo-32, SECOND (30.10) overtakes Lite (29.08), so an unchanged ordering across \\emph{all} severities would be incorrect. PV-RCNN remains first and PointPillars second in every modulo condition. We report the ordering directly without a rank-correlation table.","At modulo-32, SECOND (30.10) overtakes Lite (29.08), making the aggregate ordering severity-dependent. PV-RCNN remains first and PointPillars second in every modulo condition.")
s=s.replace("This appendix reports native-score diagnostics; confidence reliability is not a principal contribution or a claim in the title. Results are retained to document the saved prediction population and its limitations.","Native-score diagnostics complement AP and recall by documenting confidence on the saved prediction population. Their interpretation is conditional on the export and matching protocol below.")
errors=[];checked=0
for spec,body in re.findall(r"\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}",s,re.S):
 if not re.fullmatch("[lcr]+",spec):continue
 checked+=1
 for row in body.splitlines():
  if " & " in row and row.count(" & ")+1!=len(spec):errors.append((spec,row))
assert not errors,errors
d=json.loads(Path("docs/paper/results_seed42_20261012/audit.json").read_text())
names={"pointpillar":"PointPillars","second":"SECOND","centerpoint":"CenterPoint","voxelnext":"VoxelNeXt","pv_rcnn":"PV-RCNN","pointpillar_lite":"PointPillars Lite"}
def tablebody(label):
 j=s.index(r"\label{"+label+"}");k=s.index(r"\begin{tabular}",j);z=s.index(r"\end{tabular}",k)
 return s[k:z]
matched=0
for m,r in d["models"].items():
 def score(c="beam128"):return r["conditions"][c]["ap"]["ranges"]["all"]["mAP40"]
 prof=r["profile"];t=prof["end_to_end"]
 vals=[names[m],f'{prof["parameters"]/1e6:.3f}',f'{t["mean_ms"]:.3f}',f'{t["p50_ms"]:.3f}',f'{t["p95_ms"]:.3f}',f'{prof["fps_from_mean_end_to_end"]:.3f}',f'{prof["peak_cuda_allocated_mib"]:.2f}']
 vals += [f'{r["conditions"]["beam128"]["ap"]["ranges"]["all"]["classes"][n]["AP40"]:.2f}' for n in ["Car","Pedestrian","Cyclist"]]+[f'{score():.2f}']
 assert " & ".join(vals)+r"\\" in tablebody("tab:main"),m
 vals=[names[m]]+[f'{score(c):.2f}' for c in ["beam128","beam64","beam32","beam16"]]+[f'{score()-score("beam16"):.2f}',f'{100*score("beam16")/score():.2f}']
 assert " & ".join(vals)+r"\\" in tablebody("tab:robustness"),m
 vals=[names[m]]
 for n in ["Car","Pedestrian","Cyclist"]:
  vals += [f'{100*r["conditions"][c]["ap"]["ranges"]["all"]["classes"][n]["operating_point"]["recall"]:.2f}' for c in ["beam128","beam16"]]
 assert " & ".join(vals)+r"\\" in tablebody("tab:recall-results"),m
 vals=[names[m],f'{score("beam16"):.2f}',f'{score("elevation_vlp16"):.2f}',f'{score("elevation_vlp16")-score("beam16"):+.2f}',f'{score("elevation_vlp16_dense_fov"):.2f}',f'{score("beam32"):.2f}',f'{score("elevation_vlp32c"):.2f}',f'{score("elevation_vlp32c")-score("beam32"):+.2f}',f'{score("elevation_vlp32c_dense_fov"):.2f}']
 assert " & ".join(vals)+r"\\" in tablebody("tab:elevation-results"),m
 matched+=4
 cfg=r["train"]["effective_config"]
 lr=cfg["OPTIMIZATION"]["LR"];clip=cfg["OPTIMIZATION"]["GRAD_NORM_CLIP"]
 voxel=next(v for v in cfg["DATA_CONFIG"]["DATA_PROCESSOR"] if v["NAME"]=="transform_points_to_voxels")
 caps=voxel["MAX_NUMBER_OF_VOXELS"];nms=cfg["MODEL"]["POST_PROCESSING"].get("NMS_CONFIG",cfg["MODEL"].get("DENSE_HEAD",{}).get("POST_PROCESSING",{}).get("NMS_CONFIG",{}))["NMS_THRESH"]
 vals=[names[m],f'{lr:.3f}',str(clip),str(voxel["MAX_POINTS_PER_VOXEL"]),f'{caps["train"]:,}',f'{nms:.2f}']
 assert caps["train"]==caps["test"],m
 assert " & ".join(vals)+r"\\" in tablebody("tab:config"),(m,vals)
 matched+=1
for m,r in d["models"].items():
 for n in ["Car","Pedestrian","Cyclist"]:
  coords=" ".join(f'({k},{r["conditions"]["beam"+str(k)]["ap"]["ranges"]["all"]["classes"][n]["AP40"]:.4f})' for k in [16,32,64,128])
  assert coords in s,(m,n)
labels=re.findall(r"\\label\{([^}]+)\}",s);refs=re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",s)
assert len(set(labels))==len(labels) and set(refs)<=set(labels)
main=s[:s.index(r"\appendix")]
assert "Kendall" not in s and r"\tau_b" not in s
assert r"\operatorname{ECE}" not in main and r"\operatorname{Brier}" not in main
assert not re.search("seed.?666|draft assumption",s,re.I)
p.write_text(s,encoding="utf-8")
checks_path=Path("docs/paper/results_seed42_20261012/manuscript_source_checks.json")
checks=json.loads(checks_path.read_text())
checks.update(table_column_checks=checked,table_numeric_rows_verified=matched,class_ap_plot_series_verified=18)
checks_path.write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps(checks,indent=2))
print("Dense table:")
print(tablebody("tab:main"))

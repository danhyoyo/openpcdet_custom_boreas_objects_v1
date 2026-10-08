# Tài liệu nền cho benchmark LiDAR detector trên Boreas-Objects-V1

Ngày kiểm tra nguồn: 07/10/2026. Đã đối chiếu file `tom_tat_du_an_benchmark_lidar_boreas.docx`.
Phạm vi: RQ1 accuracy, RQ2 efficiency, RQ3 radial-distance performance, RQ4 controlled beam sparsification.
Đây là curated literature search, không phải systematic review có PRISMA; chưa đủ để khẳng định không có nghiên cứu tương tự.
Các paper cũ được giữ như nguồn gốc kiến trúc/dataset, không dùng để khẳng định SOTA hiện tại.

## 1. Những nguồn cần đọc đầu tiên

Boreas [1]; năm detector [2–6]; Gupta [13]; Griesbacher [11]; Feng [12]; Robo3D [8]; Dong [9]; Eskandar [10].
Chúng cung cấp dataset provenance, architecture taxonomy, beam-domain-gap background và các prior benchmark gần nhất.
PillarNet [7] hỗ trợ phần accuracy–efficiency; Wei [15] giải thích beam-induced domain gap.

## 2. Danh mục paper và vị trí dùng

| ID | Paper / nguồn gốc | Venue và năm | Dùng cho | Giới hạn / lưu ý |
|---|---|---|---|---|
| 1 | [Boreas: A multi-season autonomous driving dataset](https://journals.sagepub.com/doi/10.1177/02783649231160195) | The International Journal of Robotics Research, 2023 | Dataset, introduction | LiDAR 128 beam; Objects-V1 sunny only; 37 official training scenes/16 hidden-label test scenes; no difficulty tiers. |
| 2 | [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019 | RQ1/RQ2, detector families | Pillar representation. Cite this paper for internal PointPillars Lite; Lite itself is a channel-width ablation. |
| 3 | [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337) | Sensors, 2018 | RQ1/RQ2, sparse voxel | Publisher metadata verified; direct full-text access rate-limited in this search. Measured runtime must come from our hardware. |
| 4 | [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021 | RQ1, center-based detection | Original paper includes refinement; describe the actual one-stage OpenPCDet variant used in our experiment. |
| 5 | [VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking](https://arxiv.org/abs/2303.11301) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023 | RQ1/RQ2, fully sparse | Distinguish original method from our NMS-enabled implementation configuration. |
| 6 | [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://arxiv.org/abs/1912.13192) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020 | RQ1/RQ2, point-voxel two-stage | Point-voxel feature abstraction and proposal refinement. |
| 7 | [PillarNet: Real-Time and High-Performance Pillar-based 3D Object Detection](https://arxiv.org/abs/2205.07403) | European Conference on Computer Vision (ECCV), 2022 | RQ2, efficiency background | Useful precedent for accuracy-speed trade-offs and pillar encoding; sparse 2D operators are still sparse operators. |
| 8 | [Robo3D: Towards Robust and Reliable 3D Perception against Corruptions](https://arxiv.org/abs/2303.17597) | IEEE/CVF International Conference on Computer Vision (ICCV), 2023 | RQ4, robustness benchmark | Eight corruptions; mCE/mRR must not be treated as interchangeable with our AP drops. |
| 9 | [Benchmarking Robustness of 3D Object Detection to Common Corruptions](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Benchmarking_Robustness_of_3D_Object_Detection_to_Common_Corruptions_CVPR_2023_paper.html) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023 | RQ4, benchmark design | KITTI-C/nuScenes-C/Waymo-C; 27 corruption types and 24 detectors. Title here follows CVF BibTeX; PDF adds in Autonomous Driving. |
| 10 | [An Empirical Study of the Generalization Ability of Lidar 3D Object Detectors to Unseen Domains](https://arxiv.org/abs/2402.17562) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024 | Fairness, RQ4 | Nine detectors/six benchmarks; architecture, voxel encoding, augmentation and anchors affect generalization. Full author list follows PDF/arXiv. |
| 11 | [An Investigation of Beam Density on LiDAR Object Detection Performance](https://arxiv.org/abs/2503.15087) | 28th Computer Vision Winter Workshop (CVWW), 2025 | RQ4, closest beam-density study | Dense 128-beam setting; architecture comparison and beam-induced domain gaps. Do not claim beam-density evaluation is new. |
| 12 | [Impact of LiDAR beam loss on 3D object detection: A systematic analysis of vulnerable road user safety](https://www.sciencedirect.com/science/article/abs/pii/S0167865526001534) | Pattern Recognition Letters, 2026 | RQ4, newest close prior work | Publisher public metadata/abstract only; full text unavailable. Beam loss depends on beam location; avoid inferring unpublished protocol details. |
| 13 | [Evaluating LiDAR Perception Algorithms for All-Weather Autonomy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12736427/) | Sensors, 2025 | Boreas prior work, RQ3/RQ4 context | Full text read. Boreas-Objects synthetic fog, CADC snow; pretrained PointPillars, downsample 128 to 32; filtering emphasis. Different class/range/evaluation protocol. |
| 14 | [Augmentation of LiDAR Scenes with Adverse Weather Conditions Using Latent Diffusion Models](https://link.springer.com/chapter/10.1007/978-981-95-4097-6_14) | Neural Information Processing (ICONIP 2025), 2026 | Boreas prior work, optional weather expansion | Publisher citation year 2026, conference ICONIP2025, first online 2025-11-14. Public abstract/metadata only; linked B-3DOD-B-AWC supplies a separate sunny/snow benchmark. |
| 15 | [LiDAR Distillation: Bridging the Beam-Induced Domain Gap for 3D Object Detection](https://arxiv.org/abs/2203.14956) | European Conference on Computer Vision (ECCV), 2022 | RQ4, beam domain gap | Adaptation/distillation method, not an equivalent benchmark; test-only beam reduction and retraining must be distinguished. |
| 16 | [Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) | IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012 | Dataset and evaluation history | Cite official KITTI web protocol separately for AP40: the 2012 paper predates the 2019 AP40 update. |
| 17 | [nuScenes: A Multimodal Dataset for Autonomous Driving](https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020 | Dataset context | Use for dataset comparison, not as data used in our current experiment. nuScenes metrics differ from IoU AP_R40. |
| 18 | [Scalability in Perception for Autonomous Driving: Waymo Open Dataset](https://arxiv.org/abs/1912.04838) | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020 | Dataset context | Large-scale benchmark precedent; complete authors can be imported from publisher if desired. |
| 19 | [Canadian adverse driving conditions dataset](https://arxiv.org/abs/2001.10117) | The International Journal of Robotics Research, 2021 | Future real-snow data | Official devkit supplies article citation. Different sensor/classes/domain from Boreas; no claim currently trained/evaluated CADC. |

## 3. Nguồn dữ liệu, phần mềm và protocol

- **[The Boreas Dataset](https://www.boreas.utias.utoronto.ca/)**: Official dataset entry; JavaScript-only page, scientific details verified in paper.
- **[Boreas Autonomous Driving Dataset — AWS registry](https://registry.opendata.aws/boreas/)**: Official access and CC BY4.0 license record; not downloaded in this task.
- **[pyboreas devkit](https://github.com/utiasASRL/pyboreas)**: Raw format, official scene splits and loader; record exact used commit/version.
- **[OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds](https://github.com/open-mmlab/OpenPCDet)**: Cite development team2020 as software; also record actual checkout commit and our changes.
- **[KITTI Object Detection Evaluation](https://www.cvlibs.net/datasets/kitti/eval_object.php)**: AP40 updated2019-10-08; class IoU thresholds. Our custom Boreas labels lack KITTI difficulty tiers.
- **[PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark)**: Warmup, CUDA synchronization and timing overhead; cite documentation when describing runtime protocol.
- **[Boreas 3D Object Detection Benchmark for Adverse Weather Conditions (B-3DOD-B-AWC)](https://github.com/matteandre/B-3DOD-B-AWC)**: Separate annotation resource: sunny train labels from model inference, snowy test manual labels; classes car/pedestrian/truck; differs from official Objects-V1.
- **[CADC devkit](https://github.com/mpitropov/cadc_devkit)**: Official dataset access, labels and citation; future expansion only.
- **[Robo3D devkit](https://github.com/worldbench/Robo3D)**: Corruption protocols and reproducibility reference; not currently implemented in our experiment.

Các nguồn nuScenes, Waymo và CADC là background hoặc hướng mở rộng. Chúng không phải dữ liệu đã dùng trong benchmark hiện tại.
Không cần tải thêm dataset để bắt đầu viết Introduction, Related Work và Experimental Protocol.

## 4. Bản đồ trích dẫn cho manuscript

- **Introduction**: Boreas [1], KITTI [16], nuScenes [17], Waymo [18]; robustness benchmarks [8,9].
- **Related Work — architectural families**: pillar [2,7]; sparse voxel [3]; center based [4]; fully sparse [5]; point-voxel/two-stage [6]. Các trục phân loại này giao nhau, không phải sáu họ loại trừ nhau.
- **Related Work — sensor sparsity/generalization**: beam-density [11], beam loss [12], distillation [15], architecture/augmentation sensitivity [10].
- **Related Work — Boreas prior studies**: filtering/weather study [13], latent weather augmentation and supplementary annotations [14].
- **Dataset & Split**: paper Boreas [1] và pyboreas; ghi rõ 29 local train windows,8 local val windows từ37 official training windows, temporal purge2s. Đây là protocol riêng của dự án, không phải official split29/8 do tác giả Boreas đề xuất.
- **Evaluation**: KITTI paper [16] + official AP40 protocol; mô tả local AP40/range rules riêng. Không nói local AP40 đồng nhất với official Boreas hidden-test score.
- **Runtime**: OpenPCDet commit, PyTorch benchmark documentation, precision/batch size/hardware, warmup và ranh giới preprocessing/network/postprocessing.
- **Discussion**: đối chiếu relative AP retention và ranking stability với [8,11,12]; không so số AP/FPS trực tiếp nếu dataset, class, GPU hoặc protocol khác nhau.

## 5. Novelty và đối chiếu phản biện

### Vấn đề1: Boreas không đồng nghĩa bộ nhãn detection đa mùa

Boreas tổng thể đa mùa nhưng Objects-V1 sunny only [1]. Nguồn bổ sung B-3DOD-B-AWC có snow annotations, là một resource khác [14 + repository]. Không được viết rằng toàn bộ Boreas không có bất kỳ snow detection labels nào. Benchmark hiện tại chưa dùng resource bổ sung đó.

### Vấn đề2: Beam-density evaluation đã có tiền lệ

Griesbacher2025 [11], Feng2026 [12], Robo3D [8] và LiDAR Distillation [15] đã nghiên cứu beam/sensor domain gaps. Không claim first beam robustness study hoặc first systematic LiDAR benchmark. Kiểm tra cả vị trí và quy luật chọn ring: periodic ring IDs chưa chứng minh tương đương sensor64/32/16beam vật lý hay đều theo góc đứng.

### Vấn đề3: Prior Boreas detection/weather evaluation đã tồn tại

Gupta2025 [13] dùng Boreas-Objects với fog tổng hợp; Matteazzi2026/ICONIP2025 [14] cung cấp benchmark adverse weather riêng. Điểm khác biệt dự kiến của dự án là tổ hợp: các detector được train trên cùng native128beam Boreas split; protocol runtime chung; per-class/range analysis; controlled beam subsampling; ranking stability và Pareto analysis. Đây là hướng định vị, chưa phải bằng chứng độc quyền/new-first claim.

### Vấn đề4: Cấu hình implementation không đồng nhất với paper gốc

CenterPoint config one-stage và VoxelNeXt config có NMS phải được ghi đúng. PointPillars Lite là giảm channels nội bộ. Không gọi các run là official recommended results nếu đã đổi range, voxel size, heads, classes hoặc tuning budget.

### Vấn đề5: Citation metadata và metric phải tách rõ

AP40 không có trong KITTI2012paper: cite webupdate2019. Matteazzi: first online2025 nhưng Springer citationyear2026. Full text Feng và Matteazzi chưa đọc được; chỉ dùng public abstract/metadata và repo cho các claims đã kiểm chứng. Không copy mCE/mRR definitions sang AP-drop nếu chưa thực hiện đúng protocol.

## 6. Ghi chú nguồn và mức chắc chắn

- 19 paper records +9 official resource records; registry lưu source type,accessibility,URL,date và limitations. CADC article và devkit được tách thành nguồn học thuật và nguồn phần mềm.
- Độ chắc chắn cao cho title,venue và dataset provenance có nguồn gốc; mức vừa cho novelty gap vì search không exhaustive.
- Paper behind paywall: metadata/abstractpublic vẫn dùng được; không tuyên bố đã đọc toàn văn. SECOND bị rate-limit ở một số endpoint.
- Không dùng blog,tóm tắt AI,ResearchGate hoặc aggregator làm chứng cứ khoa học; nguồn thứ cấp chỉ giúp tìm primary record rồi bị loại khỏi bộ chứng cứ.
- Không lấy assertion về venue quartile/publication feasibility trong DOCX làm bằng chứng external. Phần này không thuộc lượt literature search hiện tại.
- BibTeX URL có thể dẫn tới author preprint nhưng venue/year dùng bản conference đã xác minh; không gán DOI arXiv như DOI proceedings.

## 7. Hướng viết tiếp

Có thể bắt đầu3 mục trước khi train hoàn tất: Introduction,Related Work,Dataset & Experimental Protocol. Results/Discussion cần chờ các detector và phân tích thống kê. Main contribution nên gắn với observed ranking changes và deployment trade-offs, không dự đoán trước detector nào tốt nhất.

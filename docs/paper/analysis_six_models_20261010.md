# Phân tích sáu mô hình hoàn tất — 10/10/2026

Kết quả chính là local 3D AP40 trên 1.326 frame validation, không phải điểm official test của Boreas. Mỗi mô hình có một checkpoint epoch 80; cả bốn mức beam dùng nguyên checkpoint đó.

| Mô hình | mAP 128 (%) | FPS | Mean / p95 (ms) | Memory allocated (MiB) | mAP 16 (%) | Giữ lại (%) |
|---|---:|---:|---:|---:|---:|---:|
| PointPillars | 53.83 | 15.84 | 63.15 / 74.07 | 1543.18 | 14.66 | 27.24 |
| SECOND | 50.61 | 24.10 | 41.49 / 48.35 | 183.25 | 14.42 | 28.50 |
| CenterPoint | 46.18 | 20.32 | 49.21 / 59.35 | 200.96 | 4.84 | 10.47 |
| VoxelNeXt | 47.10 | 20.34 | 49.18 / 62.56 | 174.79 | 7.12 | 15.12 |
| PV-RCNN | 59.11 | 3.22 | 310.81 / 340.85 | 1524.54 | 18.20 | 30.79 |
| PointPillars Lite | 49.66 | 21.59 | 46.32 / 55.94 | 876.43 | 13.40 | 26.99 |

- PV-RCNN dẫn đầu mAP chung và cả ba lớp ở 128 beam; hơn PointPillars 5,28 điểm phần trăm nhưng chậm hơn SECOND 7,49 lần theo mean latency.
- SECOND nhanh nhất, nằm trên cả Pareto accuracy–speed và accuracy–memory. Pareto accuracy–speed: SECOND, PointPillars, PV-RCNN; accuracy–memory: VoxelNeXt, SECOND, PV-RCNN.
- Lite giảm 74,82% tham số, 26,65% mean latency và 43,21% allocated memory so với PointPillars, đổi lại mất 4,17 điểm mAP. Đây không phải ablation cùng seed.
- PV-RCNN dẫn mAP ở cả bốn dải khoảng cách. Từ 60 m, PointPillars có Cyclist AP cao nhất (47,02%), nên vị trí dẫn đầu phụ thuộc lớp/khoảng cách.
- Thứ hạng mAP chung không đổi qua 128/64/32/16 beam (Kendall tau-b = 1), nhưng ở 16 beam chỉ còn 10,47–30,79% AP ban đầu. PV-RCNN dẫn Car/Cyclist; PointPillars dẫn Pedestrian.
- Cohort Pedestrian có 21–50 điểm ở dữ liệu 128 beam gồm 9.132 đối tượng; recall ở 16 beam chỉ tối đa 0,018. Cohort được cố định theo dữ liệu gốc, không chuyển nhóm khi subsampling.
- VoxelNeXt giảm Pedestrian ECE từ 0,0488 xuống 0,0078, trong khi recall giảm từ 0,6632 xuống 0,1457. ECE thấp hơn không đủ chứng minh perception tốt hơn; cần đọc cùng AP/recall và quần thể prediction.

Seed huấn luyện thực tế: PointPillars/SECOND/CenterPoint = 666 (lịch sử được tác giả xác nhận; log cũ không có trường seed số), VoxelNeXt/PV-RCNN/Lite = 42 (manifest). Seed evaluation/profiling = 42 cho tất cả. Không đổi nhãn seed huấn luyện và không tính mean/std giữa sáu kiến trúc. Chưa có multi-seed uncertainty hoặc bootstrap.

Đã kiểm tra 24 bộ report AP/extended/inference: prediction hash, cùng ground-truth hash, checkpoint/config hash bất biến trong mỗi mô hình, FP32, TF32 matmul False/cuDNN True, cutoff 0,001, caps 4096/500. Profiling cùng RTX 5060 Ti, PyTorch 2.11.0+cu128/CUDA 12.8, 200 frame/20 warmup. Chưa đọc lại result.pkl để tái tính IoU; đối chiếu trên các report gốc và metadata. Provenance training cũ và môi trường CPU/driver/package chưa đầy đủ.

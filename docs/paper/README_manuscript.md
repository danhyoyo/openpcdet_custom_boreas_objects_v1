# Bản thảo bài báo Boreas

File chính: `boreas_benchmark_paper.tex`. Viết bằng tiếng Anh, dạng article hai cột; có thể chuyển sang template của hội nghị/tạp chí sau khi chọn nơi gửi.

## Biên dịch

- Mở `.tex` trong Codex để sửa source và xem PDF preview.
- File tự chứa bibliography và đồ thị vector PGFPlots từ số liệu thực; không cần ảnh ngoài hay chạy BibTeX.
- Nếu dùng Overleaf: tải file `.tex`, chọn pdfLaTeX. Nếu dùng TeX cục bộ: chạy `pdflatex boreas_benchmark_paper.tex` hai lần.

## Phạm vi

- RQ1: local 3D AP40 và precision/recall/F1 riêng Car, Pedestrian, Cyclist.
- RQ2: latency batch 1, p50/p95, serial FPS, CUDA memory, params, input occupancy, Pareto.
- RQ3: 0–20, 20–40, 40–60, ≥60 m.
- RQ4: train 128, giữ nguyên checkpoint, eval 128/64/32/16 ring subsets.
- Split hiện tại: 29/8 temporal intervals; không random frame; purge 2 giây; không dùng official test.
- Không tuyên bố detection đa mùa hoặc weather robustness: Boreas-Objects-V1 dùng nhãn detection trong điều kiện nắng.

## Chỗ cần bổ sung

Kết quả sáu detector đã được điền ngày 10/10/2026: accuracy, latency/Pareto, range, sparsity, density, ranking và calibration. Tìm `TBD` và `\pending{...}` cho thông tin tác giả, release và provenance còn thiếu. Số frame/box trong bảng dataset là thống kê dữ liệu.

Seed, batch train, precision train, TF32, phần cứng và các version môi trường phải xác minh từ metadata/log, không suy ra từ tên folder. Score export hiện tại 0.001, khác threshold 0.1 cho precision/recall/F1; output cũ phải eval lại nếu protocol khác.

Repeated seeds, cluster bootstrap và sensitivity analyses là kế hoạch chưa được coi là đã thực hiện; bản thảo ghi rõ. Nếu không chạy các phần này, sửa wording tương ứng trước khi gửi.

Bibliography nhúng trong `.tex` được tạo từ các record đã xác minh ở `references.bib`; `citation_registry.json` và `literature_review.md` giữ nguồn đối chiếu. Nếu sửa citation, giữ các file này nhất quán.

Appendix Artifact and Completion Checklist là phần dành cho tác giả, cần bỏ trước khi nộp.

## Cập nhật 2026-10-08

Đã bổ sung density recall, native-score calibration ECE/Brier, ranking Kendall tau-b và tooling ba seed. Ghi chú lịch sử ngày 08/10: lúc đó các số liệu còn là placeholder; trạng thái hiện tại đã được cập nhật bên dưới. Ba seed độc lập vẫn cần hoàn tất. Interval bootstrap chưa được triển khai. Xem `tools/boreas_benchmark/README.md` để chạy các module mới.

## Kết quả hoàn tất 2026-10-10

- Đã cập nhật abstract, phương pháp thực tế, bảng kết quả, ba hình vector (Pareto, sparsity theo lớp, reliability), discussion, conclusion và bảng operating-point PV-RCNN.
- Mỗi mô hình một checkpoint; giữ seed train 666 cho PointPillars/SECOND/CenterPoint và 42 cho VoxelNeXt/PV-RCNN/Lite. Evaluation/profiling seed 42 cho cả sáu. Không phải một ma trận multi-seed.
- Bảng và số liệu bổ sung nằm trong `results_20261010/`; `audit.json` ghi kiểm tra tương thích và hash nguồn. Phân tích tiếng Việt: `analysis_six_models_20261010.md`.
- Backup bản thảo trước khi điền kết quả: `boreas_benchmark_paper_before_results_20261010.tex`.
- Chưa thực hiện matched-seed repeats, bootstrap, sensitivity, phân tích learning curve hội tụ hay qualitative frame selection.
- ZIP chính đã cập nhật kèm source và supplementary records; biên dịch yêu cầu PGFPlots (TeX Live/Overleaf).

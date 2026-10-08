# Bản thảo bài báo Boreas

File chính: `boreas_benchmark_paper.tex`. Viết bằng tiếng Anh, dạng article hai cột; có thể chuyển sang template của hội nghị/tạp chí sau khi chọn nơi gửi.

## Biên dịch

- Mở `.tex` trong Codex để sửa source và xem PDF preview.
- File tự chứa bibliography và các khung hình placeholder; không cần ảnh ngoài hay chạy BibTeX.
- Nếu dùng Overleaf: tải file `.tex`, chọn pdfLaTeX. Nếu dùng TeX cục bộ: chạy `pdflatex boreas_benchmark_paper.tex` hai lần.

## Phạm vi

- RQ1: local 3D AP40 và precision/recall/F1 riêng Car, Pedestrian, Cyclist.
- RQ2: latency batch 1, p50/p95, serial FPS, CUDA memory, params, input occupancy, Pareto.
- RQ3: 0–20, 20–40, 40–60, ≥60 m.
- RQ4: train 128, giữ nguyên checkpoint, eval 128/64/32/16 ring subsets.
- Split hiện tại: 29/8 temporal intervals; không random frame; purge 2 giây; không dùng official test.
- Không tuyên bố detection đa mùa hoặc weather robustness: Boreas-Objects-V1 dùng nhãn detection trong điều kiện nắng.

## Chỗ cần bổ sung

Tìm `TBD` và `\pending{...}` trong source. Tất cả kết quả detector được để trống theo yêu cầu. Số frame/box trong các bảng dataset lấy từ manifest chuẩn bị dữ liệu, không phải kết quả train.

Seed, batch train, precision train, TF32, phần cứng và các version môi trường phải xác minh từ metadata/log, không suy ra từ tên folder. Score export hiện tại 0.001, khác threshold 0.1 cho precision/recall/F1; output cũ phải eval lại nếu protocol khác.

Repeated seeds, cluster bootstrap và sensitivity analyses là kế hoạch chưa được coi là đã thực hiện; bản thảo ghi rõ. Nếu không chạy các phần này, sửa wording tương ứng trước khi gửi.

Bibliography nhúng trong `.tex` được tạo từ các record đã xác minh ở `references.bib`; `citation_registry.json` và `literature_review.md` giữ nguồn đối chiếu. Nếu sửa citation, giữ các file này nhất quán.

Appendix Artifact and Completion Checklist là phần dành cho tác giả, cần bỏ trước khi nộp.

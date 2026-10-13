# Chạy Lab Day 10 trên Kaggle

1. Tạo Kaggle Notebook, bật **GPU T4** và **Internet**. Trong **Add Input**, gắn dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`. Notebook sẽ tìm `images/` và `labels-YOLO/` dưới `/kaggle/input`.
2. Mở `lab_day10.ipynb` của repo này trên Kaggle và chạy các cell theo thứ tự. Cell đầu clone phiên bản mới nhất của repo vào `/kaggle/working/day10`. Các cell E0, E1, E2, E3, E4 và E5 chạy từng tiến trình Python riêng để giải phóng VRAM giữa các thí nghiệm.
3. Nếu SFT hết VRAM, đổi lệnh `train` của cell tương ứng thành `--batch-size 1 --grad-accum 8` rồi chạy lại cell đó. Effective batch vẫn bằng 8. E5 GRPO sinh 4 đáp án/ảnh và có thể cần GPU nhiều bộ nhớ hơn; giữ lại lỗi và log thực tế nếu Kaggle T4 không đủ.
4. Sau cell cuối, tải `/kaggle/working/day10_results.zip` và gửi lại file đó. ZIP chỉ chứa JSON bằng chứng, dự đoán thô, số liệu tài nguyên, metadata chia dữ liệu và ảnh minh hoạ. Nó không chứa dataset, token hay trọng số.
5. `REPORT.md` sẽ được hoàn thiện sau khi có kết quả thật E0–E5. Không điền điểm F1, CI, VRAM hoặc thời gian huấn luyện bằng số ước lượng. Sau khi đối chiếu báo cáo và bằng chứng, chạy `python scripts/check_submission.py` để đạt 0 FAIL rồi mới nộp.

`work/`, `data/`, `.env`, `*.pt` và `*.safetensors` đã bị `.gitignore` loại khỏi Git. `scripts/` và `tests/` của đề bài được giữ nguyên.

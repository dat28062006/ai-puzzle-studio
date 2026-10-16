# Kết quả xác minh

Ngày: 16/10/2026. Bản cập nhật Search + RL. Môi trường: Windows, Python 3.11, Streamlit 1.66.0, Pillow 12.3.0, Gymnasium 1.4.0.

- Bộ kiểm thử đầy đủ: **48 passed**. Gồm luật game, solver, generator, xử lý ảnh, xuất file, Q-learning và ba luồng AppTest cho giao diện.
- A* được đối chiếu với Dijkstra độc lập trên các bước di chuyển đơn lẻ. Màn tradeoff: ưu tiên bước đi cho kết quả **20 bước / 9 lượt đẩy**; ưu tiên lượt đẩy cho kết quả **22 bước / 7 lượt đẩy**.
- Môi trường RL vượt qua `gymnasium.utils.env_checker.check_env`.
- Kiểm thử chặn lời gọi solver khi train/suy luận RL; Q-learning vẫn học và giải được màn mẫu.
- Đã thử trên trình duyệt: nút train Q-learning chạy 50.000 episode trên seed 60/Lối vòng và lưu checkpoint; có biểu đồ reward, số transition và log thực tế.
- Đã kiểm tra giao diện trên cửa sổ hẹp: thanh bên tự thu gọn, các nút di chuyển và chỉ số vẫn nằm theo hàng.
- Đã tạo gói mẫu và đọc lại bản đồ JSON; GIF lời giải chứa các trạng thái hợp lệ.
- Ba màn mẫu có lời giải ngắn nhất lần lượt 1, 5 và 9 lượt di chuyển; các checkpoint Q-learning đi kèm đều giải được ba màn này.

## Đối chiếu trên seed 60/Lối vòng

| Agent | Bước đi | Lượt đẩy | Trạng thái/quyết định |
|---|---:|---:|---:|
| BFS | 30 | 8 | 2.788 trạng thái di chuyển |
| A* ưu tiên bước đi | 30 | 8 | 52 trạng thái sau lần đẩy |
| A* ưu tiên lượt đẩy | 30 | 8 | 39 trạng thái sau lần đẩy |
| Q-learning sau 50.000 episode | 30 | 8 | 8 lần chọn đẩy |

Các đơn vị đếm khác nhau. Không diễn giải trực tiếp chúng thành tỉ lệ tăng tốc. Thời gian đo thật được xuất trong `examples/agent-comparison-seed60.json`.

Policy ngẫu nhiên đối chứng giải được 0/24 lượt thử từ vị trí bắt đầu của seed 60. Policy Q-learning greedy sau học giải được vị trí này; kiểm tra 22 vị trí reset riêng chỉ giải được 2/22. Đây là tabular policy theo bản đồ, không phải bằng chứng tổng quát sang các trạng thái hoặc level mới. Không gắn nhãn bảo đảm tối ưu cho RL dù lời giải của seed 60 khớp A*.

Các giới hạn: ảnh GenAI được tạo trước; chưa có ComfyUI trực tiếp, chưa có DQN/PPO qua nhiều level, chưa hiệu chuẩn độ khó với người chơi thật. Lịch sử phiên nằm trong bộ nhớ; checkpoint RL được lưu vào file.

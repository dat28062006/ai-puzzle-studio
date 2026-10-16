# Kịch bản demo 60–90 giây

Chuẩn bị: mở Studio, seed 42, khung Kho mở, hai thùng, đồ họa GenAI. Bấm Tạo màn nếu cần đặt lại. Trên cửa sổ hẹp, mở thanh bên bằng ≫ để chuyển trang.

| Thời điểm | Thao tác | Nội dung trình bày |
|---|---|---|
| 0–10 giây | Giới thiệu bàn chơi | “Đây là Puzzle Lab, bản demo kết hợp tạo/kiểm thử màn puzzle với workflow đồ họa GenAI.” |
| 10–20 giây | Đi xuống một bước, Undo | “Engine xử lý luật đẩy thùng, giữ vị trí đích độc lập, hỗ trợ chơi thử và hoàn tác.” |
| 20–35 giây | Chọn A* hoặc RL, bấm Bot giải màn rồi Phát | “A* tìm lời giải tối ưu theo mục tiêu đã chọn. Agent Q-learning dùng policy học từ reward. Các lời giải được chạy lại qua engine để xác minh.” |
| 35–45 giây | Chỉ vào số liệu | “Đây là số bước, số lần đẩy, số trạng thái tìm kiếm và thời gian đo thật trên máy.” |
| 45–55 giây | Về đầu, đổi sang Tối giản rồi GenAI | “Cùng một màn và luật chơi có thể thay lớp hình ảnh mà không đổi logic.” |
| 55–70 giây | Mở Bộ asset | “Năm ảnh được tạo bằng GenAI. App nạp ảnh, chuẩn hóa RGBA 64×64 rồi tự ghép vào game; prompt và nguồn gốc đi kèm.” |
| 70–80 giây | Mở Huấn luyện RL | “Có nút train thật trên CPU, biểu đồ reward, checkpoint Q-table và log. Policy hiện tại học theo từng bản đồ.” |
| 80–90 giây | Mở Kiểm thử, So sánh 4 agent | “So sánh cùng trạng thái để thấy số bước, số lượt đẩy, thời gian và phạm vi bảo đảm tối ưu. Có thể tải toàn bộ kết quả.” |

Không cần tạo ảnh trong lúc trình bày: bộ ảnh mặc định đã có trong project. File `examples/solution.gif` là hoạt ảnh lời giải thực tế có sẵn để dùng khi giới thiệu.

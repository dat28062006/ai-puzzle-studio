# Puzzle Lab — AI Puzzle Studio

Bản demo kết hợp **thiết kế/kiểm thử level puzzle** và **workflow đồ họa GenAI**. Chạy cục bộ bằng Python + Streamlit; không cần API key hoặc ComfyUI.

## Chạy trên máy hiện tại

Mở `START_DEMO.cmd`, hoặc chạy trong thư mục project:

```powershell
python run_demo.py
```

Truy cập **http://127.0.0.1:8501**. Trong workspace Codex hiện tại, launcher tự nhận bộ thư viện riêng ở `../../work/dependencies`; không thay đổi thư viện Python toàn hệ thống.

## Cài trên máy khác

Cần Python 3.11 trở lên. Mở PowerShell trong thư mục project:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_demo.py
```

Nếu cổng 8501 đang được dùng, chạy với cổng khác:

```powershell
python run_demo.py --server.port=8502
```

## Thử demo

1. Mở **Studio**. Màn mặc định là màn được sinh với seed 42, hai thùng.
2. Dùng nút mũi tên để chơi; thử **Undo** và **Chơi lại**.
3. Chọn agent **A* · Ít bước đi**, **A* · Ít lượt đẩy**, **RL · Q-learning** hoặc **BFS · Đối chiếu**, rồi bấm **Bot giải màn**. Agent bắt đầu từ vị trí đang chơi.
4. Bấm **Phát** để xem tự động, **Bước →** để xem từng bước, hoặc **Về đầu** để về vị trí lúc bắt đầu tìm kiếm.
5. Đổi giữa **GenAI** và **Tối giản**. Luật game và trạng thái không thay đổi.
6. Đổi khung bản đồ, số thùng, seed rồi bấm **Tạo màn**. Màn được giữ lại sau khi solver xác minh có lời giải. Biểu tượng ↻ chỉ đổi seed; bấm Tạo màn để áp dụng.
7. Mở **Bộ asset** để xem năm ảnh, prompt nguồn, hoặc nạp bộ ảnh của bạn.
8. Mở **Huấn luyện RL** để huấn luyện Q-learning, xem reward, tải checkpoint/log. Bộ demo có sẵn policy cho ba màn mẫu, seed 42/Kho mở và seed 60/Lối vòng.
9. Mở **Kiểm thử**, chạy ba màn mẫu hoặc so sánh bốn agent trên cùng trạng thái.
10. Mở mục **Xuất bản đồ, báo cáo và ảnh màn chơi** để tải JSON, PNG hoặc gói ZIP có GIF lời giải.

Trên cửa sổ hẹp, bấm biểu tượng **≫** góc trên trái để mở thanh điều hướng.

## Bộ đồ họa GenAI

Chủ đề **Orbital Warehouse**, gồm `floor`, `wall`, `goal`, `box`, `player`. Năm ảnh được tạo riêng bằng công cụ **image_gen** tích hợp trong Codex. Prompt đầy đủ và nguồn gốc được lưu trong `assets/prompts.json`; ảnh gốc ở `assets/source/`, tile đã chuẩn hóa ở `assets/tiles/`.

Pipeline nạp ảnh trong app:

```text
Ảnh GenAI hoặc ảnh tải lên
→ kiểm tra loại ảnh/kích thước
→ chuẩn hóa PNG RGBA 64×64
→ giữ tỷ lệ và alpha cho các vật thể
→ ánh xạ vào renderer
→ xem trước và xuất file
```

Tên file cần là `floor.png`, `wall.png`, `goal.png`, `box.png`, `player.png` (cũng nhận JPEG/WebP với cùng tên gốc). Có thể thay một ảnh; các ảnh còn lại dùng bộ mặc định. Mỗi ảnh tối đa 10 MB và 4096×4096. Ảnh tải lên thay đổi trong phiên hiện tại; app không ghi đè bộ mặc định. Có nút khôi phục bộ ảnh gốc.

**Tạo ảnh ngay trong app bằng ComfyUI chưa được tích hợp.** Đây là bản demo với ảnh GenAI tạo trước và quy trình xử lý ảnh tự động.

## Bot và các chỉ số

- **BFS** là đối chứng ít lượt di chuyển nhất.
- **A* · Ít bước đi** tối ưu số bước đi, rồi tối ưu số lần đẩy khi số bước bằng nhau.
- **A* · Ít lượt đẩy** tối ưu số lần đẩy, rồi tối ưu số bước đi khi số lần đẩy bằng nhau.
- A* tìm kiếm trên các lần đẩy. Mỗi cạnh gồm đường đi bộ ngắn nhất tới một cạnh thùng và một lần đẩy; heuristic là ghép thùng–đích với khoảng cách đẩy ngược trên bản đồ tĩnh. Giữ vị trí robot chính xác để không mất bảo đảm về số bước đi.
- **RL · Q-learning** học Q-table qua tương tác với môi trường Gymnasium. Policy tự chọn thùng và hướng đẩy; không dùng đáp án của A*/BFS và không chuyển ngầm sang solver khi thất bại.
- Engine chung được dùng cho người chơi, solver và kiểm tra lại lời giải.
- A* loại các ô không thể đẩy thùng về bất kỳ đích phù hợp nào theo bản đồ tĩnh; BFS loại sớm các góc kẹt không có đích.
- `solved`: đã tìm và xác minh lời giải.
- `no_solution`: có chứng minh góc kẹt hoặc đã duyệt hết không gian hợp lệ.
- `limit_reached`: vượt thời gian/số trạng thái, chưa kết luận được tính giải được.
- `untrained`: chưa có checkpoint RL cho khung tường/đích này.
- `policy_failed`: policy lặp, gặp trạng thái chưa học hoặc không hoàn thành. Trạng thái này không chứng minh màn vô nghiệm.
- Thời gian tính bằng `perf_counter`, không bao gồm thời gian vẽ giao diện.
- Số bước, số lần đẩy, số trạng thái và thời gian là dữ liệu thực tế. Chưa hiệu chuẩn thành độ khó của người chơi.

Generator thử các vị trí trên khung tường định sẵn và kiểm tra bằng solver, có giới hạn thời gian/lần thử. Nếu hết giới hạn, app ghi rõ việc dùng màn mẫu (`source=preset_fallback`). Seed tái hiện quá trình lấy mẫu; trên máy quá chậm, giới hạn thời gian có thể khiến generator dừng ở kết quả khác.

## Huấn luyện RL thực tế

Mở **Huấn luyện RL**, chọn trạng thái ban đầu hoặc trạng thái đang chơi, số episode, seed và giới hạn thời gian; bấm **Huấn luyện Q-learning**. Biểu đồ là số liệu được ghi trong quá trình học, mỗi 500 episode.

- Observation: bốn lớp nhị phân cho tường, đích, thùng và robot.
- Action: chỉ số thùng theo tọa độ đã sắp xếp + hướng đẩy, với action mask theo khả năng đi bộ/đẩy hợp lệ.
- Reward: `−1/lượt đẩy −0.1/bước đi bộ`, cộng potential shaping theo khoảng cách; `+50` khi thắng và `−30` khi kẹt góc hoặc hết hành động.
- Q update: `Q(s,a) += alpha * (reward + gamma * max Q(s',a') - Q(s,a))`. Trạng thái terminal không bootstrap; truncation do giới hạn thời gian vẫn bootstrap.
- Exploration: epsilon-greedy, giảm từ 1 xuống 0.05. Mặc định alpha 0.3, gamma 0.98.
- Curriculum: tạo trạng thái bắt đầu bằng các lần kéo ngược hợp lệ từ trạng thái đã hoàn thành; không dùng nhãn hành động của solver.
- Policy giữ vị trí robot chính xác vì bước đi bộ có chi phí trong reward.
- Checkpoint JSON và log được lưu ở `models/`, theo hash của vị trí tường/đích.

Q-learning ở đây là **tabular RL theo từng khung tường/đích**, chưa phải DQN/PPO tổng quát qua nhiều level. Trạng thái mới có thể chưa được học; dùng lựa chọn huấn luyện từ vị trí đang chơi. Tập kiểm tra các vị trí reset nằm trong cùng bản đồ và không phải bằng chứng tổng quát sang level mới. Kết quả RL không được gắn nhãn bảo đảm tối ưu.

Huấn luyện lại bằng dòng lệnh:

```powershell
python scripts/train_rl.py --seed 60 --template "Lối vòng" --episodes 50000
```

Ở seed 60/Lối vòng, policy đã được huấn luyện 50.000 episode và giải được 30 bước/8 lần đẩy. Trên cùng trạng thái, BFS duyệt 2.788 trạng thái từng bước, A* ưu tiên bước đi duyệt 52 trạng thái sau lần đẩy. Các đơn vị đếm khác nhau; xem thời gian đo thật trong bảng so sánh để đánh giá chi phí chạy.

## File xuất

| File | Nội dung |
|---|---|
| `level.json` | Bản đồ **ban đầu**, nguồn và seed; có thể nhập lại qua thanh bên. |
| `report.json` | Trạng thái đang chơi, số lượt chơi và số liệu tìm kiếm; `search.start_state` là gốc lời giải. |
| `preview.png` | Ảnh màn chơi tại thời điểm xuất. |
| `solution.gif` | Hoạt ảnh từ gốc lời giải, chỉ có khi bot đã giải được. |
| `assets/*.png` | Bộ tile đã chuẩn hóa đang dùng. |
| `assets/prompts.json` | Prompt và nguồn gốc, đánh dấu riêng ảnh bạn tải lên. |

Gói mẫu chạy thật nằm trong `examples/`. Giao diện và lịch sử dùng bộ nhớ phiên Streamlit; tải lại trang hoặc dừng server có thể làm mất phiên. Tải file để giữ kết quả.

## Cấu trúc

```text
app.py                 Studio / Huấn luyện RL / Bộ asset / Kiểm thử
puzzle/engine.py       Luật game và định dạng bản đồ
puzzle/solver.py       BFS và A* với hai mục tiêu tối ưu
puzzle/planning.py     Hành động đẩy, đường đi bộ và heuristic
puzzle/rl_env.py       Môi trường Gymnasium có action mask
puzzle/q_learning.py   Huấn luyện, suy luận và checkpoint Q-table
puzzle/experiments.py  Đối chiếu bốn agent trên cùng trạng thái
puzzle/generator.py    Tạo màn có seed và xác thực
puzzle/art.py          Nạp/chuẩn hóa ảnh và quản lý nguồn gốc
puzzle/rendering.py    Renderer PNG và hoạt ảnh GIF
puzzle/reports.py      Báo cáo JSON và gói ZIP
assets/                Ảnh gốc, tile chuẩn hóa và prompt
models/                Checkpoint Q-learning và log huấn luyện
tests/                 Kiểm thử engine, solver, generator, xuất file và giao diện
DEMO.md                Kịch bản trình bày 60–90 giây
```

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Trong workspace Codex hiện tại, có thể dùng `python -m pytest -q`; `tests/conftest.py` sẽ nạp bộ thư viện riêng của workspace nếu tồn tại. AppTest trên Windows cần quyền tạo socket nội bộ cho vòng lặp sự kiện của Streamlit.

## Hướng phát triển

1. Nối ComfyUI: gửi workflow `/prompt`, chờ `/history/{prompt_id}`, lấy ảnh `/view`; giữ app hoạt động khi tạo ảnh lỗi hoặc chậm.
2. Thu thập lượt chơi để kiểm chứng cách ước lượng độ khó.
3. Thử DQN/PPO với tập train/test tách theo level để đánh giá khả năng tổng quát.

Tài liệu liên quan: [Gymnasium Env](https://gymnasium.farama.org/api/env/), [Q-learning — Watkins và Dayan](https://doi.org/10.1007/BF00992698), [Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state), [ComfyUI Server API](https://docs.comfy.org/development/comfyui-server/comms_routes).

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path
import random

from PIL import Image
import streamlit as st

from puzzle.art import ASSET_DIR, ASSET_KEYS, ASSET_LABELS, asset_manifest, load_assets, normalize_asset
from puzzle.engine import ARROWS, PRESETS, Level, corner_deadlock, from_dict, step, to_ascii, won
from puzzle.generator import TEMPLATES, generate
from puzzle.rendering import png_bytes, render_board, replay_gif
from puzzle.reports import build_report, export_bundle, json_bytes
from puzzle.solver import replay, solve
from puzzle.q_learning import checkpoint_path, load_model, rollout, save_model, train_q
from puzzle.experiments import compare_agents

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Puzzle Lab · AI Puzzle Studio", page_icon="🧩", layout="wide",
                   initial_sidebar_state="auto")
ss = st.session_state
AGENTS = {"astar_moves": "A* · Ít bước đi", "astar_pushes": "A* · Ít lượt đẩy",
          "rl": "RL · Q-learning", "bfs": "BFS · Đối chiếu"}
if "pending_seed" in ss:
    ss.seed_input = ss.pop("pending_seed")

st.markdown("""
<style>
:root { --ink:#252b44; --muted:#7d849b; --violet:#6b5ce7; }
.stApp { background:#f6f7fb; }
.block-container { max-width:1390px; padding:2.1rem 2.7rem 2rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stAppDeployButton"] { display:none; }
h1,h2,h3 { letter-spacing:-.045em; color:#22283f; }
h1 { font-size:2.3rem !important; font-weight:750 !important; padding-top:.35rem !important; }
h3 { font-size:1.05rem !important; font-weight:700 !important; letter-spacing:-.02em; }
p, label { font-size:.91rem; }
[data-testid="stSidebar"] { background:#1d2239; border-right:none; width:245px !important; }
[data-testid="stSidebar"] * { color:#d4d8ec; }
[data-testid="stSidebar"] .stCaption p { color:#9098b7; font-size:.76rem; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#aab1ce; font-size:.78rem; }
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [role="combobox"] { color:#252b44 !important; }
[data-testid="stSidebar"] .stButton button { background:#303750; border:1px solid #424a67; color:#e3e7f7; }
[data-testid="stSidebar"] .stButton button p { color:#e3e7f7; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background:transparent; border-color:#3a4059; }
[data-testid="stSidebar"] hr { border-color:#343b55; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.3rem 0; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#e6e8f0; border-radius:14px; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:14px; }
.stButton button, .stDownloadButton button { border-radius:8px; font-weight:600; min-height:2.6rem; }
.stButton button[kind="primary"] { background:#6b5ce7; border-color:#6b5ce7; }
[data-testid="stMetricValue"] { font-size:1.7rem; font-weight:720; color:#262e48; }
[data-testid="stMetricLabel"] p { color:#80879b; font-size:.78rem; }
[data-testid="stImage"] img { border-radius:12px; }
[data-testid="stCaptionContainer"] p { color:#848ba0; }
.eyebrow { color:#939ab0; font-weight:700; font-size:.69rem; letter-spacing:.15em; margin-bottom:.5rem; }
.subtitle { color:#82899c; font-size:.91rem; margin-top:-.2rem; line-height:1.6; }
.brand { display:flex; align-items:center; gap:10px; font-size:1.25rem; font-weight:750; padding:4px 0 3px; color:white; }
.brand-mark { background:#8172f1; color:white !important; border-radius:11px; width:37px; height:37px; display:grid; place-items:center; font-size:24px; }
.brand-sub { color:#848daa !important; font-size:.67rem; letter-spacing:.15em; padding:3px 0 27px 47px; }
.side-heading { color:#7e89ac !important; font-size:.66rem; letter-spacing:.15em; font-weight:700; margin:18px 0 8px; }
.sidebar-note { border:1px solid #39415d; border-radius:10px; padding:14px; margin-top:22px; font-size:.79rem; line-height:1.65; color:#9fa8c7 !important; }
.sidebar-note b { color:#e4e8f5 !important; }
.status { display:inline-block; border:1px solid #d5ebe4; background:#edf8f3; color:#39836e; padding:6px 11px; border-radius:20px; font-size:.7rem; font-weight:600; }
.chip { display:inline-block; padding:4px 9px; background:#f0edff; color:#7c6ad7; border-radius:6px; font-size:.66rem; font-weight:650; margin-right:5px; }
.chip.gray { background:#f0f2f7; color:#8990a3; }
.section-label { color:#9299ae; font-size:.64rem; font-weight:700; letter-spacing:.12em; margin:2px 0 8px; }
.card-title { font-weight:720; font-size:1rem; color:#2c334c; margin:0 0 3px; }
.legend { display:flex; justify-content:center; flex-wrap:wrap; gap:18px; color:#8c93a6; font-size:.73rem; margin:8px 0; }
.dot { display:inline-block; width:8px; height:8px; border-radius:3px; margin-right:5px; }
.tip { background:#f7f6fe; border:1px solid #eeebfb; border-radius:10px; padding:14px; color:#8a83a7; font-size:.78rem; line-height:1.65; margin-top:10px; }
.empty-result { padding:17px 0; color:#8a91a5; font-size:.82rem; line-height:1.7; }
.footer-line { border-top:1px solid #e5e8ef; margin-top:24px; padding-top:14px; color:#a1a7b7; font-size:.72rem; display:flex; justify-content:space-between; }
div.st-key-board_card { background:#fff; border:1px solid #e5e8ef; border-radius:16px; padding:20px; }
div.st-key-controls_card { background:#fff; border:1px solid #e5e8ef; border-radius:16px; padding:20px; }
div.st-key-generator_card { background:#fff; border:1px solid #e5e8ef; border-radius:13px; padding:14px 18px 5px; margin:17px 0 20px; }
div.st-key-controls_card .stButton button { min-height:2.35rem; }
div.st-key-downloads .stDownloadButton button { font-size:.8rem; }
@media(max-width:900px) { .block-container { padding:1.5rem 1rem; } h1 { font-size:1.8rem !important; } }
@media(max-width:768px) {
 div.st-key-generator_card [data-testid="stHorizontalBlock"] { display:grid !important; grid-template-columns:1.4fr .7fr 1.1fr 1.2fr 36px; gap:8px; }
 div.st-key-generator_card [data-testid="stColumn"] { min-width:0 !important; width:auto !important; }
 div.st-key-controls_card [data-testid="stHorizontalBlock"] { flex-direction:row !important; flex-wrap:nowrap !important; }
 div.st-key-controls_card [data-testid="stColumn"] { flex:1 1 0 !important; min-width:0 !important; width:0 !important; }
 div.st-key-asset_gallery [data-testid="stHorizontalBlock"] { flex-direction:row !important; flex-wrap:wrap; }
 div.st-key-asset_gallery [data-testid="stColumn"] { flex:1 1 120px !important; min-width:100px !important; max-width:180px !important; width:auto !important; }
}
</style>
""", unsafe_allow_html=True)


def set_level(level: Level, *, verified: bool = False, note: str = "") -> None:
    ss.level = level
    ss.board = level.initial
    ss.moves = 0
    ss.pushes = 0
    ss.history = []
    ss.result = None
    ss.replay_index = 0
    ss.replay_base = (0, 0)
    ss.replay_history = []
    ss.autoplay = False
    ss.verified = verified
    ss.notice = note
    ss.comparison = []


def clear_search() -> None:
    ss.result = None
    ss.replay_index = 0
    ss.autoplay = False
    ss.comparison = []


def move_player(action: str) -> None:
    movement = step(ss.level, ss.board, action)
    if movement.valid:
        ss.history.append((ss.board, ss.moves, ss.pushes))
        ss.board = movement.state
        ss.moves += 1
        ss.pushes += int(movement.pushed)
        clear_search()
        ss.notice = ""
    else:
        ss.notice = movement.reason


def undo() -> None:
    if ss.history:
        ss.board, ss.moves, ss.pushes = ss.history.pop()
        clear_search()
        ss.notice = ""


def reset() -> None:
    set_level(ss.level, verified=ss.verified)


def run_search() -> None:
    ss.autoplay = False
    agent = ss.get("agent_choice", "astar_moves")
    if agent == "rl":
        ss.result = rollout(ss.level, load_model(ss.level), ss.board)
    else:
        ss.result = solve(ss.level, ss.board, timeout=float(ss.get("solver_seconds", 3)),
                          max_states=int(ss.get("solver_states", 60_000)), algorithm=agent)
    if ss.result.solved:
        replay(ss.level, ss.result)  # Validate every returned transition before showing a result.
    ss.replay_index = 0
    ss.replay_base = (ss.moves, ss.pushes)
    ss.replay_history = list(ss.history)
    ss.notice = ""
    row = {"Màn": ss.level.title, "Nguồn": ss.level.source, "Agent": ss.result.algorithm,
           "Kết quả": ss.result.status, "Bước": len(ss.result.actions) if ss.result.solved else None,
           "Đẩy thùng": ss.result.pushes if ss.result.solved else None,
           "Trạng thái duyệt": ss.result.explored, "Thời gian (ms)": round(ss.result.elapsed_ms, 3)}
    ss.search_history.insert(0, row)
    ss.search_history = ss.search_history[:100]


def advance() -> None:
    result = ss.result
    if result is None or not result.solved or ss.replay_index >= len(result.actions):
        ss.autoplay = False
        return
    movement = step(ss.level, ss.board, result.actions[ss.replay_index])
    if not movement.valid:
        clear_search()
        ss.notice = "Màn chơi đã thay đổi. Hãy chạy bot lại."
        return
    ss.history.append((ss.board, ss.moves, ss.pushes))
    ss.board = movement.state
    ss.moves += 1
    ss.pushes += int(movement.pushed)
    ss.replay_index += 1
    if ss.replay_index >= len(result.actions):
        ss.autoplay = False


def rewind() -> None:
    if ss.result:
        ss.board = ss.result.start
        ss.moves, ss.pushes = ss.replay_base
        ss.history = list(ss.replay_history)
        ss.replay_index = 0
        ss.autoplay = False


if "level" not in ss:
    first = generate(42)
    set_level(first.level, verified=True, note=first.note)
for key, default in {"overrides": {}, "search_history": [], "skin": "GenAI", "seed_input": 42,
                     "bench_rows": [], "asset_audit": [], "comparison": []}.items():
    if key not in ss:
        ss[key] = default
# Existing browser sessions can retain instances from the previous solver class.
# Discard only the old result when upgrading; keep the user's board and moves.
if ss.result is not None and not hasattr(ss.result, "algorithm"):
    clear_search()
if ss.get("navigation_state_version") != 2:
    if ss.level.seed is not None:
        ss.seed_input = ss.level.seed
    inner_walls = frozenset((r, c) for r, c in ss.level.walls
                            if r not in (0, ss.level.rows - 1) and c not in (0, ss.level.cols - 1))
    for name, (rows, cols, obstacles) in TEMPLATES.items():
        if (rows, cols, obstacles) == (ss.level.rows, ss.level.cols, inner_walls):
            ss.template = name
    ss.box_count = min(2, len(ss.level.goals))
    ss.navigation_state_version = 2
# Preserve widget-backed settings while their page is not being rendered.
for key in ("seed_input", "template", "box_count", "skin", "agent_choice", "rl_episodes", "rl_budget", "rl_seed", "rl_start"):
    if key in ss:
        ss[key] = ss[key]

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">▧</span>Puzzle Lab<span style="font-size:10px;color:#9386f7">BETA</span></div>'
                '<div class="brand-sub">AI PUZZLE STUDIO</div>', unsafe_allow_html=True)
    page = st.radio("Điều hướng", ["Studio", "Huấn luyện RL", "Bộ asset", "Kiểm thử"], key="navigation", label_visibility="collapsed")
    st.markdown('<div class="side-heading">THƯ VIỆN LEVEL</div>', unsafe_allow_html=True)
    preset_index = st.selectbox("Màn mẫu", range(len(PRESETS)), format_func=lambda i: PRESETS[i].title,
                                label_visibility="collapsed", key="preset_index")
    if st.button("Mở màn mẫu", width="stretch", key="load_preset"):
        selected = PRESETS[preset_index]
        set_level(selected, verified=solve(selected).solved)
        st.rerun()
    with st.expander("Nhập bản đồ JSON"):
        incoming = st.file_uploader("Chọn level.json", type=["json"], max_upload_size=1, key="level_upload")
        if st.button("Nạp bản đồ", disabled=incoming is None, key="import_level"):
            try:
                if incoming.size > 100_000:
                    raise ValueError("File bản đồ quá lớn.")
                loaded = from_dict(json.loads(incoming.getvalue()))
                set_level(loaded)
                st.rerun()
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))
    st.divider()
    with st.expander("Giới hạn tìm kiếm"):
        st.slider("Thời gian tối đa (giây)", 1, 10, 3, key="solver_seconds")
        st.select_slider("Số trạng thái tối đa", [10_000, 30_000, 60_000, 100_000], value=60_000, key="solver_states")
    st.markdown('<div class="sidebar-note"><b>Một quy trình, ba bước.</b><br>Tạo màn có lời giải, kiểm thử với bot và áp dụng đồ họa GenAI.</div>', unsafe_allow_html=True)
    st.caption("A* · Q-learning · Python\n\nCheckpoint RL được lưu trên máy.")


def header(title: str, subtitle: str, eyebrow: str) -> None:
    left, right = st.columns([4, 1])
    with left:
        st.markdown(f'<div class="eyebrow">WORKSPACE / {eyebrow}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<span class="status">● &nbsp; Local demo</span>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_bundle(level, state, result, assets_png, moves, pushes, overrides, skin):
    assets = {key: Image.open(BytesIO(value)).convert("RGBA") for key, value in assets_png.items()}
    return export_bundle(level, state, result, assets=assets, moves=moves, pushes=pushes,
                         overrides=overrides, skin=skin)


def current_assets() -> dict:
    return load_assets(ss.overrides) if ss.skin == "GenAI" else {}


def downloads() -> None:
    assets = current_assets()
    report = build_report(ss.level, ss.board, ss.result, ss.moves, ss.pushes, ss.overrides, ss.skin)
    with st.expander("Xuất bản đồ, báo cáo và ảnh màn chơi", expanded=False):
        st.caption("Bản đồ là trạng thái ban đầu. Báo cáo ghi riêng trạng thái đang chơi và vị trí bắt đầu tìm kiếm.")
        columns = st.columns(4)
        with columns[0]:
            st.download_button("Bản đồ JSON", json_bytes(ss.level.to_dict()), "level.json", "application/json", width="stretch")
        with columns[1]:
            st.download_button("Báo cáo JSON", json_bytes(report), "report.json", "application/json", width="stretch")
        with columns[2]:
            st.download_button("Ảnh PNG", png_bytes(render_board(ss.level, ss.board, assets=assets)), "preview.png", "image/png", width="stretch")
        with columns[3]:
            payload = cached_bundle(ss.level, ss.board, ss.result,
                                    {key: png_bytes(value) for key, value in assets.items()},
                                    ss.moves, ss.pushes, ss.overrides, ss.skin)
            st.download_button("Tải toàn bộ ZIP", payload, "puzzle-lab-export.zip", "application/zip", width="stretch", type="primary")


if page == "Studio":
    header("Thiết kế. Kiểm thử. Chơi.",
           "Biến một bản đồ nhỏ thành trải nghiệm có thể chơi — và kiểm chứng từng bước.", "LEVEL DESIGN")
    with st.container(key="generator_card"):
        st.markdown('<div class="section-label">01 &nbsp; TẠO MÀN CHƠI</div>', unsafe_allow_html=True)
        g1, g2, g3, g4, g5 = st.columns([1.5, .8, 1, 1.1, .65], vertical_alignment="bottom")
        with g1:
            template = st.selectbox("Khung bản đồ", list(TEMPLATES), key="template")
        with g2:
            box_count = st.selectbox("Số thùng", [1, 2], index=1, key="box_count")
        with g3:
            seed = st.number_input("Seed", min_value=0, max_value=999999, step=1, key="seed_input")
        with g4:
            create = st.button("✦ Tạo màn", type="primary", width="stretch", key="generate_level")
        with g5:
            shuffle = st.button("↻", width="stretch", help="Đổi seed ngẫu nhiên", key="shuffle_seed")
        if shuffle:
            # Set the next widget value before the next script creates the widget.
            ss.pending_seed = random.SystemRandom().randint(0, 999999)
            st.rerun()
        if create:
            with st.spinner("Đang tạo màn và xác thực lời giải…"):
                generated = generate(int(seed), boxes=box_count, template=template)
            set_level(generated.level, verified=generated.verification.solved, note=generated.note)
            ss.generation_note = f"{generated.attempts} lần thử · {generated.elapsed_ms:.0f} ms"
            st.rerun()

    @st.fragment(run_every=.35 if ss.autoplay else None)
    def workspace():
        was_playing = ss.autoplay
        if was_playing:
            advance()
            if not ss.autoplay:
                st.rerun(scope="app")
        board_col, info_col = st.columns([1.65, 1], gap="large")
        with board_col, st.container(key="board_card"):
            title_col, skin_col = st.columns([1.1, 1])
            with title_col:
                st.markdown('<div class="section-label">02 &nbsp; SÂN CHƠI</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">{escape(ss.level.title)}</div>', unsafe_allow_html=True)
            with skin_col:
                st.radio("Đồ họa", ["GenAI", "Tối giản"], horizontal=True, key="skin", label_visibility="collapsed")
            tags = f'<span class="chip">{ss.level.rows} × {ss.level.cols}</span><span class="chip gray">{len(ss.level.goals)} THÙNG</span>'
            if ss.verified:
                tags += '<span class="chip" style="background:#edf8f3;color:#428a73">CÓ LỜI GIẢI</span>'
            st.markdown(tags, unsafe_allow_html=True)
            st.image(render_board(ss.level, ss.board, assets=current_assets()), width="stretch")
            st.markdown('<div class="legend"><span><i class="dot" style="background:#e8f1ed"></i>Robot</span>'
                        '<span><i class="dot" style="background:#f7ac50"></i>Thùng hàng</span>'
                        '<span><i class="dot" style="background:#ffd572;border-radius:50%"></i>Đích</span>'
                        '<span><i class="dot" style="background:#78dfbd"></i>Đã đến đích</span></div>', unsafe_allow_html=True)
            st.caption("Orbital Warehouse · Ảnh GenAI tạo sẵn" if ss.skin == "GenAI" else "Chế độ tối giản · Cùng bản đồ và luật chơi")
        with info_col, st.container(key="controls_card"):
            st.markdown('<div class="section-label">03 &nbsp; ĐIỀU KHIỂN & KIỂM THỬ</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Đưa mọi thùng về đích</div>', unsafe_allow_html=True)
            st.caption("Đẩy thùng từng ô. Robot không thể kéo thùng.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Bước đi", ss.moves)
            m2.metric("Lần đẩy", ss.pushes)
            m3.metric("Đúng đích", f"{len(ss.board.boxes & ss.level.goals)}/{len(ss.level.goals)}")
            if won(ss.level, ss.board):
                st.success("Hoàn thành! Mọi thùng đã về đúng đích.")
            elif corner_deadlock(ss.level, ss.board):
                st.warning("Thùng bị kẹt ở góc. Dùng Undo để thử hướng khác.")
            if ss.notice:
                st.info(ss.notice)
            u1, u2, u3 = st.columns(3)
            with u2:
                if st.button("↑", width="stretch", key="move_U", disabled=ss.autoplay, help="Đi lên"):
                    move_player("U"); st.rerun(scope="app")
            d1, d2, d3 = st.columns(3)
            for col, action in zip((d1, d2, d3), ("L", "D", "R")):
                with col:
                    if st.button(ARROWS[action], width="stretch", key=f"move_{action}", disabled=ss.autoplay,
                                 help={"L": "Sang trái", "D": "Đi xuống", "R": "Sang phải"}[action]):
                        move_player(action); st.rerun(scope="app")
            r1, r2 = st.columns(2)
            with r1:
                if st.button("↶ Undo", width="stretch", disabled=not ss.history or ss.autoplay, key="undo"):
                    undo(); st.rerun(scope="app")
            with r2:
                if st.button("↻ Chơi lại", width="stretch", key="reset"):
                    reset(); st.rerun(scope="app")
            st.divider()
            st.selectbox("Agent điều khiển", list(AGENTS), format_func=AGENTS.get,
                         key="agent_choice", on_change=clear_search)
            if ss.agent_choice == "rl":
                if checkpoint_path(ss.level).exists():
                    st.caption("Đã có checkpoint cho khung tường/đích này. RL tự chọn lần đẩy, không gọi solver giải thay.")
                else:
                    st.info("Chưa có checkpoint cho màn này. Mở Huấn luyện RL ở thanh bên để bắt đầu.")
            if st.button("✦ Bot giải màn", type="primary", width="stretch", key="solve", disabled=ss.autoplay):
                run_search(); st.rerun(scope="app")
            result = ss.result
            if result is None:
                st.markdown('<div class="empty-result">Bot sẽ tìm đường từ vị trí hiện tại và ghi lại toàn bộ kết quả kiểm thử.</div>', unsafe_allow_html=True)
            elif result.solved:
                st.markdown(f'<span class="status">✓ &nbsp; Tìm thấy lời giải · {len(result.actions)} bước</span>', unsafe_allow_html=True)
                a, b = st.columns(2)
                a.metric("Lượt chọn đẩy" if "Q-learning" in result.algorithm else "Trạng thái duyệt", f"{result.explored:,}")
                b.metric("Thời gian", f"{result.elapsed_ms:.2f} ms")
                guarantees = {"player_moves": "Ít bước đi nhất",
                              "player_moves_then_pushes": "Ít bước đi nhất, sau đó ít lượt đẩy",
                              "pushes_then_player_moves": "Ít lượt đẩy nhất, sau đó ít bước đi"}
                st.caption(result.algorithm)
                st.caption(f"{result.pushes} lần đẩy thùng · {guarantees.get(result.optimal_for, 'Policy học từ reward; không bảo đảm lời giải tối ưu')}.")
                p1, p2, p3 = st.columns(3)
                with p1:
                    if st.button("Ⅱ Dừng" if ss.autoplay else "▶ Phát", width="stretch", key="play_button",
                                 disabled=not result.actions):
                        if ss.replay_index == len(result.actions):
                            rewind()
                        ss.autoplay = not ss.autoplay
                        st.rerun(scope="app")
                with p2:
                    if st.button("Bước →", width="stretch", key="next_step", disabled=ss.autoplay or ss.replay_index >= len(result.actions)):
                        advance(); st.rerun(scope="app")
                with p3:
                    if st.button("|← Về đầu", width="stretch", key="rewind", disabled=ss.autoplay):
                        rewind(); st.rerun(scope="app")
                if result.actions:
                    st.progress(ss.replay_index / len(result.actions), text=f"Lời giải: {ss.replay_index}/{len(result.actions)} bước")
                    st.caption(" ".join(ARROWS[a] for a in result.actions))
                with st.expander("Tải hoạt ảnh lời giải"):
                    st.download_button("GIF lời giải", replay_gif(ss.level, result, current_assets()),
                                       "solution.gif", "image/gif", width="stretch")
            elif result.status == "no_solution":
                st.error("Không có lời giải từ vị trí này.")
                st.caption(result.reason)
            elif result.status in ("untrained", "policy_failed"):
                st.warning("RL chưa giải được từ vị trí này.")
                st.caption(result.reason)
                st.caption("Huấn luyện thêm trên màn/vị trí hiện tại, hoặc chọn A* để kiểm chứng riêng.")
            else:
                st.warning("Đã chạm giới hạn tìm kiếm.")
                st.caption(result.reason)
            st.markdown('<div class="tip">✧ <b>Mẹo thiết kế</b><br>Tránh đẩy thùng vào góc không có đích. Đổi seed để thử một bố cục khác.</div>', unsafe_allow_html=True)
        if not ss.autoplay:
            downloads()

    workspace()

elif page == "Huấn luyện RL":
    ss.autoplay = False
    header("Agent học qua từng lượt thử.",
           "Q-learning trên môi trường Gymnasium: thử hành động, nhận reward và cập nhật Q-table.", "REINFORCEMENT LEARNING")
    st.info("Policy học trên khung tường và vị trí đích của màn hiện tại. Màn có đích/tường khác cần huấn luyện riêng. Đây là tabular RL, không phải mô hình đã học tổng quát mọi level.")
    setup, preview = st.columns([1.5, 1])
    with setup, st.container(border=True):
        st.subheader("Huấn luyện trên màn hiện tại")
        st.caption(f"{ss.level.title} · {ss.level.rows}×{ss.level.cols} · {len(ss.level.goals)} thùng")
        rl_start = st.radio("Vị trí bắt đầu", ["Màn ban đầu", "Vị trí đang chơi"], horizontal=True, key="rl_start")
        episodes = st.slider("Số episode", 1000, 100000, 50000, step=1000, key="rl_episodes")
        c1, c2 = st.columns(2)
        with c1:
            training_seed = st.number_input("Seed huấn luyện", min_value=0, max_value=999999, value=42, key="rl_seed")
        with c2:
            training_budget = st.slider("Giới hạn thời gian (giây)", 5, 60, 20, step=5, key="rl_budget")
        st.caption("Reward: −1/lượt đẩy, −0.1/bước đi bộ, +50 hoàn thành, −30 kẹt thùng, có shaping theo khoảng cách. Khởi tạo từ các thế kéo ngược hợp lệ để học dần; không cung cấp đáp án A*/BFS.")
        train = st.button("✦ Huấn luyện Q-learning", type="primary", width="stretch", key="train_rl")
    with preview, st.container(border=True):
        st.subheader("Màn đang học")
        st.image(render_board(ss.level, ss.level.initial if rl_start == "Màn ban đầu" else ss.board,
                              assets=current_assets()), width="stretch")
        st.caption("Agent chọn thùng + hướng đẩy. BFS cục bộ chỉ tìm đường đi bộ tới cạnh thùng được chọn.")
    if train:
        progress_bar = st.progress(0.0, text="Khởi tạo môi trường và curriculum…")
        def progress(episode, total, row):
            progress_bar.progress(episode / total,
                                  text=f"Episode {episode:,}/{total:,} · reward TB {row['mean_reward']:.1f} · epsilon {row['epsilon']:.2f}")
        try:
            model = train_q(ss.level, ss.level.initial if rl_start == "Màn ban đầu" else ss.board,
                            episodes=episodes, seed=int(training_seed), budget=float(training_budget), on_progress=progress)
            saved = save_model(ss.level, model)
            progress_bar.progress(1.0, text=f"Đã học {model.training['episodes']:,} episode và lưu checkpoint.")
            clear_search()
            st.success(f"Đã lưu {saved.name}. Về Studio, chọn RL · Q-learning để cho agent chơi.")
        except ValueError as exc:
            st.error(str(exc))
    model = load_model(ss.level)
    if model:
        stats = model.training
        st.subheader("Kết quả huấn luyện đã lưu")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Episode thực chạy", f"{stats['episodes']:,}")
        m2.metric("Trạng thái đã học", f"{stats['learned_states']:,}")
        m3.metric("Màn bắt đầu", "Giải được" if stats['root_greedy_solved'] else "Chưa giải")
        m4.metric("Thời gian học", f"{stats['elapsed_seconds']:.2f} s")
        if model.history:
            curves = {"Episode": [r["episode"] for r in model.history],
                      "Reward trung bình": [r["mean_reward"] for r in model.history],
                      "Tỉ lệ thắng khi train (%)": [100 * r["training_success_rate"] for r in model.history]}
            ch1, ch2 = st.columns(2)
            with ch1:
                st.caption("Reward trung bình mỗi nhóm episode")
                st.line_chart(curves, x="Episode", y="Reward trung bình", color="#6b5ce7")
            with ch2:
                st.caption("Có exploration; phân bố curriculum thay đổi khi học")
                st.line_chart(curves, x="Episode", y="Tỉ lệ thắng khi train (%)", color="#39967c")
        baseline = stats["random_policy_baseline"]
        st.write(f"**Đối chứng policy ngẫu nhiên:** giải {baseline['successes']}/{baseline['episodes']} lượt thử từ cùng vị trí bắt đầu.")
        st.write(f"**Greedy policy sau học:** {'giải được' if stats['root_greedy_solved'] else 'chưa giải được'} màn bắt đầu"
                 + (f", {stats['root_greedy_moves']} bước / {stats['root_greedy_pushes']} lượt đẩy." if stats['root_greedy_solved'] else "."))
        st.caption(f"Kiểm tra các vị trí reset riêng: {stats['held_out_reset_successes']}/{stats['held_out_reset_states']}. Cùng khung tường/đích; trạng thái có thể đã được gặp khi train. Đây không phải kết quả trên level mới.")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Tải checkpoint Q-table", json_bytes(model.to_dict()), checkpoint_path(ss.level).name,
                               "application/json", width="stretch")
        with d2:
            st.download_button("Tải log huấn luyện", json_bytes({"training": stats, "history": model.history}),
                               "rl-training.json", "application/json", width="stretch")
        with st.expander("Thông số và phạm vi mô hình"):
            st.json(stats)
    else:
        st.caption("Chưa có checkpoint cho màn này. Chọn Huấn luyện để tạo policy từ đầu.")

elif page == "Bộ asset":
    ss.autoplay = False
    header("Một thế giới, cùng phong cách.",
           "Bộ ảnh GenAI tạo sẵn → chuẩn hóa kích thước → ghép vào cùng một màn chơi.", "GENAI ASSETS")
    st.info("Bộ Orbital Warehouse được tạo bằng image_gen. App xử lý và áp dụng ảnh; tạo ảnh trực tiếp qua ComfyUI là phần mở rộng tiếp theo.")
    assets = load_assets(ss.overrides, size=144)
    columns = st.container(key="asset_gallery").columns(5)
    for col, key in zip(columns, ASSET_KEYS):
        with col, st.container(border=True):
            st.image(assets[key], width="stretch")
            st.markdown(f"**{ASSET_LABELS[key]}**")
            st.caption(f"{key}.png · 64 × 64 · RGBA")
            if key in ss.overrides:
                st.caption("Nguồn: ảnh bạn tải lên")
    st.write("")
    left, right = st.columns([1.2, 1])
    with left, st.container(border=True):
        st.subheader("Nạp bộ đồ họa của bạn")
        st.caption("Đặt tên file: floor.png, wall.png, goal.png, box.png, player.png. Có thể thay từng ảnh.")
        uploads = st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, max_upload_size=10, key="asset_uploads")
        if st.button("Chuẩn hóa & áp dụng", type="primary", key="apply_assets", disabled=not uploads):
            audit = []
            updated = dict(ss.overrides)
            try:
                keys = [Path(upload.name).stem.lower() for upload in uploads]
                if len(keys) != len(set(keys)):
                    raise ValueError("Mỗi loại asset chỉ được chọn một ảnh.")
                for upload, key in zip(uploads, keys):
                    if key not in ASSET_KEYS:
                        raise ValueError(f"Tên ảnh cần là một trong: {', '.join(ASSET_KEYS)}.")
                    data = upload.getvalue()
                    normalized = normalize_asset(data, key)
                    with Image.open(BytesIO(data)) as original:
                        original_size = f"{original.width} × {original.height}"
                    updated[key] = normalized
                    audit.append({"File": upload.name, "Ban đầu": original_size, "Sau xử lý": "64 × 64 RGBA", "Kích thước (KB)": round(len(normalized) / 1024, 1)})
                ss.overrides = updated
                ss.asset_audit = audit
                ss.skin = "GenAI"
                st.success("Đã chuẩn hóa và áp dụng ảnh. Mở Studio để xem.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Khôi phục bộ mặc định", key="restore_assets", disabled=not ss.overrides):
            ss.overrides = {}; ss.asset_audit = []; st.rerun()
        if ss.asset_audit:
            st.dataframe(ss.asset_audit, hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.subheader("Xem trước trên màn hiện tại")
        st.image(render_board(ss.level, ss.board, assets=load_assets(ss.overrides)), width="stretch")
    with st.expander("Prompt & nguồn gốc bộ ảnh"):
        st.json(asset_manifest(ss.overrides))
        st.download_button("Tải prompt JSON", json_bytes(asset_manifest(ss.overrides)), "prompts.json", "application/json")

elif page == "Kiểm thử":
    ss.autoplay = False
    header("Đo lường trước khi kết luận.",
           "Chạy kiểm thử trên các màn mẫu và xem lịch sử tìm kiếm của phiên làm việc.", "GAME TESTING")
    with st.container(border=True):
        st.subheader("BFS, A* và RL trên cùng màn chơi")
        start_choice = st.radio("So sánh từ", ["Màn ban đầu", "Vị trí đang chơi"], horizontal=True, key="comparison_start")
        st.caption(f"Màn hiện tại: {ss.level.title}. RL dùng checkpoint đã học của đúng khung tường/đích.")
        if st.button("So sánh 4 agent", type="primary", key="compare_agents"):
            with st.spinner("Đang chạy các agent từ cùng một vị trí…"):
                rows = compare_agents(ss.level, ss.level.initial if start_choice == "Màn ban đầu" else ss.board,
                                      timeout=float(ss.get("solver_seconds", 3)), max_states=int(ss.get("solver_states", 60_000)))
            ss.comparison = rows
            ss.comparison_origin = {"level": ss.level.to_dict(), "state": (ss.level.initial if start_choice == "Màn ban đầu" else ss.board).to_dict()}
        if ss.comparison:
            st.dataframe(ss.comparison, hide_index=True, width="stretch")
            st.download_button("Tải bảng so sánh JSON", json_bytes({"origin": ss.comparison_origin, "results": ss.comparison}),
                               "agent-comparison.json", "application/json")
            st.caption("BFS đếm trạng thái từng bước; A* đếm trạng thái sau lần đẩy; RL đếm lần chọn đẩy. Thời gian chỉ gồm tìm đường/suy luận, không gồm huấn luyện. RL không được gắn nhãn tối ưu.")
    left, right = st.columns([2, 1], vertical_alignment="bottom")
    with left:
        st.markdown("**Bộ kiểm thử mẫu**")
        st.caption("Mỗi lời giải được chạy lại qua luật game để xác nhận màn đã hoàn thành.")
    with right:
        batch = st.button("▶ Chạy 3 màn mẫu", type="primary", width="stretch", key="run_benchmark")
    if batch:
        rows = []
        for level in PRESETS:
            result = solve(level, timeout=float(ss.get("solver_seconds", 3)), max_states=int(ss.get("solver_states", 60_000)))
            verified = result.solved and won(level, replay(level, result)[-1])
            rows.append({"Màn": level.title, "Kết quả": result.status, "Xác minh lời giải": "Đạt" if verified else "Chưa đạt",
                         "Bước": len(result.actions) if result.solved else None, "Đẩy thùng": result.pushes if result.solved else None,
                         "Trạng thái duyệt": result.explored, "Thời gian (ms)": round(result.elapsed_ms, 3)})
        ss.bench_rows = rows
    if ss.bench_rows:
        st.dataframe(ss.bench_rows, hide_index=True, width="stretch")
        st.download_button("Tải kết quả kiểm thử", json_bytes(ss.bench_rows), "benchmark.json", "application/json")
    else:
        st.info("Bấm chạy kiểm thử để thu thập số liệu trên máy của bạn.")
    st.divider()
    st.subheader("Lịch sử chạy bot")
    if ss.search_history:
        st.dataframe(ss.search_history, hide_index=True, width="stretch")
        st.download_button("Tải lịch sử JSON", json_bytes(ss.search_history), "search-history.json", "application/json")
    else:
        st.caption("Các lần chạy bot trong Studio sẽ xuất hiện ở đây.")
    st.caption("Các chỉ số tìm kiếm chưa được hiệu chuẩn thành độ khó đối với người chơi. BFS tối ưu số lượt di chuyển, không mặc nhiên tối ưu số lần đẩy thùng.")

st.markdown('<div class="footer-line"><span>PUZZLE LAB &nbsp; / &nbsp; AI for level design & workflow automation</span><span>Prototype 02 · Search + RL</span></div>', unsafe_allow_html=True)

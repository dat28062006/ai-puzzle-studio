from pathlib import Path

from streamlit.testing.v1 import AppTest

from puzzle.engine import won

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, key):
    return next(item for item in app.button if item.key == key)


def test_play_undo_reset_solver_and_replay():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    initial = app.session_state["board"]
    button(app, "move_D").click().run()
    assert not app.exception
    assert app.session_state["moves"] == 1
    button(app, "undo").click().run()
    assert app.session_state["board"] == initial
    assert app.session_state["moves"] == 0
    button(app, "solve").click().run()
    assert not app.exception
    result = app.session_state["result"]
    assert result.solved and len(result.actions) == 10
    button(app, "next_step").click().run()
    assert app.session_state["replay_index"] == 1
    button(app, "rewind").click().run()
    assert app.session_state["board"] == initial
    for _ in result.actions:
        button(app, "next_step").click().run()
    assert not app.exception
    assert won(app.session_state["level"], app.session_state["board"])
    button(app, "reset").click().run()
    assert app.session_state["moves"] == 0 and app.session_state["result"] is None


def test_navigation_benchmark_generation_and_shuffle():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio(key="navigation").set_value("Bộ asset").run()
    assert not app.exception
    app.radio(key="navigation").set_value("Kiểm thử").run()
    button(app, "run_benchmark").click().run()
    assert not app.exception
    assert len(app.session_state["bench_rows"]) == 3
    assert all(row["Xác minh lời giải"] == "Đạt" for row in app.session_state["bench_rows"])
    app.radio(key="navigation").set_value("Studio").run()
    app.number_input(key="seed_input").set_value(17).run()
    button(app, "generate_level").click().run()
    assert not app.exception
    assert app.session_state["level"].seed == 17
    button(app, "shuffle_seed").click().run()
    assert not app.exception
    assert app.number_input(key="seed_input").value != 17


def test_rl_agent_checkpoint_and_training_page():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.selectbox(key="agent_choice").select("rl").run()
    button(app, "solve").click().run()
    assert not app.exception
    result = app.session_state["result"]
    assert result.solved and result.algorithm.startswith("RL") and result.optimal_for is None
    app.radio(key="navigation").set_value("Huấn luyện RL").run()
    assert not app.exception
    assert any("Agent học" in title.value for title in app.title)
    assert len(app.get("vega_lite_chart")) == 2
    app.radio(key="navigation").set_value("Kiểm thử").run()
    button(app, "compare_agents").click().run()
    assert not app.exception and len(app.session_state["comparison"]) == 4
    assert all(row["Kết quả"] == "solved" for row in app.session_state["comparison"])

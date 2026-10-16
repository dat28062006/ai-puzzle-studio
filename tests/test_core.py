from io import BytesIO
import json
import random
import zipfile

from PIL import Image
import pytest

from puzzle.art import ASSET_KEYS, asset_manifest, load_assets, normalize_asset
from puzzle.engine import DIRECTIONS, PRESETS, State, from_ascii, from_dict, step, to_ascii, won
from puzzle.generator import TEMPLATES, generate
from puzzle.rendering import png_bytes, render_board, replay_gif
from puzzle.reports import export_bundle
from puzzle.solver import replay, solve


def test_push_onto_and_off_goal_keeps_goal():
    level = from_ascii("#######\n#     #\n# @$. #\n#     #\n#######")
    first = step(level, level.initial, "R")
    assert first.valid and first.pushed and won(level, first.state)
    second = step(level, first.state, "R")
    assert second.valid and second.pushed and not won(level, second.state)
    assert level.goals == frozenset({(2, 4)})
    assert to_ascii(level, second.state).splitlines()[2][4] == "+"
    third = step(level, second.state, "D")
    assert to_ascii(level, third.state).splitlines()[2][4] == "."


def test_wall_and_two_boxes_are_immovable():
    level = from_ascii("#######\n#     #\n#@$$..#\n#     #\n#######")
    assert not step(level, level.initial, "L").valid
    assert not step(level, level.initial, "R").valid
    assert step(level, level.initial, "R").state == level.initial


@pytest.mark.parametrize("index,shortest", [(0, 1), (1, 5), (2, 9)])
def test_presets_have_known_shortest_paths(index, shortest):
    level = PRESETS[index]
    result = solve(level)
    assert result.solved and len(result.actions) == shortest
    states = replay(level, result)
    assert won(level, states[-1])
    pushes = sum(step(level, state, action).pushed for state, action in zip(states, result.actions))
    assert pushes == result.pushes


def test_limit_is_not_reported_as_unsolvable():
    result = solve(PRESETS[2], max_states=1)
    assert result.status == "limit_reached"
    assert solve(PRESETS[2], timeout=0).status == "limit_reached"


def test_static_corner_is_unsolvable():
    level = from_ascii("######\n#$ . #\n# @  #\n#    #\n######")
    assert solve(level).status == "no_solution"


def test_exhausted_search_is_unsolvable_without_a_corner():
    level = from_ascii("#######\n#  $  #\n#  .  #\n#  @  #\n#######")
    assert solve(level).status == "no_solution"


def test_search_from_current_position_not_original_position():
    level = PRESETS[2]
    original = solve(level)
    moved = step(level, level.initial, original.actions[0]).state
    result = solve(level, moved)
    assert result.start == moved
    assert len(result.actions) == len(original.actions) - 1
    assert won(level, replay(level, result)[-1])


def test_already_solved_needs_no_moves():
    level = from_ascii("#####\n#@* #\n#   #\n#####")
    result = solve(level)
    assert result.solved and result.actions == () and result.pushes == 0


def test_random_moves_preserve_game_invariants():
    rng = random.Random(3)
    for level in PRESETS:
        state = level.initial
        for _ in range(150):
            state = step(level, state, rng.choice(list(DIRECTIONS))).state
            assert state.player not in state.boxes
            assert not level.blocked(state.player)
            assert len(state.boxes) == len(level.goals)
            assert all(not level.blocked(box) for box in state.boxes)
            assert from_ascii(to_ascii(level, state)).initial == state


@pytest.mark.parametrize("template", list(TEMPLATES))
@pytest.mark.parametrize("boxes", [1, 2])
def test_generated_levels_can_be_replayed(template, boxes):
    generated = generate(42, boxes=boxes, template=template)
    assert len(generated.level.initial.boxes) == boxes
    assert generated.verification.solved
    assert won(generated.level, replay(generated.level, generated.verification)[-1])
    assert from_dict(generated.level.to_dict()).initial == generated.level.initial


def test_same_seed_reproduces_layout():
    one, two = generate(42), generate(42)
    assert one.level == two.level
    assert one.verification.actions == two.verification.actions


def test_generator_fallback_is_explicit():
    generated = generate(99, budget=0)
    assert generated.level.source == "preset_fallback"
    assert generated.note
    assert generated.verification.solved


@pytest.mark.parametrize("data", [
    {"grid": ["###", "#@#", "###"]},
    {"grid": ["#####", "#@$.#", "###"]},
    {"grid": ["#####", "#@@.#", "#$  #", "#####"]},
    {"grid": ["#####", "#@$.#", "#   #", "#   #"]},
    {"grid": ["#" * 11] * 11},
])
def test_bad_level_import_is_rejected(data):
    with pytest.raises(ValueError):
        from_dict(data)


def test_asset_normalization_preserves_transparency_and_ratio():
    source = Image.new("RGBA", (120, 60), (255, 0, 0, 255))
    data = normalize_asset(png_bytes(source), "player")
    image = Image.open(BytesIO(data))
    assert image.size == (64, 64) and image.mode == "RGBA"
    assert image.getpixel((32, 0))[3] == 0
    assert image.getpixel((32, 32))[3] == 255
    assert Image.open(BytesIO(normalize_asset(png_bytes(source), "floor"))).size == (64, 64)
    with pytest.raises(ValueError):
        normalize_asset(b"not an image", "box")


def test_all_genai_assets_exist_and_manifest_records_prompts():
    assets = load_assets()
    assert set(assets) == set(ASSET_KEYS)
    assert all(a.mode == "RGBA" for a in assets.values())
    assert all(asset_manifest()["assets"][key]["generation_prompt"] for key in ASSET_KEYS)
    assert asset_manifest({"player": b"override"})["assets"]["player"]["source"] == "user_upload"
    assert asset_manifest()["assets"]["player"]["source"] == "image_gen"


def test_export_roundtrip_and_animated_solution():
    level = PRESETS[1]
    result = solve(level)
    state = step(level, level.initial, result.actions[0]).state
    assets = load_assets()
    bundle = export_bundle(level, state, result, assets=assets, moves=1, pushes=1)
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        imported = from_dict(json.loads(archive.read("level.json")))
        assert imported.initial == level.initial
        report = json.loads(archive.read("report.json"))
        assert report["play"]["current_state"] == state.to_dict()
        assert report["search"]["start_state"] == level.initial.to_dict()
        assert report["search"]["moves"] == 5
        assert Image.open(BytesIO(archive.read("preview.png"))).size == (560, 560)
        gif = Image.open(BytesIO(archive.read("solution.gif")))
        assert gif.n_frames == len(result.actions) + 1
        assert "assets/player.png" in archive.namelist()


def test_skin_change_does_not_change_engine_state():
    level = PRESETS[1]
    state = level.initial
    simple = render_board(level, state)
    generated = render_board(level, state, assets=load_assets())
    assert simple.size == generated.size
    assert png_bytes(simple) != png_bytes(generated)
    assert level.initial == state

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from .engine import Level, State
from .solver import SearchResult, replay


def png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def render_board(level: Level, state: State | None = None, *, assets: dict | None = None,
                 size: int = 88) -> Image.Image:
    state = state or level.initial
    assets = assets or {}
    pad = 16
    canvas = Image.new("RGBA", (level.cols * size + 2 * pad, level.rows * size + 2 * pad), "#13242f")
    draw = ImageDraw.Draw(canvas)

    def paste(key: str, x: int, y: int) -> bool:
        if key not in assets:
            return False
        sprite = assets[key]
        if sprite.size != (size, size):
            sprite = sprite.resize((size, size), Image.Resampling.LANCZOS)
        canvas.alpha_composite(sprite, (x, y))
        return True

    for row in range(level.rows):
        for col in range(level.cols):
            cell = row, col
            x, y = pad + col * size, pad + row * size
            box = (x + 2, y + 2, x + size - 3, y + size - 3)
            if not paste("floor", x, y):
                draw.rounded_rectangle(box, radius=6, fill="#213c48", outline="#2a4853", width=1)
            if cell in level.walls:
                if not paste("wall", x, y):
                    draw.rounded_rectangle(box, radius=9, fill="#597484", outline="#8498a4", width=2)
                    draw.line((x + 10, y + size - 14, x + size - 10, y + size - 14), fill="#3e596b", width=5)
                continue
            if cell in level.goals:
                if not paste("goal", x, y):
                    draw.ellipse((x + size * .22, y + size * .22, x + size * .78, y + size * .78),
                                 outline="#ffd183", width=max(3, size // 15))
                    draw.ellipse((x + size * .44, y + size * .44, x + size * .56, y + size * .56), fill="#ffd183")
            if cell in state.boxes:
                if not paste("box", x, y):
                    margin = int(size * .16)
                    crate = (x + margin, y + margin, x + size - margin, y + size - margin)
                    draw.rounded_rectangle(crate, radius=9, fill="#f7a94b", outline="#ffe0a7", width=3)
                    draw.line((crate[0] + 8, crate[1] + 8, crate[2] - 8, crate[3] - 8), fill="#9d642f", width=5)
                    draw.line((crate[0] + 8, crate[3] - 8, crate[2] - 8, crate[1] + 8), fill="#9d642f", width=5)
                if cell in level.goals:
                    # Goal occupancy stays visible even if an uploaded crate is opaque.
                    draw.rounded_rectangle((x + 5, y + 5, x + size - 6, y + size - 6),
                                           radius=10, outline="#74efd0", width=4)
                    draw.ellipse((x + size - 24, y + 6, x + size - 7, y + 23), fill="#74efd0")
                    draw.line((x + size - 20, y + 14, x + size - 16, y + 18, x + size - 10, y + 11),
                              fill="#153a35", width=2)
            if cell == state.player:
                if not paste("player", x, y):
                    draw.rounded_rectangle((x + size * .19, y + size * .12, x + size * .81, y + size * .86),
                                           radius=15, fill="#edf3ef", outline="#a6ddd0", width=2)
                    draw.rounded_rectangle((x + size * .25, y + size * .35, x + size * .75, y + size * .63),
                                           radius=8, fill="#182e3c")
                    for eye in (.38, .62):
                        draw.ellipse((x + size * eye - 4, y + size * .48 - 4,
                                      x + size * eye + 4, y + size * .48 + 4), fill="#82efcd")
                if cell in level.goals:
                    draw.rounded_rectangle((x + 5, y + 5, x + size - 6, y + size - 6),
                                           radius=10, outline="#ffd183", width=3)
    return canvas.convert("RGB")


def replay_gif(level: Level, result: SearchResult, assets: dict | None = None) -> bytes:
    frames = [render_board(level, state, assets=assets, size=56) for state in replay(level, result)]
    stream = BytesIO()
    durations = [220] * len(frames)
    durations[-1] = 1400
    frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2)
    return stream.getvalue()

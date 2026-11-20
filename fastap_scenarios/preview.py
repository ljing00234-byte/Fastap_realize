"""Render all three worlds immediately after seeded reset as a standalone SVG."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

from . import make_world


SIZE = 300
INK = "#27384a"
TEAM = "#368fce"
APPLE = "#e06955"
PREY = "#399765"
OBSTACLE = "#b6bfca"
LANDMARK = "#735e85"


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def label(parent, x, y, content, size=16, **attrs):
    element(parent, "text", x=x, y=y, font_size=size, **attrs).text = str(content)


def marker(parent, x, y, r, color, title):
    dot = element(parent, "circle", cx=x, cy=y, r=r, fill=color, stroke="white", stroke_width=1.3)
    element(dot, "title").text = title


def cross(parent, x, y, r, title):
    path = f"M {x-r} {y-r} l {2*r} {2*r} M {x+r} {y-r} l {-2*r} {2*r}"
    element(parent, "path", d=path, stroke="white", stroke_width=5)
    shape = element(parent, "path", d=path, stroke=LANDMARK, stroke_width=2.5, stroke_linecap="round")
    element(shape, "title").text = title


def draw_lbf(panel, world):
    rows, cols = world.field.shape
    cell = SIZE / cols
    for index in range(1, cols):
        element(panel, "line", x1=index * cell, y1=0, x2=index * cell, y2=SIZE, stroke="#d1d9e0")
    for index in range(1, rows):
        element(panel, "line", x1=0, y1=index * cell, x2=SIZE, y2=index * cell, stroke="#d1d9e0")
    items = [(row, col, int(world.field[row, col]), APPLE, "食物") for row, col in
             zip(*world.field.nonzero())]
    for slot, player in enumerate(world.players):
        color, name = (INK, "受控角色") if slot < 2 else (TEAM, "队友")
        items.append((*player.position, player.level, color, name))
    for row, col, level, color, name in items:
        x, y = (col + 0.5) * cell, (row + 0.5) * cell
        marker(panel, x, y, 16, color, f"{name}（等级 {level}）")
        label(panel, x, y + 5, level, 15, fill="white", text_anchor="middle", font_weight="bold")


def draw_particles(panel, world, name):
    if name == "pp":
        players = world.players
        objects = [*players, world.prey, *world.obstacles]
    else:
        players = world.agents
        objects = [*players, *world.landmarks]
    # A square viewport with equal x/y scaling and space for all physical radii.
    extent = 0.05 + max(1.0, *(max(abs(body.position)) + body.radius for body in objects))
    scale = SIZE / (2 * extent)

    def coord(body):
        x, y = body.position
        return SIZE / 2 + x * scale, SIZE / 2 - y * scale

    if name == "pp":
        for obstacle in world.obstacles:
            marker(panel, *coord(obstacle), obstacle.radius * scale, OBSTACLE, "障碍物")
        marker(panel, *coord(world.prey), world.prey.radius * scale, PREY, "猎物")
    for i, body in enumerate(players):
        color, role = (INK, "受控角色") if i < 2 else (TEAM, "队友")
        marker(panel, *coord(body), body.radius * scale, color, role)
    if name == "cn":
        for i, target in enumerate(world.landmarks, 1):
            cross(panel, *coord(target), 6, f"地标 {i}")


def render(seed):
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="1032", height="450",
                     viewBox="0 0 1032 450", role="img")
    element(svg, "title").text = f"Fastap 三个场景的开局状态（随机种子 {seed}）"
    drawing = element(svg, "g", font_family="Inter, Noto Sans CJK SC, sans-serif", fill=INK)
    element(drawing, "rect", width=1032, height=450, fill="#ffffff")
    label(drawing, 28, 36, f"Fastap 场景　·　初始状态　·　随机种子 {seed}", 21, font_weight="bold")
    for column, (key, title, teammates) in enumerate((
        ("lbf", "LBF　协作采食", 2), ("pp", "PP　追捕", 1), ("cn", "CN　协作导航", 2)
    )):
        left = 28 + column * 338
        label(drawing, left, 79, title, 17, font_weight="bold")
        panel = element(drawing, "g", id=key, transform=f"translate({left} 92)")
        element(panel, "rect", width=SIZE, height=SIZE, fill="#f6f8fa", stroke="#97a6b3")
        world = make_world(key, seed=seed)
        try:
            world.reset(n_teammates=teammates)
            if key == "lbf":
                draw_lbf(panel, world)
            else:
                draw_particles(panel, world, key)
        finally:
            world.close()
    legend = [(INK, "受控角色"), (TEAM, "队友"), (APPLE, "食物"),
              (PREY, "猎物"), (OBSTACLE, "障碍物"), (LANDMARK, "地标")]
    for i, (color, word) in enumerate(legend):
        x = 36 + i * 117
        if color == LANDMARK:
            cross(drawing, x, 422, 5, word)
        else:
            marker(drawing, x, 422, 7, color, word)
        label(drawing, x + 15, 427, word, 14)
    label(drawing, 1004, 427, "LBF 圆内数字表示等级", 14, text_anchor="end")
    return ET.ElementTree(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="SVG output path (default: outputs/scenarios/seedN.svg)")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = (args.output or project / "outputs" / "scenarios" / f"seed{args.seed}.svg").resolve()
    if project not in output.parents or output.suffix.lower() != ".svg":
        parser.error("--output must be an SVG file inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    render(args.seed).write(output, encoding="utf-8", xml_declaration=True)
    print(output)


if __name__ == "__main__":
    main()

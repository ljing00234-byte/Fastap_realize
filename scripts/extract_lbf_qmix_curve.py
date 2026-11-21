"""Read the QMIX centre line in Figure 3(a) of the final Fastap paper using Poppler.

Source PDF: https://proceedings.mlr.press/v216/zhang23a/zhang23a.pdf
"""

import argparse
from bisect import bisect_left
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree


PAGE = ["-f", "7", "-l", "7"]
LEGEND = ["Fastap", "ODITS", "LIAM", "QMIX", "PEARL", "ESCP"]
NUMBER = r"[-+\d.eE]+"


def extract_curve(pdf):
    text = subprocess.check_output(["pdftotext", *PAGE, "-raw", str(pdf), "-"], text=True)
    labels = "0.0M 0.4M 0.8M 1.2M 1.6M 2.0M Timesteps 0.00 0.15 0.30 0.45 0.60 0.75 Test Return Mean"
    if f"{labels} {' '.join(LEGEND)} (a) LBF (Stationary)" not in " ".join(text.split()):
        raise ValueError("Expected the axes and legend of LBF Figure 3(a) on page 7 of the PMLR PDF")

    svg = subprocess.check_output(["pdftocairo", "-svg", *PAGE, str(pdf), "-"])
    paths = []
    for element in ElementTree.fromstring(svg).iter("{http://www.w3.org/2000/svg}path"):
        attrs = dict(element.attrib)
        for declaration in attrs.get("style", "").split(";"):
            if ":" in declaration:
                key, value = declaration.split(":", 1)
                attrs[key.strip()] = value.strip()
        if attrs.get("fill") == "none" and "stroke" in attrs:
            paths.append(attrs)

    # Each subplot has its own transform. The first long polyline belongs to the top-left panel.
    transform = next((p.get("transform") for p in paths if p["d"].count("L") > 100), None)
    if transform is None:
        raise ValueError("Missing vector curves on page 7")
    curves, segments = {}, []
    for path in paths:
        if path.get("transform") != transform:
            continue
        points = [(float(x), float(y)) for x, y in re.findall(rf"[ML]\s+({NUMBER})\s+({NUMBER})", path["d"])]
        if len(points) > 100:
            curves[path["stroke"]] = points
        elif len(points) == 2:
            segments.append((path["stroke"], *points))

    # The six short coloured segments are legend swatches. PDF plot coordinates increase upward.
    swatches = sorted(((y, colour) for colour, (_, y), (_, y2) in segments
                       if colour in curves and y == y2), reverse=True)
    grid_x = sorted(x for colour, (x, _), (x2, _) in segments if colour not in curves and x == x2)
    grid_y = sorted(y for colour, (_, y), (_, y2) in segments if colour not in curves and y == y2)
    if len(curves) != 6 or len({c for _, c in swatches}) != 6 or len(swatches) != 6:
        raise ValueError("Expected a unique line and legend swatch for each of the six algorithms")
    if len(grid_x) != 6 or len(grid_y) != 6:
        raise ValueError("Expected six labelled gridlines on each axis")
    points = curves[swatches[LEGEND.index("QMIX")][1]]
    xs = [x for x, _ in points]
    if xs[0] != grid_x[0] or xs[-1] < grid_x[-1] or any(a >= b for a, b in zip(xs, xs[1:])):
        raise ValueError("The QMIX line must cover the 0–2M range in increasing order")

    # The PDF path continues beyond the last labelled tick. Interpolate exactly at 2M.
    stop = bisect_left(xs, grid_x[-1])
    x1, y1 = points[stop - 1]
    x2, y2 = points[stop]
    last_y = y1 + (y2 - y1) * (grid_x[-1] - x1) / (x2 - x1)
    return [
        {"steps": round(2_000_000 * (x - grid_x[0]) / (grid_x[-1] - grid_x[0])),
         "mean_return": round(0.75 * (y - grid_y[0]) / (grid_y[-1] - grid_y[0]), 8)}
        for x, y in [*points[:stop], (grid_x[-1], last_y)]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Final proceedings PDF from PMLR volume 216")
    parser.add_argument("output", type=Path, help="Output JSON path")
    args = parser.parse_args()
    if args.pdf.resolve() == args.output.resolve():
        parser.error("Choose an output path different from the PDF")
    try:
        curve = extract_curve(args.pdf)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")

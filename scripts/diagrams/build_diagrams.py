#!/usr/bin/env python3
"""Generates the README diagrams as Excalidraw files.

The .excalidraw files are the source of truth (openable in
excalidraw.com); export.sh renders them to PNGs for the README. Layout
is declared as boxes/arrows below — tweak coordinates, regenerate,
re-export.

Run: python3 scripts/diagrams/build_diagrams.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "assets" / "diagrams"

# palette: harness = blue, go service = green, artifacts = yellow,
# external/ops = grey, judge = purple
BLUE = "#a5d8ff"
GREEN = "#d3f9d8"
YELLOW = "#ffec99"
GREY = "#dee2e6"
PURPLE = "#e5dbff"
INK = "#1e1e1e"

_seed = [0]


def _nid(prefix: str) -> str:
    _seed[0] += 1
    return f"{prefix}{_seed[0]}"


def _base(eid: str, kind: str) -> dict:
    return {
        "id": eid,
        "type": kind,
        "angle": 0,
        "strokeColor": INK,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "seed": _seed[0],
        "version": 1,
        "versionNonce": _seed[0],
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def box(elements: list[dict], x: float, y: float, w: float, h: float, fill: str, text: str, *, fontSize: int = 20) -> str:
    eid = _nid("box")
    el = _base(eid, "rectangle")
    el.update({
        "x": x, "y": y, "width": w, "height": h,
        "backgroundColor": fill,
        "roundness": {"type": 3},
    })
    elements.append(el)
    label(elements, x, y, w, h, text, fontSize=fontSize)
    return eid


def label(elements: list[dict], x: float, y: float, w: float, h: float, text: str, *, fontSize: int = 20, color: str = INK) -> None:
    """Free-floating centered text inside a box of given bounds. Line
    height and width are estimated (Helvetica ~0.52em per char)."""
    lines = text.split("\n")
    line_h = fontSize * 1.25
    text_w = max(len(l) for l in lines) * fontSize * 0.52
    el = _base(_nid("txt"), "text")
    el.update({
        "x": x + (w - text_w) / 2,
        "y": y + (h - line_h * len(lines)) / 2,
        "width": text_w,
        "height": line_h * len(lines),
        "text": text,
        "fontSize": fontSize,
        "fontFamily": 1,
        "textAlign": "center",
        "verticalAlign": "middle",
        "containerId": None,
        "originalText": text,
        "lineHeight": 1.25,
        "strokeColor": color,
    })
    elements.append(el)


def arrow(elements: list[dict], x1: float, y1: float, x2: float, y2: float, text: str = "", *, color: str = INK, label_dy: float = 0, bend: float | None = None) -> None:
    """Straight arrow, or an L-shaped one when bend is set: the shaft
    runs horizontally to (x1+bend, y1), then vertically to the end."""
    if bend is None:
        pts = [[0, 0], [x2 - x1, y2 - y1]]
    else:
        pts = [[0, 0], [bend, 0], [bend, y2 - y1]]
    el = _base(_nid("arr"), "arrow")
    el.update({
        "x": x1, "y": y1,
        "width": max(abs(x2 - x1), abs(bend or 0)),
        "height": max(abs(y2 - y1), abs(bend or 0)),
        "strokeColor": color,
        "points": pts,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "roundness": {"type": 2},
    })
    elements.append(el)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + label_dy
        tw = len(text) * 14 * 0.52
        t = _base(_nid("albl"), "text")
        t.update({
            "x": mx - tw / 2,
            "y": my - 14,
            "width": tw,
            "height": 20,
            "text": text,
            "fontSize": 14,
            "fontFamily": 1,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": None,
            "originalText": text,
            "lineHeight": 1.25,
        })
        elements.append(t)


def save(name: str, elements: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    (OUT / f"{name}.excalidraw").write_text(json.dumps(doc, indent=1))


def architecture() -> None:
    e: list[dict] = []
    a1 = box(e, 40, 60, 380, 90, BLUE, "Your agents\nPython harness: workers, tools,\nsupervisor")
    a2 = box(e, 40, 200, 380, 80, BLUE, "OpenTelemetry SDK\nrecords every model call and\ntool call as a span")
    arrow(e, 230, 150, 230, 200)
    f = box(e, 40, 330, 380, 70, YELLOW, "Fault injection (experiments)\nseeded: malformed JSON, policy\nviolations, loops, budget blowouts")
    arrow(e, 230, 330, 230, 280, "faults corrupt provider replies")
    c = box(e, 40, 450, 380, 70, GREY, "watchtower verify (CLI)\ncheck a saved trace file, offline")
    arrow(e, 420, 485, 600, 185, bend=180)
    label(e, 440, 470, 160, 20, "saved spans (JSON)", fontSize=14)

    g = box(e, 540, 40, 680, 330, GREEN, "Watchtower — the Go service")
    g1 = box(e, 580, 130, 130, 60, GREEN, "ingest")
    g2 = box(e, 740, 130, 150, 60, GREEN, "reconstruct")
    g3 = box(e, 920, 130, 130, 60, GREEN, "verify")
    g4 = box(e, 1080, 130, 110, 60, GREEN, "report")
    arrow(e, 710, 160, 740, 160)
    arrow(e, 890, 160, 920, 160)
    arrow(e, 1050, 160, 1080, 160)

    arrow(e, 420, 245, 580, 165, "OTLP/HTTP")
    v = box(e, 1280, 150, 370, 100, YELLOW, "Verdict report\nPASS / FLAGGED / FAIL\n+ evidence: the spans behind it")
    arrow(e, 1190, 180, 1280, 180)
    r = box(e, 1280, 300, 370, 70, GREY, "You / CI / dashboards\nread the report, act on it")
    arrow(e, 1465, 300, 1465, 250, "GET /v1/reports/{traceId}", label_dy=-10)
    j = box(e, 700, 420, 330, 70, PURPLE, "LLM judge (optional)\nGemini or DeepSeek via Bedrock")
    arrow(e, 865, 420, 985, 190, "second opinion", label_dy=-14)
    save("architecture", e)


def pipeline() -> None:
    e: list[dict] = []
    s = box(e, 40, 200, 190, 70, GREY, "Spans\n(OTLP)")
    r = box(e, 290, 200, 230, 70, GREEN, "Reconstruct the run\nstep order, agents,\nbudget summary")
    arrow(e, 210, 235, 290, 235, "one trace")
    vs = ["schema", "policy", "loop", "budget", "status"]
    vids = []
    for i, vname in enumerate(vs):
        vids.append(box(e, 580, 40 + i * 75, 190, 60, BLUE, f"{vname}\ncheck"))
    for i in range(5):
        arrow(e, 520, 235, 580, 70 + i * 75)
    f = box(e, 830, 160, 240, 70, YELLOW, "Findings with evidence\nspan ids, values, timestamps")
    for i in range(5):
        arrow(e, 770, 70 + i * 75, 830, 190)
    j = box(e, 830, 420, 240, 70, PURPLE, "LLM judge\n(optional)")
    arrow(e, 950, 420, 950, 230, "second opinion")
    vd = box(e, 1130, 160, 240, 70, YELLOW, "Verdict\nPASS · FLAGGED · FAIL")
    arrow(e, 1070, 195, 1130, 195)
    rep = box(e, 1130, 290, 240, 70, GREY, "Report\nverdict + evidence bundle")
    arrow(e, 1250, 230, 1250, 290)
    save("pipeline", e)


def experiment() -> None:
    e: list[dict] = []
    g = box(e, 40, 60, 280, 90, GREY, "Experiment grid\n6 faults × 5 seeds × 2 models\n× 3 runs + clean controls\n= 210 cells")
    run = box(e, 380, 60, 260, 90, BLUE, "Seeded runs\nFakeProvider (offline, free)\nor real Gemini")
    wt = box(e, 700, 60, 250, 90, GREEN, "Watchtower\nverifies every run,\njudge on a capped sample")
    art = box(e, 1010, 60, 250, 90, YELLOW, "Artifacts (JSON)\nchecksummed, versioned,\nprotocol-pinned")
    an = box(e, 1320, 60, 290, 90, PURPLE, "Analysis\n100% detection, 0% false\npositives, judge agreement\n(kappa matrix)")
    arrow(e, 320, 105, 380, 105, "per cell")
    arrow(e, 640, 105, 700, 105, "OTLP")
    arrow(e, 950, 105, 1010, 105, "verdicts")
    arrow(e, 1260, 105, 1320, 105, "tables")
    note = box(e, 700, 220, 560, 60, GREY, "judge vs judge: Gemini · DeepSeek · Kimi — pairwise kappa + consensus\noffline vs live: identical results (delta reports)")
    save("experiment", e)


def main() -> None:
    architecture()
    pipeline()
    experiment()
    print(f"wrote {len(list(OUT.glob('*.excalidraw')))} diagrams -> {OUT}")


if __name__ == "__main__":
    main()

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

# Keep the source palette aligned with the README exports: green is the
# harness/input side, blue is the Go pipeline, amber is the optional judge,
# and purple is evidence, reports, and analysis.
SOURCE = "#dcfce7"
WIRE = "#e0f2fe"
CORE = "#dbeafe"
JUDGE = "#fef3c7"
RESULT = "#f3e8ff"
FAULT = "#fee2e2"
SETUP = "#fce7f3"
PASS = "#dcfce7"
FAIL = "#fee2e2"
GAP = "#ffedd5"
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
    h = box(e, 40, 80, 230, 80, SOURCE, "Python harness\nworkers, supervisor,\nseeded faults")
    x = box(e, 40, 230, 230, 70, SOURCE, "CLI / saved JSON trace")
    o = box(e, 350, 80, 250, 80, WIRE, "OTel GenAI spans\nmodel and tool calls")
    i = box(e, 700, 135, 240, 75, CORE, "Go ingest\nOTLP/HTTP and JSON")
    g = box(e, 1030, 135, 270, 75, CORE, "Reconstruct run\ntree, steps, budget")
    v = box(e, 1390, 65, 280, 95, CORE, "Deterministic checks\nschema, policy, loop,\nbudget, status")
    j = box(e, 1390, 245, 280, 75, JUDGE, "Optional LLM judge\nGemini / Bedrock")
    evi = box(e, 1780, 135, 280, 75, RESULT, "Evidence report\nspan IDs, timestamps, values")
    rep = box(e, 2170, 135, 260, 75, RESULT, "Report store\nPASS, FLAGGED, FAIL")
    q = box(e, 2530, 135, 250, 75, RESULT, "Operator and\nexperiments")

    arrow(e, 270, 120, 350, 120)
    arrow(e, 600, 120, 700, 165, "OTLP/HTTP protobuf", label_dy=-10)
    arrow(e, 270, 265, 700, 180, "legacy JSON", label_dy=18)
    arrow(e, 940, 172, 1030, 172)
    arrow(e, 1300, 172, 1390, 112)
    arrow(e, 1300, 172, 1390, 282)
    arrow(e, 1670, 112, 1780, 172)
    arrow(e, 1670, 282, 1780, 172)
    arrow(e, 2060, 172, 2170, 172)
    arrow(e, 2430, 172, 2530, 172)
    save("architecture", e)


def pipeline() -> None:
    e: list[dict] = []
    r = box(e, 40, 210, 270, 80, WIRE, "Reconstructed run\nordered steps and budget")
    checks = [
        ("Schema", "output contract"),
        ("Policy", "tool allowlist"),
        ("Loop", "repeated calls"),
        ("Budget", "steps, tokens, time"),
        ("Status", "span errors"),
        ("Evidence", "id and parent gaps"),
    ]
    for i, (name, detail) in enumerate(checks):
        y = 20 + i * 105
        box(e, 430, y, 280, 75, CORE, f"{name}: {detail}")
        arrow(e, 310, 250, 430, y + 38)
        arrow(e, 710, y + 38, 820, 250)
    j = box(e, 430, 650, 280, 75, JUDGE, "Optional judge\nsemantic review")
    arrow(e, 310, 250, 430, 688)
    arrow(e, 710, 688, 820, 250)
    f = box(e, 820, 205, 300, 90, RESULT, "Finding\nverifier, severity, evidence")
    m = box(e, 1230, 205, 300, 90, RESULT, "Severity mapping\nhighest finding wins")
    arrow(e, 1120, 250, 1230, 250)
    p = box(e, 1660, 50, 260, 70, PASS, "PASS: no findings")
    w = box(e, 1660, 175, 260, 70, JUDGE, "FLAGGED: warning")
    inc = box(e, 1660, 300, 260, 70, GAP, "INCONCLUSIVE: gap")
    fail = box(e, 1660, 425, 260, 70, FAIL, "FAIL: critical")
    arrow(e, 1530, 250, 1660, 85, "clean", label_dy=-8)
    arrow(e, 1530, 250, 1660, 210, "warning")
    arrow(e, 1530, 250, 1660, 335, "evidence gap")
    arrow(e, 1530, 250, 1660, 460, "critical", label_dy=8)
    save("pipeline", e)


def experiment() -> None:
    e: list[dict] = []
    g = box(e, 40, 70, 300, 90, SETUP, "Grid: 6 faults, 5 seeds,\n2 models, 3 runs, clean controls\n= 210 cells")
    f = box(e, 410, 70, 300, 90, FAULT, "Seeded faults: malformed JSON,\nschema, policy, loop, budget, timeout")
    a = box(e, 780, 70, 260, 90, SOURCE, "Agent run\nworkers and supervisor")
    o = box(e, 1110, 70, 270, 90, WIRE, "OTel trace\nproduction wire contract")
    w = box(e, 1450, 70, 280, 90, CORE, "Watchtower\nreconstruct and verify")
    d = box(e, 1800, 20, 280, 80, CORE, "Deterministic results\nfindings and verdict")
    j = box(e, 1800, 140, 280, 80, JUDGE, "Optional judge\nusage and model vote")
    m = box(e, 2150, 70, 330, 90, RESULT, "Analysis: detection, false positives,\noverhead, Cohen's kappa")
    v = box(e, 2550, 70, 300, 90, RESULT, "Artifacts: protocol,\nconfig, cell signatures")
    arrow(e, 340, 115, 410, 115)
    arrow(e, 710, 115, 780, 115)
    arrow(e, 1040, 115, 1110, 115)
    arrow(e, 1380, 115, 1450, 115)
    arrow(e, 1730, 115, 1800, 60)
    arrow(e, 1730, 115, 1800, 180)
    arrow(e, 2080, 60, 2150, 115)
    arrow(e, 2080, 180, 2150, 115)
    arrow(e, 2480, 115, 2550, 115)
    save("experiment", e)


def evidence_flow() -> None:
    """Flow for evidence-aware supervision experiments.

    Keep this separate from the original pipeline diagram: it shows the
    new research variable, evidence quality, without changing the
    existing architecture drawing until the verifier contract is stable.
    """
    e: list[dict] = []
    supervisor = box(e, 40, 100, 270, 90, SETUP, "Supervisor meta-agent\ndecides accept, retry,\nblock, or escalate")
    workers = box(e, 390, 100, 250, 90, SOURCE, "Worker agents\nproduce results")
    behavior = box(e, 720, 35, 285, 90, FAULT, "Behavior faults\nside effects, drift,\npolicy or goal gaming")
    telemetry = box(e, 720, 190, 285, 90, GAP, "Evidence faults\ndropped, duplicate,\nlate, or forged spans")
    trace = box(e, 1085, 100, 285, 105, WIRE, "OTLP trace\nbehavior + telemetry\narrive as evidence")
    reconstruct = box(e, 1450, 100, 310, 105, CORE, "Reconstruct run\nordered steps + evidence\nquality inventory")
    checks = box(e, 1840, 35, 300, 90, CORE, "Deterministic checks\nproperties + evidence\nobligations")
    judge = box(e, 1840, 190, 300, 90, JUDGE, "Optional LLM judge\nsemantic second opinion")
    report = box(e, 2220, 100, 300, 105, RESULT, "Evidence report\nfindings, spans,\nconfidence state")
    passed = box(e, 2600, 10, 230, 62, PASS, "PASS\nsufficient evidence")
    inconclusive = box(e, 2600, 92, 230, 78, GAP, "INCONCLUSIVE\nevidence gap")
    flagged = box(e, 2600, 190, 230, 62, JUDGE, "FLAGGED\nwarning")
    failed = box(e, 2600, 272, 230, 62, FAIL, "FAIL\nobserved violation")

    arrow(e, 310, 145, 390, 145, "delegates")
    arrow(e, 640, 145, 720, 80, "run")
    arrow(e, 640, 145, 720, 235, "telemetry")
    arrow(e, 1005, 80, 1085, 145, "behavior")
    arrow(e, 1005, 235, 1085, 165, "evidence")
    arrow(e, 1370, 152, 1450, 152)
    arrow(e, 1760, 152, 1840, 80)
    arrow(e, 1760, 152, 1840, 235)
    arrow(e, 2140, 80, 2220, 145)
    arrow(e, 2140, 235, 2220, 160)
    arrow(e, 2520, 150, 2600, 40, "clean")
    arrow(e, 2520, 150, 2600, 130, "gap")
    arrow(e, 2520, 150, 2600, 220, "warning")
    arrow(e, 2520, 150, 2600, 302, "critical")
    save("evidence_flow", e)


def main() -> None:
    architecture()
    pipeline()
    experiment()
    evidence_flow()
    print(f"wrote {len(list(OUT.glob('*.excalidraw')))} diagrams -> {OUT}")


if __name__ == "__main__":
    main()

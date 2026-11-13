"""Generate the revised FAIR-AL framework figure (rev1) — compact single-column-width design.

One geometry spec -> drawio (editable) + SVG -> PDF/PNG (rsvg-convert).

Legibility budget (manuscript-revision rule: effective font >= 8 pt at the inserted width):
    canvas 700 x ~430 px = 525 pt = 7.29 in wide, inserted at \\textwidth (6.5 in) -> scale 0.89
    body text 13 px = 9.75 pt drawn -> 8.7 pt effective.

Content is bound to src/baalexp/strategies.py:
  * candidate pool = top-m by predictive entropy (m = --k-candidates = 200)
  * f1 = -H, f2 = TV(Q_t + {x}, pi), Q_t = cumulative queried set
  * historical main implementation scores the whole batch against Q_t (batch-prior);
    the intra-batch sequential variant is a separate ablation
  * Pareto: non-dominated set, min-max inside it, nearest to the ideal point
  * Weighted: min-max inside the candidate set, lambda*f1 + (1-lambda)*f2, lambda = 0.5
  * Constrained: raw TV <= tau = 0.03, ranked by entropy; tau is dropped (and recorded)
    when no candidate is feasible

Usage: python scripts/33_make_framework_figure.py <outdir>
"""
from __future__ import annotations

import html
import os
import sys
import xml.etree.ElementTree as ET

W = 700
FONT = "Liberation Sans, DejaVu Sans, sans-serif"
INK, MUTED, LINE = "#1f2933", "#52606d", "#3e4c59"
FILL = {"c1": "#eef2f7", "c2": "#e8f1fb", "c3": "#eef7ee", "warn": "#fdf3e3", "none": "#ffffff"}

COLS = {"c1": (8, 190), "c2": (224, 238), "c3": (488, 204)}
GAP = 22
PAD = 8
TITLE_SIZE, HEAD_SIZE, BOX_HEAD, BODY = 16, 13, 12.5, 13.0


def esc_svg(text: str) -> str:
    """Subscript markup is '_{...}', superscript '^{...}' (braces alone stay literal)."""
    out, i = [], 0
    while i < len(text):
        if text[i] in "_^" and i + 2 < len(text) and text[i + 1] == "{":
            j = text.find("}", i + 2)
            if j > i:
                inner = html.escape(text[i + 2:j])
                shift = "sub" if text[i] == "_" else "super"
                out.append(f'<tspan baseline-shift="{shift}" font-size="78%">{inner}</tspan>')
                i = j + 1
                continue
        out.append(html.escape(text[i]))
        i += 1
    return "".join(out)


def esc_html(text: str) -> str:
    out, i = [], 0
    while i < len(text):
        if text[i] in "_^" and i + 2 < len(text) and text[i + 1] == "{":
            j = text.find("}", i + 2)
            if j > i:
                inner = html.escape(text[i + 2:j])
                tag = "sub" if text[i] == "_" else "sup"
                out.append(f"<{tag}>{inner}</{tag}>")
                i = j + 1
                continue
        out.append(html.escape(text[i]))
        i += 1
    return "".join(out)


class Fig:
    def __init__(self):
        self.boxes: list[dict] = []
        self.edges: list[dict] = []
        self.y = {k: 118 for k in COLS}

    def wrap(self, text: str, size: float, width: int) -> list[str]:
        max_chars = max(int(width / (0.50 * size)), 8)
        words, lines, cur = text.split(), [], ""
        for wd in words:
            cand = (cur + " " + wd).strip()
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return lines

    def box(self, bid, col, head, lines=(), fill="c1", dashed=False, gap=GAP,
            head_size=BOX_HEAD, size=BODY, x=None, w=None):
        cx, cw = COLS.get(col, (COLS["c1"][0], W - 2 * COLS["c1"][0]))
        x = cx if x is None else x
        w = cw if w is None else w
        wrapped: list[str] = []
        for ln in lines:
            wrapped.extend(self.wrap(ln, size, w - 2 * PAD - 2))
        h = PAD + (head_size + 4 if head else 0) + len(wrapped) * (size + 3.5) + PAD - 2
        y = self.y.get(col, 118)
        self.boxes.append(dict(id=bid, x=x, y=y, w=w, h=h, head=head, lines=wrapped,
                               fill=FILL[fill], dashed=dashed, size=size,
                               head_size=head_size, col=col))
        if col in self.y:
            self.y[col] = y + h + gap
        return bid

    def header(self, bid, col, text, fill):
        x, w = COLS[col]
        self.boxes.append(dict(id=bid, x=x, y=78, w=w, h=26, head=text, lines=[],
                               fill=FILL[fill], dashed=False, size=BODY,
                               head_size=HEAD_SIZE, col=col))

    def edge(self, src, dst, label="", dashed=False, route="v"):
        self.edges.append(dict(src=src, dst=dst, label=label, dashed=dashed, route=route))


def build() -> Fig:
    f = Fig()
    f.boxes.append(dict(id="title", x=8, y=14, w=W - 16, h=34,
                        head="FAIR-AL: auditing and controlling demographic query drift",
                        lines=[], fill=FILL["none"], dashed=False, size=BODY,
                        head_size=TITLE_SIZE, col="title"))
    f.header("h1", "c1", "1 · Learning state", "c1")
    f.header("h2", "c2", "2 · Scoring and selection", "c2")
    f.header("h3", "c3", "3 · Update and audit", "c3")

    f.box("U", "c1", "Unlabeled pool U_{t}", ["no labels available"])
    f.box("C", "c1", "Candidate pool C_{t}",
          ["top-m by entropy,", "m = 200"])
    f.box("L", "c1", "Labeled set L_{t}", ["labels known"])
    f.box("H", "c1", "Classifier h_{t}", ["fit on L_{t} only"])
    f.box("Q", "c1", "Queried set Q_{t}", ["grows only after", "a batch is annotated"])
    f.box("PI", "c1", "Target π", ["protocol choice"])

    f.box("F", "c2", "Objectives (lower is better)",
          ["inputs: h_{t}, C_{t}, Q_{t}, π",
           "f_{1}(x) = −H(p(y|x; h_{t}))",
           "f_{2}(x) = TV(Q_{t} ∪ {x}, π)"])
    f.box("BP", "c2", "Batch-prior scoring",
          ["historical main arm:", "f_{2} uses Q_{t} fixed per batch"])
    f.box("PA", "c2", "Pareto",
          ["non-dominated set P_{t};", "min–max inside P_{t};", "nearest the ideal point"],
          fill="c2", gap=6)
    f.box("WE", "c2", "Weighted",
          ["min–max inside C_{t};", "λ·f_{1} + (1 − λ)·f_{2}, λ = 0.5"],
          fill="c2", gap=6)
    f.box("CO", "c2", "Constrained",
          ["raw TV ≤ τ, τ = 0.03;", "rank feasible by entropy;",
           "empty → τ dropped,", "event recorded"], fill="c2")

    f.box("B", "c3", "Batch B_{t} (b = 10)", [])
    f.box("AN", "c3", "Annotation", ["labels for B_{t}"])
    f.box("UP", "c3", "Update",
          ["L_{t+1} = L_{t} ∪ B_{t}",
           "U_{t+1} = U_{t} \\ B_{t}",
           "Q_{t+1} = Q_{t} ∪ B_{t}",
           "retrain h_{t+1} on L_{t+1}"])
    f.box("QB", "c3", "Audit",
          ["Q-Bias = TV(Q_{t}, π)", "reported separately"])
    f.box("IB", "c3", "Intra-batch update", ["ablation only; not the", "historical arm"],
          fill="warn", dashed=True)

    f.edge("U", "C")
    f.edge("L", "H")
    f.edge("H", "F", route="h_stub")
    f.edge("C", "F", route="h_stub")
    f.edge("Q", "F", route="h_up")
    f.edge("PI", "F", dashed=True, route="h_up")
    f.edge("F", "BP")
    f.edge("BP", "PA")
    f.edge("PA", "B", route="right_spine")
    f.edge("WE", "B", route="right_spine")
    f.edge("CO", "B", route="right_spine")
    f.edge("B", "AN", "query")
    f.edge("AN", "UP")
    f.edge("UP", "QB")
    return f


def svg(f: Fig) -> tuple[str, int]:
    height = max(b["y"] + b["h"] for b in f.boxes) + 56
    f.boxes.append(dict(id="note", x=8, y=height - 40, w=W - 16, h=32,
                        head="Query composition is audited separately from predictive "
                             "performance and group error disparities.",
                        lines=[], fill=FILL["c1"], dashed=False, size=BODY,
                        head_size=BODY, col="note"))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
             f'viewBox="0 0 {W} {height}">',
             f'<rect width="{W}" height="{height}" fill="#ffffff"/>',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
             'markerHeight="6" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker></defs>']
    for b in f.boxes:
        dash = ' stroke-dasharray="6 4"' if b["dashed"] else ""
        parts.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" '
                     f'rx="5" fill="{b["fill"]}" stroke="{LINE}" stroke-width="1.4"{dash}/>')
        ty = b["y"] + PAD + b["head_size"] - 1
        if b["head"]:
            parts.append(f'<text x="{b["x"] + PAD}" y="{ty}" font-family="{FONT}" '
                         f'font-size="{b["head_size"]}" font-weight="bold" fill="{INK}">'
                         f'{esc_svg(b["head"])}</text>')
            ty += b["size"] + 1
        for ln in b["lines"]:
            parts.append(f'<text x="{b["x"] + PAD}" y="{ty}" font-family="{FONT}" '
                         f'font-size="{b["size"]}" fill="{INK}">{esc_svg(ln)}</text>')
            ty += b["size"] + 3.5
    B = {b["id"]: b for b in f.boxes}

    def seg(x1, y1, x2, y2, dashed):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        return (f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{LINE}" '
                f'stroke-width="1.4"{dash} marker-end="url(#arr)"/>')

    def poly(points, dashed):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        d = " L ".join(f"{x} {y}" for x, y in points)
        return (f'<path d="M {d}" fill="none" stroke="{LINE}" stroke-width="1.4"{dash} '
                f'marker-end="url(#arr)"/>')

    for e in f.edges:
        s, d = B[e["src"]], B[e["dst"]]
        r = e["route"]
        if r == "v":
            x = s["x"] + 0.5 * s["w"]
            parts.append(seg(x, s["y"] + s["h"], x, d["y"], e["dashed"]))
            if e["label"]:
                parts.append(f'<text x="{x + 5}" y="{s["y"] + s["h"] + 13}" '
                             f'font-family="{FONT}" font-size="10.5" font-style="italic" '
                             f'fill="{MUTED}">{esc_svg(e["label"])}</text>')
        elif r in {"h_stub", "h_up"}:
            x1 = COLS["c1"][0] + COLS["c1"][1]
            y1 = s["y"] + 0.5 * s["h"]
            frac = {"H": 0.36, "C": 0.52, "Q": 0.74, "PI": 0.90}[e["src"]]
            y2 = d["y"] + frac * d["h"]
            gx = x1 + {"H": 6, "C": 11, "Q": 16, "PI": 21}[e["src"]]
            parts.append(poly([(x1, y1), (gx, y1), (gx, y2), (d["x"], y2)], e["dashed"]))
            if e["label"]:
                parts.append(f'<text x="{x1 + 2}" y="{min(y1, y2) - 4}" font-family="{FONT}" '
                             f'font-size="10.5" font-style="italic" fill="{MUTED}">'
                             f'{esc_svg(e["label"])}</text>')
        elif r == "right_spine":
            x1 = COLS["c2"][0] + COLS["c2"][1]
            y1 = s["y"] + 0.5 * s["h"]
            gx = COLS["c3"][0] - 10
            y2 = d["y"] + 0.5 * d["h"]
            parts.append(poly([(x1, y1), (gx, y1), (gx, y2), (d["x"], y2)], e["dashed"]))
            if e["label"]:
                parts.append(f'<text x="{x1 + 3}" y="{y1 - 4}" font-family="{FONT}" '
                             f'font-size="10.5" font-style="italic" fill="{MUTED}">'
                             f'{esc_svg(e["label"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts), height


def drawio(f: Fig, height: int) -> str:
    cells = []
    for b in f.boxes:
        dash = "dashed=1;" if b["dashed"] else ""
        style = (f"rounded=1;whiteSpace=wrap;html=1;arcSize=6;fillColor={b['fill']};"
                 f"strokeColor={LINE};strokeWidth=1.4;{dash}fontFamily=Helvetica;"
                 f"fontSize={b['size']:.0f};align=left;verticalAlign=top;spacingLeft={PAD};"
                 f"spacingTop=4;")
        label = f"<b>{esc_html(b['head'])}</b>"
        if b["lines"]:
            label += "<br>" + "<br>".join(esc_html(x) for x in b["lines"])
        cells.append(
            f'<mxCell id="{b["id"]}" value="{html.escape(label, quote=True)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" '
            f'height="{b["h"]}" as="geometry"/></mxCell>')
    for i, e in enumerate(f.edges):
        dash = "dashed=1;" if e["dashed"] else ""
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;"
                 f"strokeColor={LINE};strokeWidth=1.4;{dash}fontSize=10;fontStyle=2;"
                 f"labelBackgroundColor=#ffffff;")
        cells.append(
            f'<mxCell id="e{i}" value="{html.escape(e["label"], quote=True)}" style="{style}" '
            f'edge="1" parent="1" source="{e["src"]}" target="{e["dst"]}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    return (
        '<mxfile host="hermes" modified="2026-11-03T00:00:00.000Z" agent="hermes" '
        'version="24.0.0"><diagram id="fairal-rev1" name="FAIR-AL rev1">'
        f'<mxGraphModel dx="900" dy="700" grid="0" gridSize="10" guides="1" tooltips="1" '
        f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" '
        f'pageHeight="{height}" math="0" shadow="0"><root><mxCell id="0"/>'
        '<mxCell id="1" parent="0"/>' + "".join(cells) +
        '</root></mxGraphModel></diagram></mxfile>')


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    fig = build()
    svg_text, height = svg(fig)
    with open(os.path.join(outdir, "fairal_framework_rev1.svg"), "w", encoding="utf-8") as fh:
        fh.write(svg_text)
    dx = drawio(fig, height)
    ET.fromstring(dx)
    with open(os.path.join(outdir, "fairal_framework_rev1.drawio"), "w", encoding="utf-8") as fh:
        fh.write(dx)
    print(f"wrote {outdir}: {len(fig.boxes)} boxes, {len(fig.edges)} edges, canvas {W}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

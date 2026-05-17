"""
Generate DDFed-Markov diagram images as SVG and HTML files.

NO external dependencies - uses only Python standard library.
SVG/HTML files can be opened in any browser; right-click to save as PNG.

Creates:
1. Markov Chain State Diagram (3-state)
2. DFD Level 0: Context Diagram
3. DFD Level 1: Process Flow
4. Full DDFed-Markov Flowchart
"""
import math
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "figures" / "markov_diagrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _arrow_svg(x1, y1, x2, y2, color="#000"):
    """Return SVG path for arrow from (x1,y1) to (x2,y2)."""
    angle = math.atan2(y2 - y1, x2 - x1)
    al = 12
    x3 = x2 - al * math.cos(angle - 0.4)
    y3 = y2 - al * math.sin(angle - 0.4)
    x4 = x2 - al * math.cos(angle + 0.4)
    y4 = y2 - al * math.sin(angle + 0.4)
    line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"/>'
    head = f'<polygon points="{x2},{y2} {x3},{y3} {x4},{y4}" fill="{color}" stroke="{color}"/>'
    return line + head


def _svg_header(w, h, title=""):
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{title}</title>
<rect width="100%" height="100%" fill="white"/>
'''


def draw_markov_state_diagram():
    """Draw the 3-state Markov chain diagram as SVG."""
    w, h = 700, 650
    cx, cy = w // 2, h // 2
    r = 180
    angles = [90, 210, 330]
    positions = {}
    for i, name in enumerate(["LOW", "MEDIUM", "HIGH"]):
        th = math.radians(angles[i])
        positions[name] = (cx + r * math.cos(th), cy - r * math.sin(th))

    node_info = {"LOW": ("0.01", "#90EE90"), "MEDIUM": ("0.05", "#FFD700"), "HIGH": ("0.10", "#FF6B6B")}
    transitions = [
        ("LOW", "LOW", 0.6), ("LOW", "MEDIUM", 0.3), ("LOW", "HIGH", 0.1),
        ("MEDIUM", "LOW", 0.2), ("MEDIUM", "MEDIUM", 0.6), ("MEDIUM", "HIGH", 0.2),
        ("HIGH", "LOW", 0.1), ("HIGH", "MEDIUM", 0.3), ("HIGH", "HIGH", 0.6),
    ]

    parts = [_svg_header(w, h, "Markov State Diagram")]

    # Edges (non-self)
    for src, dst, prob in transitions:
        if src == dst:
            continue
        p1, p2 = positions[src], positions[dst]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(dx * dx + dy * dy)
        start = (p1[0] + 45 * dx / dist, p1[1] + 45 * dy / dist)
        end = (p2[0] - 45 * dx / dist, p2[1] - 45 * dy / dist)
        parts.append(_arrow_svg(start[0], start[1], end[0], end[1]))
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        parts.append(f'<ellipse cx="{mid[0]}" cy="{mid[1]}" rx="18" ry="12" fill="white" stroke="#666"/>')
        parts.append(f'<text x="{mid[0]}" y="{mid[1]+4}" text-anchor="middle" font-size="11">{prob}</text>')

    # Self-loops
    for src, dst, prob in transitions:
        if src != dst:
            continue
        idx = list(positions.keys()).index(src)
        p = positions[src]
        angle = math.radians(angles[idx])
        loop_r = 55
        pts = []
        for i in range(25):
            t = angle - 0.9 + 1.8 * i / 24
            pts.append((p[0] + loop_r * math.cos(t), p[1] - loop_r * math.sin(t)))
        path_d = f"M {pts[0][0]} {pts[0][1]}"
        for pt in pts[1:]:
            path_d += f" L {pt[0]} {pt[1]}"
        parts.append(f'<path d="{path_d}" fill="none" stroke="black" stroke-width="2"/>')
        parts.append(_arrow_svg(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1]))
        lbl = (p[0] + (loop_r + 25) * math.cos(angle), p[1] - (loop_r + 25) * math.sin(angle))
        parts.append(f'<ellipse cx="{lbl[0]}" cy="{lbl[1]}" rx="14" ry="10" fill="white" stroke="#666"/>')
        parts.append(f'<text x="{lbl[0]}" y="{lbl[1]+4}" text-anchor="middle" font-size="11">{prob}</text>')

    # Nodes
    for name, pos in positions.items():
        sigma, color = node_info[name]
        x, y = pos[0] - 50, pos[1] - 40
        parts.append(f'<ellipse cx="{pos[0]}" cy="{pos[1]}" rx="50" ry="40" fill="{color}" stroke="black" stroke-width="2"/>')
        parts.append(f'<text x="{pos[0]}" y="{pos[1]-10}" text-anchor="middle" font-size="14" font-weight="bold">{name}</text>')
        parts.append(f'<text x="{pos[0]}" y="{pos[1]+18}" text-anchor="middle" font-size="11">&#963;={sigma}</text>')

    parts.append(f'<text x="{w//2}" y="35" text-anchor="middle" font-size="18">Markov Chain State Diagram (DDFed-Markov)</text>')
    parts.append(f'<text x="{w//2}" y="55" text-anchor="middle" font-size="12" fill="#666">3-State Noise Intensity</text>')
    parts.append("</svg>")

    out = OUTPUT_DIR / "markov_state_diagram.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {out}")


def draw_dfd_context():
    """Draw DFD Level 0: Context diagram as SVG."""
    w, h = 700, 600
    parts = [_svg_header(w, h, "DFD Context")]

    # System box
    parts.append('<rect x="220" y="240" width="260" height="80" rx="10" fill="#87CEEB" stroke="black" stroke-width="2"/>')
    parts.append('<text x="350" y="275" text-anchor="middle" font-size="16" font-weight="bold">DDFED-MARKOV</text>')
    parts.append('<text x="350" y="300" text-anchor="middle" font-size="12">SYSTEM</text>')

    # Clients
    parts.append('<rect x="40" y="380" width="160" height="100" rx="8" fill="#E8F5E9" stroke="black"/>')
    parts.append('<text x="120" y="415" text-anchor="middle" font-size="12">Client 1</text>')
    parts.append('<text x="120" y="440" text-anchor="middle" font-size="10">(Data D1)</text>')
    parts.append('<rect x="500" y="380" width="160" height="100" rx="8" fill="#E8F5E9" stroke="black"/>')
    parts.append('<text x="580" y="415" text-anchor="middle" font-size="12">Client n</text>')
    parts.append('<text x="580" y="440" text-anchor="middle" font-size="10">(Data Dn)</text>')

    # Server
    parts.append('<rect x="240" y="80" width="220" height="80" rx="8" fill="#FFF3E0" stroke="black"/>')
    parts.append('<text x="350" y="115" text-anchor="middle" font-size="12">Server</text>')
    parts.append('<text x="350" y="138" text-anchor="middle" font-size="10">(Aggregation)</text>')

    # Arrows
    parts.append(_arrow_svg(200, 430, 280, 280))
    parts.append('<text x="240" y="360" font-size="10" fill="#666">Updates + votes</text>')
    parts.append(_arrow_svg(500, 430, 420, 280))
    parts.append('<text x="455" y="360" font-size="10" fill="#666">Updates + votes</text>')
    parts.append(_arrow_svg(350, 240, 350, 160))
    parts.append('<text x="365" y="210" font-size="10" fill="#666">W_G</text>')
    parts.append(_arrow_svg(120, 380, 200, 200))
    parts.append(_arrow_svg(580, 380, 500, 200))
    parts.append('<text x="350" y="55" text-anchor="middle" font-size="10" fill="#666">Broadcast W_G to clients</text>')
    parts.append('<text x="350" y="25" text-anchor="middle" font-size="16">DFD Level 0: DDFed-Markov Context Diagram</text>')
    parts.append("</svg>")

    out = OUTPUT_DIR / "dfd_context_diagram.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {out}")


def draw_dfd_process_flow():
    """Draw DFD Level 1: Process flow as SVG."""
    w, h = 750, 650
    parts = [_svg_header(w, h, "DFD Process Flow")]

    boxes = [
        (80, 100, 220, 60, "P1: Receive &amp; Train", "#E3F2FD"),
        (80, 200, 220, 60, "P2: Markov Noise", "#E8F5E9"),
        (80, 300, 220, 60, "P3: Encrypt", "#FFF3E0"),
        (80, 400, 220, 60, "P4: Vote", "#FCE4EC"),
    ]
    for x, y, wb, hb, text, color in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{wb}" height="{hb}" rx="8" fill="{color}" stroke="black"/>')
        parts.append(f'<text x="{x+wb//2}" y="{y+hb//2+5}" text-anchor="middle" font-size="10">{text}</text>')

    srv = [(350, 150, 200, 50, "P5: Collect Votes", "#E3F2FD"), (350, 250, 200, 50, "P6: Consensus?", "#FFF9C4"),
           (500, 350, 180, 50, "P7: SSS + Fusion", "#E8F5E9"), (500, 430, 180, 50, "P8: FHE Aggregate", "#E8F5E9"),
           (500, 510, 180, 50, "P9: Decrypt", "#E8F5E9")]
    for x, y, wb, hb, text, color in srv:
        parts.append(f'<rect x="{x}" y="{y}" width="{wb}" height="{hb}" rx="8" fill="{color}" stroke="black"/>')
        parts.append(f'<text x="{x+wb//2}" y="{y+hb//2+5}" text-anchor="middle" font-size="10">{text}</text>')
    parts.append('<rect x="250" y="245" width="90" height="60" rx="8" fill="#FFCDD2" stroke="black"/>')
    parts.append('<text x="295" y="280" text-anchor="middle" font-size="10" font-weight="bold">REJECT</text>')

    arrows = [(190, 160, 190, 200), (190, 260, 190, 300), (190, 360, 190, 400), (300, 400, 350, 175),
              (350, 200, 350, 250), (450, 275, 500, 375), (590, 400, 590, 430), (590, 455, 590, 510)]
    for x1, y1, x2, y2 in arrows:
        parts.append(_arrow_svg(x1, y1, x2, y2))

    parts.append('<text x="100" y="75" font-size="14" font-weight="bold">Client</text>')
    parts.append('<text x="400" y="135" font-size="14" font-weight="bold">Server</text>')
    parts.append('<text x="375" y="35" text-anchor="middle" font-size="16">DFD Level 1: DDFed-Markov Process Flow</text>')
    parts.append("</svg>")

    out = OUTPUT_DIR / "dfd_process_flow.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {out}")


def draw_full_flowchart():
    """Draw full DDFed-Markov flowchart as SVG."""
    w, h = 600, 900
    parts = [_svg_header(w, h, "DDFed-Markov Flowchart")]

    nodes = [
        (300, 50, 120, 35, "Start Round t", "#E8F5E9"),
        (300, 110, 200, 35, "Server broadcasts W_G^(t-1)", "#E3F2FD"),
        (300, 180, 280, 35, "Each client: Train -> Markov -> Encrypt -> Vote", "#FFF3E0"),
        (300, 250, 180, 35, "Server collects [W_i], v_i", "#E3F2FD"),
        (300, 320, 160, 45, "Consensus? Sum(v_i) > n/2", "#FFF9C4"),
        (150, 400, 130, 55, "REJECT  W_G^t = W_G^(t-1)", "#FFCDD2"),
        (450, 400, 140, 55, "ACCEPT  SSS+Fusion+FHE", "#C8E6C9"),
        (450, 500, 140, 40, "Decrypt S(t) -> W_G^t", "#C8E6C9"),
        (300, 590, 100, 40, "t < T?", "#FFF9C4"),
        (150, 680, 100, 35, "Next round", "#E8F5E9"),
        (450, 680, 80, 35, "End", "#FFCDD2"),
    ]
    for cx, cy, wb, hb, text, color in nodes:
        x, y = cx - wb // 2, cy - hb // 2
        parts.append(f'<rect x="{x}" y="{y}" width="{wb}" height="{hb}" rx="6" fill="{color}" stroke="black"/>')
        parts.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="10">{text}</text>')

    arrows = [(300, 85, 300, 110), (300, 145, 300, 180), (300, 215, 300, 250), (300, 285, 300, 320),
              (300, 365, 215, 400), (300, 365, 385, 400), (450, 455, 450, 500), (450, 540, 300, 590),
              (250, 428, 250, 680), (300, 610, 200, 680), (300, 610, 410, 680), (200, 698, 260, 80)]
    for x1, y1, x2, y2 in arrows:
        parts.append(_arrow_svg(x1, y1, x2, y2))

    parts.append('<text x="300" y="25" text-anchor="middle" font-size="16">DDFed-Markov Full Process Flowchart</text>')
    parts.append("</svg>")

    out = OUTPUT_DIR / "ddfed_markov_flowchart.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {out}")


def create_html_index():
    """Create HTML file to view all diagrams and save as PNG."""
    svg_files = [
        "markov_state_diagram.svg",
        "dfd_context_diagram.svg",
        "dfd_process_flow.svg",
        "ddfed_markov_flowchart.svg",
    ]
    titles = ["Markov State Diagram", "DFD Context", "DFD Process Flow", "Full Flowchart"]
    items = "\n".join(
        f'<div class="diagram"><h2>{t}</h2><img src="{f}" alt="{t}"/><p>Right-click image → Save image as... to save as PNG</p></div>'
        for t, f in zip(titles, svg_files)
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/><title>DDFed-Markov Diagrams</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:20px auto;padding:20px}}
.diagram{{margin:40px 0;border:1px solid #ccc;padding:20px;border-radius:8px}}
.diagram img{{max-width:100%;height:auto;display:block}}
.diagram p{{color:#666;font-size:12px;margin-top:10px}}</style></head>
<body><h1>DDFed-Markov Diagrams</h1>
<p>Open this file in a browser. Right-click each image → "Save image as..." to save as PNG.</p>
{items}
</body></html>"""
    out = OUTPUT_DIR / "view_diagrams.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out} (open in browser to view/save as PNG)")


def main():
    print("Generating DDFed-Markov diagrams (no external deps)...")
    draw_markov_state_diagram()
    draw_dfd_context()
    draw_dfd_process_flow()
    draw_full_flowchart()
    create_html_index()
    print(f"\nAll diagrams saved to: {OUTPUT_DIR}")
    print("Open view_diagrams.html in a browser, then right-click each image -> Save image as PNG")


if __name__ == "__main__":
    main()

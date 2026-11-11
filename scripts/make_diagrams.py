"""Draw the README and documentation diagrams as SVG, in a light and a dark variant.

    python scripts/make_diagrams.py        # writes docs/images/diagrams/*.svg

Hand-laid-out (no graph layout engine), so the diagrams stay neat at README width. Edit the
layouts below and rerun.
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "images" / "diagrams"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

THEMES = {
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fb", "panel_line": "#e3e8ef", "card": "#ffffff", "line": "#d0d7e2",
        "text": "#0f172a", "muted": "#5b6577", "faint": "#94a3b8", "arrow": "#94a3b8", "shadow": "#0f172a",
        "shadow_opacity": "0.06",
        "blue": "#2563eb", "blue_soft": "#eff4ff", "teal": "#0d9488", "teal_soft": "#ecfdf8",
        "violet": "#7c3aed", "violet_soft": "#f5f0ff", "amber": "#d97706", "amber_soft": "#fff8eb",
        "rose": "#e11d48", "rose_soft": "#fff1f3", "slate_soft": "#f1f5f9",
    },
    "dark": {
        "bg": "#0d1117", "panel": "#131a24", "panel_line": "#263041", "card": "#161e2b", "line": "#2d3a4f",
        "text": "#e6edf3", "muted": "#9aa7b8", "faint": "#6b7a90", "arrow": "#5f6f86", "shadow": "#000000",
        "shadow_opacity": "0.35",
        "blue": "#6ea8ff", "blue_soft": "#15233d", "teal": "#3ed3bd", "teal_soft": "#10302c",
        "violet": "#b294ff", "violet_soft": "#241b3d", "amber": "#f5b544", "amber_soft": "#33270f",
        "rose": "#ff7a93", "rose_soft": "#3a1720", "slate_soft": "#1a2331",
    },
}


class Svg:
    def __init__(self, width: int, height: int, theme: dict, title: str):
        self.w, self.h, self.t, self.parts = width, height, theme, []
        self.title = title

    def add(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=14, weight=400, color="text", anchor="start", mono=False, spacing=None):
        fill = self.t.get(color, color)
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{ls}>{escape(s)}</text>')

    def rect(self, x, y, w, h, r=12, fill="card", stroke="line", width=1, shadow=False, dash=None):
        f, st = self.t.get(fill, fill), self.t.get(stroke, stroke)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        filt = ' filter="url(#shadow)"' if shadow else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{f}" stroke="{st}" '
                 f'stroke-width="{width}"{d}{filt}/>')

    def path(self, d, color="arrow", width=1.6, arrow=True, dash=None):
        c = self.t.get(color, color)
        m = ' marker-end="url(#arrow-%s)"' % color if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{width}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{ds}{m}/>')

    def pill(self, x, y, s, color, size=11.5, mono=False, pad=9):
        w = len(s) * size * (0.62 if mono else 0.54) + 2 * pad
        self.rect(x, y, w, size + 10, r=(size + 10) / 2, fill=f"{color}_soft", stroke=f"{color}_soft")
        self.text(x + w / 2, y + size + 2.5, s, size=size, weight=600, color=color, anchor="middle", mono=mono)
        return w

    def badge(self, cx, cy, n, color):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{self.t[color]}"/>')
        self.text(cx, cy + 4.5, str(n), size=12.5, weight=700, color=self.t["bg"], anchor="middle")

    def card(self, x, y, w, h, title, lines=(), color="blue", n=None, accent=True):
        self.rect(x, y, w, h, fill="card", stroke="line", shadow=True)
        if accent:
            self.add(f'<rect x="{x}" y="{y + 12}" width="3" height="{h - 24}" rx="1.5" fill="{self.t[color]}"/>')
        tx = x + 18
        if n is not None:
            self.badge(x + 28, y + 26, n, color)
            tx = x + 48
        self.text(tx, y + 31, title, size=15, weight=650)
        for i, line in enumerate(lines):
            self.text(x + 18, y + 56 + i * 19, line, size=12.5, color="muted")

    def render(self) -> str:
        t = self.t
        markers = "".join(
            f'<marker id="arrow-{c}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M1,1.5 L8.5,5 L1,8.5" fill="none" stroke="{t[c]}" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>'
            for c in ("arrow", "blue", "teal", "violet", "amber", "rose"))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(self.title)}">'
                f'<title>{escape(self.title)}</title><defs>{markers}'
                f'<filter id="shadow" x="-10%" y="-10%" width="120%" height="140%">'
                f'<feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="{t["shadow"]}" '
                f'flood-opacity="{t["shadow_opacity"]}"/></filter></defs>'
                f'<rect width="{self.w}" height="{self.h}" rx="16" fill="{t["bg"]}"/>'
                + "".join(self.parts) + "</svg>\n")


# -- 1. How it works --------------------------------------------------------------------------------

def architecture(t):
    s = Svg(960, 640, t, "How ForensiX works: media comes in through the Python API, HTTP API or web "
            "console; the engine prepares it, runs detectors and provenance checks, fuses and calibrates "
            "the scores, and returns a verdict with evidence.")
    # inputs
    s.text(40, 46, "YOUR MEDIA", size=11, weight=700, color="faint", spacing="1.2")
    x = 40
    for label in ("Images", "Video", "Audio and voice"):
        s.rect(x, 60, 128 if label != "Audio and voice" else 150, 40, r=20, fill="card", stroke="line", shadow=True)
        s.text(x + (64 if label != "Audio and voice" else 75), 85, label, size=13.5, weight=600, anchor="middle")
        x += (128 if label != "Audio and voice" else 150) + 12
    # ways in
    s.text(520, 46, "WAYS IN", size=11, weight=700, color="faint", spacing="1.2")
    ways = [("Python API", "forensix.load()"), ("HTTP API", "POST /v1/detect"), ("Web console", "/ui")]
    for i, (name, code) in enumerate(ways):
        x = 520 + i * 136
        s.rect(x, 60, 128, 58, r=12, fill="card", stroke="line", shadow=True)
        s.text(x + 64, 82, name, size=13.5, weight=600, anchor="middle")
        s.text(x + 64, 102, code, size=11, color="blue", anchor="middle", mono=True)
        s.path(f"M {x + 64} 120 L {x + 64} 140", color="arrow", arrow=False)
    s.path("M 465 80 L 505 80", color="arrow")
    # the three ways in join one bus into the engine
    s.path("M 584 140 L 856 140", color="arrow", arrow=False)
    s.path("M 720 140 L 720 156 Q 720 164 712 164 L 488 164 Q 480 164 480 172 L 480 180", color="arrow")
    # engine panel
    s.rect(24, 184, 912, 300, r=18, fill="panel", stroke="panel_line")
    s.text(48, 216, "FORENSIX ENGINE", size=11, weight=700, color="faint", spacing="1.2")
    steps = [
        (48, "Prepare", ["Decode, sample frames,", "find faces, cut audio", "into windows"], "blue"),
        (268, "Detect", ["38 detectors: image, video,", "audio, lip-sync, physics,", "identity"], "blue"),
        (488, "Fuse", ["Optional: one score from", "several models, with", "fitted weights"], "violet"),
        (708, "Calibrate", ["Probability, a threshold", "for your false-alarm rate,", "a confidence verdict"], "violet"),
    ]
    for i, (x, title, lines, color) in enumerate(steps):
        s.card(x, 236, 204, 124, title, lines, color=color, n=i + 1)
        if i < 3:
            s.path(f"M {x + 208} 298 L {x + 216} 298", color="arrow")
    # provenance lane
    s.rect(48, 380, 864, 84, r=12, fill="card", stroke="line", shadow=True)
    s.add(f'<rect x="48" y="392" width="3" height="60" rx="1.5" fill="{t["teal"]}"/>')
    s.text(68, 412, "Provenance", size=15, weight=650)
    s.text(68, 436, "What the file proves about itself, independent of any score", size=12.5, color="muted")
    px = 500
    for label in ("C2PA Content Credentials", "Generator watermarks", "Metadata"):
        px += s.pill(px, 403, label, "teal", size=11.5) + 8
    s.path("M 150 360 L 150 378", color="teal", arrow=False, dash="3 4")
    # verdict
    s.path("M 480 486 L 480 512", color="arrow")
    s.rect(170, 518, 620, 98, r=14, fill="card", stroke="blue", width=1.4, shadow=True)
    s.text(194, 548, "Verdict", size=16, weight=700)
    s.text(766, 548, "fake", size=13, weight=700, color="rose", anchor="end")
    s.rect(194, 562, 572, 8, r=4, fill="slate_soft", stroke="slate_soft")
    s.rect(194, 562, 543, 8, r=4, fill="rose", stroke="rose")
    s.text(194, 596, "P(fake) 0.97", size=12.5, weight=600, color="muted")
    x = 300
    for label, color in (("confident: fake", "violet"), ("C2PA: AI-generated", "teal"), ("face swap in 31 of 32 frames", "blue")):
        x += s.pill(x, 584, label, color, size=11) + 8
    return s


# -- 2. From data to production ---------------------------------------------------------------------

def lifecycle(t):
    s = Svg(960, 330, t, "From data to production: measure on your data, adapt with fine-tuning or LoRA, "
            "calibrate, audit fairness, deploy and monitor; drift or an accuracy drop sends you back to "
            "calibration or fine-tuning.")
    steps = [("Measure", "accuracy on", "your data", "blue"), ("Adapt", "fine-tune", "or LoRA", "blue"),
             ("Calibrate", "your false-", "alarm budget", "violet"), ("Audit", "fairness by", "group", "violet"),
             ("Deploy", "laptop, k8s,", "any cloud", "teal"), ("Monitor", "dashboard,", "drift checks", "teal")]
    w, gap, x0, y = 136, 22, 24, 92
    # phase brackets
    for label, first, last, color in (("BUILD ON YOUR DATA", 0, 3, "blue"), ("RUN IN PRODUCTION", 4, 5, "teal")):
        xa, xb = x0 + first * (w + gap), x0 + last * (w + gap) + w
        s.add(f'<path d="M {xa} 66 L {xa} 58 L {xb} 58 L {xb} 66" fill="none" stroke="{t[color]}" '
              f'stroke-width="1.5" stroke-linecap="round" opacity="0.55"/>')
        s.text((xa + xb) / 2, 46, label, size=11, weight=700, color=color, anchor="middle", spacing="1.2")
    for i, (title, l1, l2, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, 112, r=14, fill="card", stroke="line", shadow=True)
        s.badge(x + 26, y + 26, i + 1, color)
        s.text(x + 44, y + 31, title, size=15, weight=650)
        s.text(x + 18, y + 64, l1, size=12.5, color="muted")
        s.text(x + 18, y + 83, l2, size=12.5, color="muted")
        if i < len(steps) - 1:
            s.path(f"M {x + w + 4} {y + 56} L {x + w + gap - 4} {y + 56}", color="arrow")
    # skip adapt when already accurate
    xm, xc = x0 + w / 2, x0 + 2 * (w + gap) + w / 2
    s.path(f"M {xm} {y} Q {xm} 74 {xm + 20} 74 L {xc - 20} 74 Q {xc} 74 {xc} {y - 2}", color="faint",
           dash="4 5", width=1.4)
    s.rect((xm + xc) / 2 - 62, 66, 124, 18, r=9, fill="bg", stroke="bg")
    s.text((xm + xc) / 2, 79, "already accurate", size=11.5, color="muted", anchor="middle")
    # feedback loop from monitor
    xmon = x0 + 5 * (w + gap) + w / 2
    xa, xcal = x0 + 1 * (w + gap) + w / 2, x0 + 2 * (w + gap) + w / 2
    s.path(f"M {xmon} {y + 116} L {xmon} 250 Q {xmon} 266 {xmon - 16} 266 L {xcal + 16} 266 "
           f"Q {xcal} 266 {xcal} 250 L {xcal} {y + 118}", color="amber", width=1.6)
    s.path(f"M {xcal - 16} 266 L {xa + 16} 266 Q {xa} 266 {xa} 250 L {xa} {y + 118}", color="amber",
           width=1.6)
    s.rect(420, 280, 300, 24, r=12, fill="amber_soft", stroke="amber_soft")
    s.text(570, 296, "drift or accuracy drop: recalibrate or retrain", size=12, weight=600, color="amber",
           anchor="middle")
    return s


# -- 3. Deployment -----------------------------------------------------------------------------------

def deployment(t):
    s = Svg(960, 560, t, "Deployment: a load balancer sends management paths to a single manager and "
            "detection to autoscaled inference replicas; both share storage for calibrations and datasets. "
            "Runs on Kubernetes, AWS, Google Cloud and Azure.")
    # clients and load balancer
    s.rect(380, 28, 200, 44, r=22, fill="card", stroke="line", shadow=True)
    s.text(480, 55, "Clients and web console", size=13.5, weight=600, anchor="middle")
    s.path("M 480 74 L 480 96", color="arrow")
    s.rect(330, 100, 300, 58, r=14, fill="card", stroke="line", shadow=True)
    s.text(480, 125, "Load balancer", size=15, weight=650, anchor="middle")
    s.text(480, 145, "one host, TLS, routes by path", size=12.5, color="muted", anchor="middle")
    # routes
    s.path("M 380 160 L 380 178 Q 380 190 368 190 L 210 190 Q 198 190 198 202 L 198 244", color="violet")
    s.path("M 580 160 L 580 178 Q 580 190 592 190 L 750 190 Q 762 190 762 202 L 762 244", color="blue")
    s.rect(48, 204, 300, 24, r=12, fill="violet_soft", stroke="violet_soft")
    s.text(198, 220, "/ui  /v1/datasets  /v1/calibrations ...", size=10.5, weight=600, color="violet",
           anchor="middle", mono=True)
    s.rect(660, 204, 204, 24, r=12, fill="blue_soft", stroke="blue_soft")
    s.text(762, 220, "/v1/detect and the rest", size=10.5, weight=600, color="blue", anchor="middle", mono=True)
    # manager
    s.card(68, 250, 260, 112, "Manager", ["One replica: the single writer.", "Web console, dataset uploads,",
                                          "calibration and fairness jobs"], color="violet")
    # inference: stacked cards
    for k in (2, 1):
        s.rect(632 + 10 * k, 250 - 8 * k, 260, 112, r=12, fill="card", stroke="line")
    s.card(632, 250, 260, 112, "Inference replicas", ["Autoscaled on load, CPU or GPU.",
                                                      "Prediction only: refuse writes,", "read calibrations live"],
           color="blue")
    # shared storage cylinder
    cx, cy, rw = 480, 404, 140
    s.add(f'<path d="M {cx - rw} {cy} L {cx - rw} {cy + 74} A {rw} 16 0 0 0 {cx + rw} {cy + 74} L {cx + rw} {cy} '
          f'" fill="{t["card"]}" stroke="{t["line"]}" stroke-width="1" filter="url(#shadow)"/>')
    s.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rw}" ry="16" fill="{t["slate_soft"]}" stroke="{t["line"]}"/>')
    s.text(cx, cy + 42, "Shared storage", size=15, weight=650, anchor="middle")
    s.text(cx, cy + 63, "calibrations, datasets, galleries", size=12.5, color="muted", anchor="middle")
    s.path("M 198 364 L 198 428 Q 198 440 210 440 L 330 440", color="violet")
    s.text(206, 410, "writes", size=11.5, weight=600, color="violet")
    s.path("M 762 364 L 762 428 Q 762 440 750 440 L 630 440", color="blue")
    s.text(754, 410, "reads", size=11.5, weight=600, color="blue", anchor="end")
    # platforms
    x = 40
    s.text(x, 540, "RUNS ON", size=10.5, weight=700, color="faint", spacing="1.2")
    x += 70
    for label in ("Docker Compose", "Kubernetes", "AWS: ECS, EKS", "Google Cloud: Cloud Run, GKE",
                  "Azure: Container Apps, AKS"):
        x += s.pill(x, 526, label, "teal", size=11) + 8
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in (("architecture", architecture), ("lifecycle", lifecycle), ("deployment", deployment)):
        for theme, colors in THEMES.items():
            (OUT / f"{name}-{theme}.svg").write_text(draw(colors).render())
    print(f"wrote {len(list(OUT.glob('*.svg')))} diagrams to {OUT}")


if __name__ == "__main__":
    main()

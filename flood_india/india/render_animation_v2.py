"""Render the semantic-encoder LinkedIn explainer v2 — INDIA (43 s, 1280x720, 20 fps).
Adapted from the v1 script (same scenes, pacing, palette). All numbers are real:
the window, latent and decoder probability come from the India-trained models run
on the actual Jainagar (Kamla-Balan, Bihar) CWC gauge window of 20 Jun 2024 —
see ~/workspace/edge-apps-prep/semantic_encoder/india/animation_window_india.json
(latent cross-verified by re-running the traced encoder on the dataset window).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INDIA = "/home/hatch/workspace/edge-apps-prep/semantic_encoder/india"
D = json.load(open(f"{INDIA}/animation_window_india.json"))
SEG_L = np.array(D["segment_levels_m"])                       # 60 hourly levels
SEG_T = np.arange(len(SEG_L)) / 24.0                          # days 0..2.458
WIN_LVL = np.array(D["window_levels_m"])                      # 16 levels
REL0 = D["segment_window_position"]["window_start_in_segment"]
REL1 = D["segment_window_position"]["window_end_in_segment"]
WIN_T = SEG_T[REL0:REL1 + 1]
DANGER = D["danger_level_m"]                                  # 67.75

# the exact 16x4 normalised window the encoder consumed (dataset row matched
# on the window levels; channels: level/danger, thr=1.0, rain clip, temp/50)
_z = np.load(f"{INDIA}/dataset_india.npz")
_X = _z["X"].reshape(-1, 16, 4)
_row = [i for i in range(len(_X)) if np.allclose(_X[i][:, 0] * DANGER, WIN_LVL, atol=0.01)]
assert len(_row) == 1, _row
WIN = _X[_row[0]]

LAT_I8 = D["latent_int8"]                                     # [-72 13 10 7 25 -43 16 -96]
LAT = np.array(LAT_I8) / 127.0                                # dequantised, as transmitted
PROB = D["decoder_probability"]                               # 0.6614
THR = D["decision_threshold"]                                 # 0.35

FPS, DUR = 20, 43
NFRAMES = FPS * DUR
INK = "#16324f"; RIVER = "#2471a3"; RED = "#c0392b"; TEAL = "#148f77"
ORANGE = "#e67e22"; PURPLE = "#6c3483"; GREY = "#5d6d7e"; BG = "#fbfcfd"
plt.rcParams["font.family"] = "DejaVu Sans"


def ss(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def caption(fig, text, alpha=1.0):
    fig.text(0.5, 0.045, text, ha="center", va="center", fontsize=17.5,
             color=INK, alpha=alpha, linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#d5dbdb", alpha=0.92 * alpha))


def header(fig, text):
    fig.text(0.5, 0.945, text, ha="center", va="center", fontsize=19,
             fontweight="bold", color=INK)


# ------------------------------------------------------------------ scenes
def scene_title(fig, t):
    a = ss(t / 1.2)
    fig.text(0.5, 0.60, "Task-Oriented Semantic Communications", ha="center",
             fontsize=28, fontweight="bold", color=INK, alpha=a)
    fig.text(0.5, 0.505, "the learned encoder", ha="center", fontsize=23,
             color=TEAL, alpha=a)
    fig.text(0.5, 0.40, "How a flood sensor sends the meaning — not the data",
             ha="center", fontsize=15.5, color=GREY, alpha=a)
    fig.text(0.5, 0.315, "Trained on real CWC gauge records — six stations, five monsoons (2020–2024)",
             ha="center", fontsize=13.5, color=TEAL, alpha=a)


def trace_axes(fig, pos, xlim, ylim, n_show, highlight=False):
    ax = fig.add_axes(pos)
    ax.set_facecolor("white")
    ax.plot(SEG_T[:n_show], SEG_L[:n_show], color=RIVER, lw=1.8)
    ax.axhline(DANGER, color=RED, ls="--", lw=1.4)
    ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]), DANGER + 0.07, f"danger level {DANGER:g} m",
            color=RED, fontsize=12.5, ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.6))
    if highlight:
        ax.axvspan(WIN_T[0], WIN_T[-1], color=ORANGE, alpha=0.18)
        ax.plot(WIN_T, WIN_LVL, "o", color=ORANGE, ms=6, mec=INK, mew=0.7)
        ax.annotate("the latest window:\n16 readings", xy=(1.79, 68.42),
                    xytext=(1.065, 68.56), fontsize=13.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("time (days)", fontsize=13)
    ax.set_ylabel("river level (m)", fontsize=13)
    ax.tick_params(labelsize=11.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return ax


def scene_trace(fig, t):
    header(fig, "A river in monsoon — the Kamla-Balan at Jainagar, Bihar")
    p = ss((t - 3.6) / 4.6)
    trace_axes(fig, [0.085, 0.17, 0.87, 0.68], (-0.05, 2.55), (66.5, 69.0),
               max(2, int(p * len(SEG_T))))
    caption(fig, "A CWC gauge watches the river level — real data, June 2024")


def scene_window(fig, t):
    header(fig, "Every reading, recorded — the raw window")
    zp = ss((t - 9.0) / 2.0)                       # zoom 9 -> 11 s
    xlim = tuple(np.interp(zp, [0, 1], [a, b]) for a, b in [(-0.05, 1.05), (2.55, 2.00)])
    ylim = tuple(np.interp(zp, [0, 1], [a, b]) for a, b in [(66.5, 67.1), (69.0, 68.9)])
    gp = ss((t - 12.5) / 1.0)                      # grid panel 12.5 -> 13.5 s
    width = np.interp(gp, [0, 1], [0.87, 0.545])
    trace_axes(fig, [0.085, 0.17, width, 0.68], xlim, ylim, len(SEG_T), highlight=True)
    if gp > 0.02:
        axg = fig.add_axes([0.685, 0.235, 0.27, 0.56])
        axg.axis("off")
        axg.set_xlim(0, 4); axg.set_ylim(0, 17.6)
        scales = [1.05, 1.05, 2.0, 1.0]
        for r in range(16):
            for c in range(4):
                v = min(1.0, WIN[r, c] / scales[c])
                axg.add_patch(Rectangle((c + 0.08, 15 - r + 0.08), 0.84, 0.84,
                                        fc=RIVER, alpha=float(gp * (0.12 + 0.88 * v)),
                                        ec="white", lw=0.6))
        for c, lab in enumerate(["level", "thr.", "rain", "temp"]):
            axg.text(c + 0.5, 16.35, lab, ha="center", fontsize=10.5, color=INK, alpha=gp)
        axg.text(2, 17.35, "one window = 64 numbers", ha="center", fontsize=12.5,
                 color=INK, alpha=gp)
    caption(fig, "One window = 64 numbers — 256 bytes raw,\nfar more than a satellite message should carry")


def _grid_mini(ax, x0, y0, cw, ch, alpha=1.0):
    scales = [1.05, 1.05, 2.0, 1.0]
    for r in range(16):
        for c in range(4):
            v = min(1.0, WIN[r, c] / scales[c])
            ax.add_patch(Rectangle((x0 + c * cw, y0 + (15 - r) * ch), cw * 0.82, ch * 0.82,
                                   fc=RIVER, alpha=alpha * (0.12 + 0.88 * v), ec="none"))


def _latent_bars(ax, x0, base, scale, grow, labels=True):
    for k in range(8):
        x = x0 + k * 5.2
        h = LAT[k] * 13.0 * grow
        ax.add_patch(Rectangle((x, base if h >= 0 else base + h), 3.4, abs(h),
                               fc=ORANGE, ec=INK, lw=0.6))
        if labels:
            ax.text(x + 1.7, base - 17.5, f"{LAT_I8[k]:+d}", ha="center", fontsize=12.5,
                    color=INK)
    ax.plot([x0 - 2, x0 + 8 * 5.2], [base, base], color=INK, lw=1.0)


def _satellite(ax, x, y):
    ax.add_patch(Rectangle((x - 2.2, y - 1.2), 4.4, 2.4, fc=GREY, ec=INK, lw=0.8))
    ax.add_patch(Rectangle((x - 6.4, y - 0.9), 3.6, 1.8, fc=RIVER, ec=INK, lw=0.8))
    ax.add_patch(Rectangle((x + 2.8, y - 0.9), 3.6, 1.8, fc=RIVER, ec=INK, lw=0.8))


def scene_encoder(fig, t):
    header(fig, "Semantic reporting — compress on the device")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 128); ax.set_ylim(0, 72)
    _grid_mini(ax, 3, 18, 3.1, 2.5)
    ax.text(9.2, 62, "one window\n64 values", ha="center", fontsize=13.5, color=INK)
    ax.add_patch(FancyArrowPatch((17, 38), (26, 38), arrowstyle="-|>", mutation_scale=26,
                                 color=INK, lw=2))
    ax.add_patch(FancyBboxPatch((28, 27), 30, 22, boxstyle="round,pad=1.2",
                                fc=TEAL, alpha=0.14, ec=TEAL, lw=2.2))
    ax.text(43, 41.5, "Semantic encoder", ha="center", fontsize=16.5, fontweight="bold", color=INK)
    ax.text(43, 34.5, "(on the device)", ha="center", fontsize=13, color=GREY)
    ax.add_patch(FancyArrowPatch((60.5, 38), (69, 38), arrowstyle="-|>", mutation_scale=26,
                                 color=INK, lw=2))
    grow = ss((t - 20.0) / 3.0)
    _latent_bars(ax, 72, 38, 1.0, grow)
    ax.text(92, 12.0, "8 latent values — int8, one byte each", ha="center",
            fontsize=12.5, color=GREY)
    # link arc (quadratic Bezier) + travelling dots, 23 -> 27 s
    lp = (t - 23.0) / 4.0
    if lp > 0:
        P0 = np.array([76.0, 47.0]); C = np.array([106.0, 66.0]); P1 = np.array([126.0, 48.0])
        tt = np.linspace(0, 1, 60)
        bx = (1 - tt) ** 2 * P0[0] + 2 * (1 - tt) * tt * C[0] + tt ** 2 * P1[0]
        by = (1 - tt) ** 2 * P0[1] + 2 * (1 - tt) * tt * C[1] + tt ** 2 * P1[1]
        ax.plot(bx, by, ls=(0, (4, 3)), color=GREY, lw=1.4)
        _satellite(ax, 101, 59.5)
        ax.text(109, 64.8, "NTN satellite link", ha="center", fontsize=12.5, color=GREY)
        for k in range(8):
            pk = min(1.0, max(0.0, lp * 1.5 - k * 0.07))
            if 0 < pk < 1:
                dx = (1 - pk) ** 2 * P0[0] + 2 * (1 - pk) * pk * C[0] + pk ** 2 * P1[0]
                dy = (1 - pk) ** 2 * P0[1] + 2 * (1 - pk) * pk * C[1] + pk ** 2 * P1[1]
                ax.plot(dx, dy, "o", color=ORANGE, ms=8, mec=INK, mew=0.7)
    caption(fig, "The learned encoder sends only the meaning: 8 values — 8 bytes")


def scene_gateway(fig, t):
    header(fig, "At the gateway — decide from the 8 values")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 128); ax.set_ylim(0, 72)
    _latent_bars(ax, 4, 44, 1.0, 1.0, labels=False)
    ax.text(24, 63, "8 values arrive", ha="center", fontsize=13.5, color=INK)
    ax.add_patch(FancyArrowPatch((48, 44), (56, 44), arrowstyle="-|>", mutation_scale=26,
                                 color=INK, lw=2))
    ax.add_patch(FancyBboxPatch((58, 33), 30, 22, boxstyle="round,pad=1.2",
                                fc=PURPLE, alpha=0.12, ec=PURPLE, lw=2.2))
    ax.text(73, 47.5, "Gateway decoder", ha="center", fontsize=16.5, fontweight="bold", color=INK)
    ax.text(73, 40.5, "(at the gateway)", ha="center", fontsize=13, color=GREY)
    # probability meter
    mx0, mx1, my = 30, 112, 22.0
    ax.add_patch(Rectangle((mx0, my), mx1 - mx0, 4.6, fc="#e5e8e8", ec=INK, lw=0.8))
    fp = ss((t - 28.0) / 3.2)
    ax.add_patch(Rectangle((mx0, my), (mx1 - mx0) * PROB * fp, 4.6, fc=PURPLE, ec="none"))
    tx = mx0 + (mx1 - mx0) * THR
    ax.plot([tx, tx], [my - 2.2, my + 6.8], color=RED, ls="--", lw=1.6)
    ax.text(tx - 1.5, my + 8.6, f"decision threshold {THR:.2f}", ha="right", fontsize=12.5, color=RED)
    ax.text(mx0, my - 5.4, "decoder probability of a flood event", fontsize=13, color=INK)
    ax.text(mx1, my - 5.4, f"{PROB:.3f}", ha="right", fontsize=15, fontweight="bold", color=PURPLE)
    cap_alpha = 1.0
    if t > 31.6:
        fa = ss((t - 31.6) / 0.7)
        cap_alpha = 1.0 - fa
        ax.add_patch(FancyBboxPatch((36, 6.0), 56, 8.5, boxstyle="round,pad=0.8",
                                    fc=RED, ec="none", alpha=fa))
        ax.text(64, 10.25, "FLOOD EVENT detected", ha="center", va="center", fontsize=18,
                fontweight="bold", color="white", alpha=fa)
    caption(fig, "The gateway reconstructs the decision, not the data", alpha=cap_alpha)


def scene_closing(fig, t):
    card_a = [
        (34.3, 0.700, "Trained on real CWC river data", 21, INK, True),
        (35.5, 0.605, "8 bytes instead of the 256-byte raw window — 32× smaller", 18, INK, False),
        (36.7, 0.510, "Close to the paper's hand-crafted encoding on the 2024 monsoon (F1 0.87 vs 0.91)", 17, INK, False),
        (37.9, 0.415, "The value is the message: learned, not hand-tuned", 17.5, INK, False),
        (39.1, 0.320, "Runs in 42 µs on a Galaxy S24 NPU — Qualcomm AI Hub", 17.5, INK, False),
    ]
    fade_a = 1.0 - ss((t - 39.9) / 0.7) if t > 39.9 else 1.0
    for t0, y, txt, fs, col, bold in card_a:
        a = ss((t - t0) / 0.8) * fade_a
        if a > 0:
            fig.text(0.5, y, txt, ha="center", fontsize=fs, color=col,
                     fontweight="bold" if bold else "normal", alpha=a)
    card_b = [
        (40.7, 0.600, "Rao S. Yenamandra · Mubanga Nsofu · Asokan Ram", 16.5, INK, False),
        (41.2, 0.515, "“Task-oriented Semantic Communications”", 15.5, TEAL, False),
        (41.7, 0.450, "DOI 10.5281/zenodo.23220770", 13.5, GREY, False),
        (42.1, 0.385, "River data: Central Water Commission (CWC), via the GUARDIAN and INDOFLOODS datasets", 13, GREY, False),
    ]
    for t0, y, txt, fs, col, bold in card_b:
        a = ss((t - t0) / 0.5)
        if a > 0:
            fig.text(0.5, y, txt, ha="center", fontsize=fs, color=col,
                     fontweight="bold" if bold else "normal", alpha=a)


def draw(frame):
    fig = plt.gcf()
    fig.clear()
    fig.patch.set_facecolor(BG)
    t = frame / FPS
    if t < 3.0:
        scene_title(fig, t)
    elif t < 9.0:
        scene_trace(fig, t)
    elif t < 17.0:
        scene_window(fig, t)
    elif t < 27.0:
        scene_encoder(fig, t)
    elif t < 34.0:
        scene_gateway(fig, t)
    else:
        scene_closing(fig, t)
    return []


fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor(BG)
ani = animation.FuncAnimation(fig, draw, frames=NFRAMES, interval=1000 / FPS, blit=False)
writer = animation.FFMpegWriter(fps=FPS, codec="libx264",
                                extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"])
ani.save("/tmp/semantic-encoder-animation-v2-india.mp4", writer=writer,
         progress_callback=lambda i, n: print(f"frame {i}/{n}", flush=True) if i % 120 == 0 else None)
print("RENDER DONE")

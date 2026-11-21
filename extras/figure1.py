"""Figure 1 of the paper, v2.0: the pair (danger level shown as a device input)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
W, H = 2000, 1082
fig = plt.figure(figsize=(W/200, H/200), dpi=250); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
BLUE, BLUEF, GREEN, GREENF, INK, GREY = "#3a5f8a", "#e8eef6", "#5c7564", "#edf1ea", "#1a1a1a", "#444444"
def box(x, y, w, h, ec, fc, lw=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14", ec=ec, fc=fc, lw=lw))
def t(x, y, s, size, w="normal", c=INK, style="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=size*0.82, fontweight=w, color=c, style=style, family="DejaVu Sans")
def arrow(x1, y1, x2, y2, c, lw=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>,head_length=0.7,head_width=0.45", color=c, lw=lw, mutation_scale=22))
t(512, 75, "ON THE DEVICE — thin", 21, "bold"); t(512, 128, "kilobytes · battery-powered", 13, c=GREY)
t(1520, 75, "AT THE GATEWAY — rich", 21, "bold"); t(1520, 128, "server · no size or energy budget", 13, c=GREY)
ax.plot([1060, 1060], [185, 1030], ls=(0, (6, 6)), color="#888888", lw=2.2)
box(33, 420, 465, 323, BLUE, BLUEF); t(265, 490, "Sensing window", 17, "bold")
t(265, 592, "16 timesteps × 4 channels", 11.5); t(265, 665, "= 64 values (256 bytes)", 13)
t(265, 700, "incl. level/danger-level ratio", 10, c=GREY)
box(563, 420, 445, 323, BLUE, BLUEF); t(786, 490, "Encoder", 20, "bold"); t(786, 592, "64 → 48 → 24 → 8", 18); t(786, 676, "4,496 parameters · 4.4 KB quantised", 9.5, c=GREY)
arrow(505, 582, 552, 582, INK, 3)
arrow(1020, 582, 1140, 582, BLUE, 4.5)
t(1060, 380, "8-byte message", 17, "bold", c=BLUE)
box(1158, 185, 465, 170, GREEN, "#f8faf6"); t(1390, 228, "Context held at the gateway", 11, "bold")
t(1390, 276, "catchment · upstream gauges", 13, c=GREY); t(1390, 318, "other sensors · history", 13, c=GREY)
arrow(1390, 362, 1390, 405, GREEN, 4)
box(1158, 420, 465, 323, GREEN, GREENF); t(1390, 490, "Decoder", 20, "bold"); t(1390, 592, "8 → 24 → 48", 18)
t(1390, 657, "fuses context the sensor lacks", 11, c=GREY); t(1390, 697, "(catchment, other sensors, history)", 10, c=GREY)
arrow(1626, 582, 1662, 582, INK, 3)
box(1673, 420, 294, 323, GREEN, GREENF); t(1820, 537, "Event probability", 11, "bold"); t(1820, 628, "→ decision", 18)
t(512, 832, "trained offline, jointly with the decoder;", 13.5, c=GREY, style="italic"); t(512, 884, "only the weights travel to the device", 13.5, c=GREY, style="italic")
import os
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure1_pair.png"), dpi=250, facecolor="white"); print("ok")

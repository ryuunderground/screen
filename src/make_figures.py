"""슬라이드용 도식 생성 — 남색 팔레트, 영어. slides/assets/ 에 저장.

새로 만드는 것:
  point_vs_curve.png   핵심 아이디어: 점(contrastive) vs 곡선(우리)
  field_channels.png   원자 -> 다채널 장
  fragment_tree.png    고이동 결합에서만 자르기 + 운동학 트리
  corr_landscape.png   FFT 상관지형 (실제 2D 상관 계산)

기존 재사용(코드 밖): out/torsion_spectrum.png (영어, 그대로).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np
from scipy.signal import fftconvolve

NAVY = "#1F3864"; NAVY_DK = "#142445"; GOLD = "#C8A94B"
INK = "#2B2B2B"; GRAY = "#5A5A5A"; LIGHT = "#EEF1F7"
RED = "#B23A3A"; BLUE = "#3B5BA5"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GRAY, "ytick.color": GRAY, "figure.dpi": 200,
})
NAVYGOLD = LinearSegmentedColormap.from_list("navygold", [NAVY_DK, NAVY, "#5B6E96", "#E8E2CF", GOLD])
OUT = Path(__file__).resolve().parent.parent / "slides" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print("saved:", name)


# --- 1. 점 vs 곡선 --------------------------------------------------------
def point_vs_curve():
    fig = plt.figure(figsize=(10, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.4], wspace=0.22)

    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Circle((0.28, 0.66), 0.075, color=NAVY, zorder=3))
    ax.add_patch(Circle((0.70, 0.34), 0.065, color=GOLD, zorder=3))
    ax.annotate("", (0.66, 0.37), (0.33, 0.63),
                arrowprops=dict(arrowstyle="-", color=GRAY, ls=(0, (4, 3)), lw=1.6))
    ax.text(0.28, 0.80, "receptor", ha="center", fontsize=10, color=NAVY, weight="bold")
    ax.text(0.70, 0.20, "ligand", ha="center", fontsize=10, color="#9c7f27", weight="bold")
    ax.text(0.50, 0.50, "similarity\n= 1 number", ha="center", va="center",
            fontsize=10, color=GRAY, style="italic")
    ax.set_title("Contrastive — a point", fontsize=13, color=NAVY, weight="bold", pad=8)

    ax = fig.add_subplot(gs[1])
    x = np.linspace(0, 360, 500)
    y = (1.0 * np.exp(-((x - 95) ** 2) / (2 * 22 ** 2))
         + 0.45 * np.exp(-((x - 205) ** 2) / (2 * 30 ** 2))
         + 0.55 * np.exp(-((x - 320) ** 2) / (2 * 26 ** 2)) + 0.06)
    ax.fill_between(x, 0, y, color=LIGHT, zorder=1)
    ax.plot(x, y, color=NAVY, lw=2.4, zorder=2)
    px = x[np.argmax(y)]; py = y.max()
    ax.plot([px, px], [0, py], color=GOLD, ls=":", lw=1.6, zorder=2)
    ax.plot(px, py, "o", color=GOLD, ms=11, zorder=4, mec=NAVY, mew=1.2)
    ax.annotate("pose + score", (px, py), (px + 32, py + 0.02),
                fontsize=11, color="#9c7f27", weight="bold")
    ax.set_xlabel("pose   (orientation · position · torsion)", fontsize=11)
    ax.set_ylabel("field overlap", fontsize=11)
    ax.set_title("Ours — a curve", fontsize=13, color=NAVY, weight="bold", pad=8)
    ax.set_xlim(0, 360); ax.set_ylim(0, py * 1.25); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([])
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "point_vs_curve.png")


# --- 2. 다채널 장 --------------------------------------------------------
def field_channels():
    fig = plt.figure(figsize=(8.4, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.25, 1.6], wspace=0.1)

    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rng = np.random.default_rng(3)
    cols = {"C": "#5A5A5A", "O": RED, "N": BLUE}
    pts = [(0.5, 0.55, "C"), (0.63, 0.66, "O"), (0.4, 0.44, "N"),
           (0.58, 0.4, "C"), (0.37, 0.64, "C"), (0.68, 0.5, "O")]
    for i, (px, py, el) in enumerate(pts):
        for (qx, qy, _) in pts[i + 1:]:
            if (px - qx) ** 2 + (py - qy) ** 2 < 0.05:
                ax.plot([px, qx], [py, qy], color="#BBBBBB", lw=2, zorder=1)
    for px, py, el in pts:
        ax.add_patch(Circle((px, py), 0.052, color=cols[el], zorder=3))
    ax.set_title("ligand atoms", fontsize=12, color=NAVY, weight="bold")

    axm = fig.add_subplot(gs[1]); axm.axis("off")
    axm.annotate("", (0.9, 0.5), (0.0, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.5))
    axm.text(0.45, 0.60, "render", ha="center", fontsize=10, color="#9c7f27", weight="bold")

    gsr = gs[2].subgridspec(2, 2, wspace=0.18, hspace=0.42)
    names = ["Occupancy", "Polar", "Hydrophobic", "Charge"]
    xx, yy = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-2, 2, 40))
    fields = [
        np.exp(-(xx ** 2 + yy ** 2) / 1.5),
        np.exp(-((xx - 0.6) ** 2 + (yy - 0.4) ** 2) / 0.6),
        np.exp(-((xx + 0.7) ** 2 + yy ** 2) / 1.0),
        np.tanh(1.5 * xx) * np.exp(-(xx ** 2 + yy ** 2) / 2.0),
    ]
    for k, (nm, fld) in enumerate(zip(names, fields)):
        a = fig.add_subplot(gsr[k // 2, k % 2])
        cm = "RdBu_r" if nm == "Charge" else NAVYGOLD
        a.imshow(fld, cmap=cm, extent=[-2, 2, -2, 2])
        a.set_title(nm, fontsize=9.5, color=INK, pad=2)
        a.set_xticks([]); a.set_yticks([])
    save(fig, "field_channels.png")


# --- 3. 단편 분해 + 트리 -------------------------------------------------
def fragment_tree():
    fig, (axc, axt) = plt.subplots(2, 1, figsize=(6.2, 4.4),
                                   gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.35))
    # 사슬
    axc.set_xlim(0, 10); axc.set_ylim(0, 2); axc.axis("off")
    xs = np.arange(1, 10)
    ys = 1 + 0.25 * np.sin(np.arange(9))
    for i in range(8):
        axc.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], color="#BBBBBB", lw=3, zorder=1)
    for i in range(9):
        axc.add_patch(Circle((xs[i], ys[i]), 0.2, color=NAVY, zorder=3))
    for bi, kind in [(2, "cut"), (5, "cut"), (7, "keep")]:
        mx, my = (xs[bi] + xs[bi + 1]) / 2, (ys[bi] + ys[bi + 1]) / 2
        if kind == "cut":
            axc.plot(mx, my + 0.05, marker="x", color=GOLD, ms=15, mew=3.5, zorder=4)
            axc.text(mx, my + 0.5, "cut", ha="center", fontsize=10, color="#9c7f27", weight="bold")
        else:
            axc.text(mx, my + 0.45, "keep", ha="center", fontsize=9.5, color=GRAY)
    axc.set_title("Cut only where a bond moves atoms a lot (data-driven, ligand-only)",
                  fontsize=10.5, color=NAVY, weight="bold")

    # 트리
    axt.set_xlim(0, 10); axt.set_ylim(0, 3); axt.axis("off")
    nodes = {"A": (2.0, 2.3), "B": (5.0, 1.4), "C": (8.0, 0.6)}
    for (p, c) in [("A", "B"), ("B", "C")]:
        (x0, y0), (x1, y1) = nodes[p], nodes[c]
        axt.annotate("", (x1, y1), (x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
        axt.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, "dihedral φ",
                 ha="center", fontsize=10, color="#9c7f27", style="italic")
    labels = {"A": "Fragment A\n(root)", "B": "Fragment B", "C": "Fragment C"}
    for nm, (x, y) in nodes.items():
        axt.add_patch(FancyBboxPatch((x - 0.95, y - 0.33), 1.9, 0.66,
                      boxstyle="round,pad=0.04", fc=LIGHT, ec=NAVY, lw=1.8, zorder=2))
        axt.text(x, y, labels[nm], ha="center", va="center", fontsize=9.5,
                 color=NAVY, weight="bold", zorder=3)
    axt.set_title("Kinematic tree: one dihedral free per step",
                  fontsize=10.5, color=NAVY, weight="bold")
    save(fig, "fragment_tree.png")


# --- 4. 상관지형 (실제 2D 상관) ------------------------------------------
def corr_landscape():
    n = 120
    g = np.linspace(-6, 6, n)
    X, Y = np.meshgrid(g, g)
    # 리셉터: C자 포켓 (끌림 +), 벽 (반발 -)
    pocket = np.exp(-((X + 1) ** 2 + (Y - 0.5) ** 2) / 2.2) + np.exp(-((X + 0.5) ** 2 + (Y + 1.6) ** 2) / 2.0)
    wall = np.exp(-((X - 2.2) ** 2 + Y ** 2) / 3.0)
    rec = pocket - 0.7 * wall
    lig = np.exp(-((X) ** 2 + (Y) ** 2) / 1.0)  # 리간드 blob
    corr = fftconvolve(rec, lig[::-1, ::-1], mode="same")

    fig, ax = plt.subplots(figsize=(4.7, 4.0))
    im = ax.imshow(corr, cmap=NAVYGOLD, extent=[-6, 6, -6, 6], origin="lower")
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    ax.plot(g[ix], g[iy], "o", ms=16, mfc="none", mec=GOLD, mew=2.8)
    ax.annotate("peak =\ndocked pose", (g[ix], g[iy]), (g[ix] + 1.0, g[iy] + 1.8),
                fontsize=10.5, color="#9c7f27", weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.8))
    ax.set_xlabel("translation x", fontsize=11); ax.set_ylabel("translation y", fontsize=11)
    ax.set_title("Correlation landscape — one FFT", fontsize=12, color=NAVY, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("overlap score", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    save(fig, "corr_landscape.png")


# --- 5. 스크리닝 깔때기 --------------------------------------------------
def funnel():
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 10); ax.axis("off")
    cx = 4.6
    widths = [7.6, 5.9, 4.2, 2.9]
    fonts = [13, 12, 11, 10]
    names = ["Compound library", "Field pre-screen", "FFT docking", "Validation"]
    counts = [r"$10^9$", r"$10^6$", r"$10^3$", "hits"]
    cols = [NAVY_DK, NAVY, "#3E5A8C", GOLD]
    h = 1.7
    y = 9.2
    for i in range(4):
        wt = widths[i]; wb = widths[i + 1] if i < 3 else widths[i] * 0.65
        poly = plt.Polygon([(cx - wt / 2, y), (cx + wt / 2, y),
                            (cx + wb / 2, y - h), (cx - wb / 2, y - h)],
                           closed=True, fc=cols[i], ec="white", lw=2)
        ax.add_patch(poly)
        ax.text(cx, y - h / 2, names[i], ha="center", va="center",
                fontsize=fonts[i], color="white", weight="bold")
        # 오른쪽에 개수 (화살표)
        ax.annotate(counts[i], (cx + wt / 2 + 0.1, y - h / 2), (8.9, y - h / 2),
                    fontsize=15, color=cols[i] if i < 3 else "#9c7f27", weight="bold",
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=1, ls=(0, (2, 2))))
        y -= h + 0.18
    save(fig, "funnel.png")


# --- 6. 등변성: 돌려도 점수 그대로 -------------------------------------
def equivariance():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    th = 0.0
    for ax, ang, tag in [(axes[0], 0.0, "score  S"), (axes[1], 50.0, "score  S  (unchanged)")]:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.axis("off"); ax.set_aspect("equal")
        a = np.deg2rad(ang)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        # 리셉터 포켓(윤곽) + 리간드 blob, 함께 회전
        t = np.linspace(0, 2 * np.pi, 200)
        pocket = np.array([1.9 * np.cos(t), 1.4 * np.sin(t)])
        pocket = R @ pocket
        ax.plot(pocket[0], pocket[1], color=NAVY, lw=2.2)
        lig = R @ np.array([0.4, -0.2])
        ax.add_patch(Circle((lig[0], lig[1]), 0.6, color=GOLD, alpha=0.9))
        ax.text(0, -2.7, tag, ha="center", fontsize=11, color="#9c7f27", weight="bold")
    axes[0].annotate("", (3.25, 1.6), (2.6, 1.6), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    axes[0].text(1.02, 0.56, "rotate", transform=axes[0].transAxes,
                 fontsize=10, color=GRAY, ha="center")
    fig.suptitle("A field rotates with the system — overlap score is preserved",
                 fontsize=12, color=NAVY, weight="bold", y=1.02)
    save(fig, "equivariance.png")


# --- 7. 클러스터 기반 교차검증 -------------------------------------------
def cv_split():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5); ax.axis("off")
    nfold = 5
    bw = 7.2 / nfold
    for f in range(nfold):
        y = 5.3 - f * 0.95
        for k in range(nfold):
            x = 0.4 + k * bw
            val = (k == f)
            ax.add_patch(plt.Rectangle((x, y), bw - 0.08, 0.7,
                         fc=GOLD if val else NAVY, ec="white", lw=1.5))
        ax.text(0.4 + nfold * bw + 0.25, y + 0.35, f"fold {f+1}",
                va="center", fontsize=9.5, color=INK)
    ax.text(0.4 + nfold * bw / 2, 5.3 + 1.0, "PDBbind — grouped by protein / scaffold cluster",
            ha="center", fontsize=10.5, color=NAVY, weight="bold")
    # 범례
    ax.add_patch(plt.Rectangle((8.9, 4.7), 0.4, 0.4, fc=NAVY)); ax.text(9.4, 4.9, "train", va="center", fontsize=9.5, color=INK)
    ax.add_patch(plt.Rectangle((8.9, 4.0), 0.4, 0.4, fc=GOLD)); ax.text(9.4, 4.2, "validation", va="center", fontsize=9.5, color=INK)
    # held-out test
    ax.add_patch(FancyBboxPatch((8.6, 0.6), 3.1, 2.4, boxstyle="round,pad=0.05",
                 fc=LIGHT, ec=NAVY, lw=1.8))
    ax.text(10.15, 2.5, "Held-out test", ha="center", fontsize=11, color=NAVY, weight="bold")
    ax.text(10.15, 1.9, "(kept entirely separate)", ha="center", fontsize=8.5, color=GRAY, style="italic")
    ax.text(10.15, 1.35, "PoseBusters v2", ha="center", fontsize=9.5, color=INK)
    ax.text(10.15, 0.95, "LIT-PCBA", ha="center", fontsize=9.5, color=INK)
    save(fig, "cv_split.png")


if __name__ == "__main__":
    point_vs_curve()
    field_channels()
    fragment_tree()
    corr_landscape()
    funnel()
    equivariance()
    cv_split()

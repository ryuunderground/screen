"""Tier 3 개념 그림 — 숫자 / 화살표 / 럭비공.

분자를 돌릴 때 정보가 세 종류로 갈리고, 짝짓기는 같은 종류끼리만 된다는 것을
그림으로 보이는 스크립트. 데이터 분석이 아니라 개념 도식이다.

패널:
  a  연필 하나: 길이는 안 바뀌고(숫자) 방향은 같이 돌아간다(화살표)
  b  숫자를 적은 지도: 자리만 옮겨간다
  c  화살표를 그린 지도: 자리도 옮기고 방향도 돌아간다
  d  고리 평면을 화살표로 적으면: 위/아래 두 입력이 같은 판을 가리킨다
  e  럭비공으로 적으면: 뒤집어도 하나
  f  화살표 둘을 곱하면: 숫자 1 + 화살표 3 + 럭비공 5 = 9

실행: uv run python src/concepts_equivariance.py
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch

mpl.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.titlesize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 개념별 색을 그림 전체에서 일관되게 쓴다 (숫자 / 화살표 / 럭비공)
C_NUM = "#1f6feb"    # 숫자  (돌려도 안 바뀜)
C_ARR = "#d1495b"    # 화살표 (같이 돌아감, 앞뒤 있음)
C_RUG = "#6a4c93"    # 럭비공 (같이 돌아감, 앞뒤 없음)
C_GREY = "#9aa3b0"   # 회전 전 상태, 보조선
BASE, ANNO = 9, 8    # 글자 크기 2단계 (도식이라 눈금 글자가 없다)

ROT = 55.0           # 그림 전체에서 쓰는 회전각


def rot2(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def arrow(ax, start, end, color, lw=2.0, ls="-", alpha=1.0, mutation=14):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", lw=lw, ls=ls,
                                 color=color, alpha=alpha,
                                 mutation_scale=mutation, shrinkA=0, shrinkB=0,
                                 joinstyle="miter", capstyle="butt"))


def double_headed(ax, p0, p1, color, lw=2.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>", lw=lw, color=color,
                                 mutation_scale=14, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------------------
# a) 연필
# ---------------------------------------------------------------------------
def panel_pencil(ax):
    L = 7.0
    p_before = np.array([L, 0.0])
    p_after = rot2(ROT) @ p_before

    arrow(ax, (0, 0), p_before, C_GREY, lw=2.4, ls=(0, (4, 2)))
    arrow(ax, (0, 0), p_after, C_ARR, lw=2.6)
    ax.add_patch(Arc((0, 0), 4.6, 4.6, theta1=0, theta2=ROT,
                     color=C_GREY, lw=1.0))
    ax.text(2.9, 1.0, f"{ROT:.0f}° 회전", fontsize=ANNO, color=C_GREY)

    ax.text(*(p_before * 1.06 + np.array([0, -0.55])), "회전 전",
            fontsize=ANNO, color=C_GREY, ha="center")
    ax.text(*(p_after * 1.10), "회전 후", fontsize=ANNO, color=C_ARR, ha="center")

    ax.text(0.02, 0.98, "길이 = 7  →  회전 후에도 7   (숫자)", transform=ax.transAxes,
            fontsize=ANNO, color=C_NUM, va="top")
    ax.text(0.02, 0.89, "방향 = 0°  →  회전 후 55°   (화살표)", transform=ax.transAxes,
            fontsize=ANNO, color=C_ARR, va="top")

    ax.set_title("연필 하나에 두 종류의 정보가 있다")
    ax.set_xlim(-1.2, 8.6)
    ax.set_ylim(-1.6, 7.2)


# ---------------------------------------------------------------------------
# b) 숫자를 적은 지도 / c) 화살표를 그린 지도
# ---------------------------------------------------------------------------
SPOT = np.array([0.42, 0.30])     # 지도에서 표시할 지점
WIND = np.array([0.95, 0.30])     # 바람 방향 (정규화 전)


def _disk_frame(ax, cx, label):
    ax.add_patch(Circle((cx, 0), 1.0, fill=False, ec="#444444", lw=1.3))
    ax.text(cx, -1.28, label, fontsize=ANNO, ha="center", color="#444444")


def panel_scalar_map(ax):
    n = 220
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    outside = X ** 2 + Y ** 2 > 1.0

    for cx, ang in ((-1.35, 0.0), (1.35, ROT)):
        c = rot2(ang) @ SPOT
        f = np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * 0.30 ** 2))
        ax.imshow(np.where(outside, np.nan, f), extent=[cx - 1, cx + 1, -1, 1],
                  origin="lower", cmap="Blues", vmin=0, vmax=1.15, zorder=0)
        _disk_frame(ax, cx, "회전 전" if ang == 0 else "회전 후")
        ax.plot([cx + c[0]], [c[1]], marker="o", ms=5, color=C_NUM, zorder=3)
        ax.text(cx + c[0], c[1] + 0.20, "30", fontsize=ANNO, color=C_NUM,
                ha="center", fontweight="bold", zorder=3)

    arrow(ax, (-0.22, 0), (0.22, 0), "#444444", lw=1.6)
    ax.text(0.0, 0.20, "회전", fontsize=ANNO, ha="center", color="#444444")
    ax.text(0.5, 0.045, "적어둔 숫자 30은 그대로 — 자리만 옮겨갔다",
            transform=ax.transAxes, fontsize=ANNO, ha="center", color=C_NUM)

    ax.set_title("숫자를 적은 지도 (온도)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.62, 1.28)


def panel_vector_map(ax):
    g = np.linspace(-0.8, 0.8, 7)
    GX, GY = np.meshgrid(g, g)
    keep = GX ** 2 + GY ** 2 <= 0.86
    GX, GY = GX[keep], GY[keep]

    for cx, ang in ((-1.35, 0.0), (1.35, ROT)):
        R = rot2(ang)
        c = R @ SPOT
        d = R @ (WIND / np.linalg.norm(WIND))
        mag = np.exp(-((GX - c[0]) ** 2 + (GY - c[1]) ** 2) / (2 * 0.52 ** 2))
        ax.quiver(cx + GX, GY, d[0] * mag, d[1] * mag, color=C_ARR,
                  angles="xy", scale_units="xy", scale=3.1, width=0.006,
                  alpha=0.95, zorder=2)
        _disk_frame(ax, cx, "회전 전" if ang == 0 else "회전 후")
        # 하나만 굵게 — 자리도 옮기고 방향도 돈다는 것을 따라가기 쉽게
        ax.quiver([cx + c[0]], [c[1]], [d[0]], [d[1]], color="#7a1020",
                  angles="xy", scale_units="xy", scale=2.4, width=0.012, zorder=4)

    arrow(ax, (-0.22, 0), (0.22, 0), "#444444", lw=1.6)
    ax.text(0.0, 0.20, "회전", fontsize=ANNO, ha="center", color="#444444")
    ax.text(0.5, 0.045, "자리도 옮기고 화살표 방향도 같이 돌았다",
            transform=ax.transAxes, fontsize=ANNO, ha="center", color=C_ARR)

    ax.set_title("화살표를 그린 지도 (바람)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.62, 1.28)


# ---------------------------------------------------------------------------
# d) 고리를 화살표로 / e) 럭비공으로
# ---------------------------------------------------------------------------
def _ring(ax, lw=2.2):
    ax.add_patch(Ellipse((0, 0), 2.3, 0.62, fill=False, ec="#333333", lw=lw))
    ax.add_patch(Ellipse((0, 0), 1.5, 0.36, fill=False, ec="#333333", lw=1.0))


def panel_ring_arrow(ax):
    _ring(ax)
    arrow(ax, (0, 0.05), (0, 1.35), C_ARR, lw=2.6)
    arrow(ax, (0, -0.05), (0, -1.35), C_ARR, lw=2.6)
    ax.text(0.16, 1.18, "위로?", fontsize=ANNO, color=C_ARR)
    ax.text(0.16, -1.30, "아래로?", fontsize=ANNO, color=C_ARR)
    ax.text(0.5, 0.055, "입력이 둘인데 가리키는 분자는 하나 — 모델이 헷갈린다",
            transform=ax.transAxes, fontsize=ANNO, ha="center", color=C_ARR)
    ax.set_title("고리 평면을 화살표로 적으면: 문제")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.95, 1.75)


def panel_ring_rugby(ax):
    ax.add_patch(Ellipse((0, 0), 2.75, 1.45, facecolor=C_RUG, alpha=0.18,
                         edgecolor=C_RUG, lw=1.6))
    _ring(ax)
    double_headed(ax, (0, -0.95), (0, 0.95), C_RUG, lw=2.4)
    ax.text(0.18, 0.72, "축만 있고\n앞뒤가 없다", fontsize=ANNO, color=C_RUG,
            va="top")
    ax.text(0.5, 0.055, "뒤집어도 똑같은 하나 — 입력이 유일해진다",
            transform=ax.transAxes, fontsize=ANNO, ha="center", color=C_RUG)
    ax.set_title("럭비공(눌린 공)으로 적으면: 해결")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.95, 1.75)


# ---------------------------------------------------------------------------
# f) 화살표 x 화살표 = 숫자 + 화살표 + 럭비공
# ---------------------------------------------------------------------------
def panel_product(ax):
    a = np.array([1.05, 0.25])
    b = np.array([0.35, 1.0])
    arrow(ax, (0, 0), a, C_ARR, lw=2.6)
    arrow(ax, (0, 0), b, C_ARR, lw=2.6)
    ax.text(*(a + np.array([0.10, -0.16])), "화살표 a", fontsize=ANNO, color=C_ARR)
    ax.text(*(b + np.array([-0.05, 0.12])), "화살표 b", fontsize=ANNO, color=C_ARR)
    ax.text(0.0, -0.45, "이 둘을 곱하면", fontsize=ANNO, color="#444444",
            ha="center")

    x0, ys = 2.35, [1.28, 0.42, -0.62]

    # 숫자
    ax.text(x0, ys[0], "숫자", fontsize=BASE, color=C_NUM, va="center")
    ax.text(x0 + 0.85, ys[0], "1개", fontsize=ANNO, color=C_NUM, va="center")
    ax.text(x0, ys[0] - 0.30, "얼마나 같은 방향인가", fontsize=ANNO,
            color="#555555", va="center")

    # 화살표 (화면 밖으로 = 외적)
    ax.text(x0, ys[1], "화살표", fontsize=BASE, color=C_ARR, va="center")
    ax.text(x0 + 1.15, ys[1], "3개", fontsize=ANNO, color=C_ARR, va="center")
    ax.add_patch(Circle((x0 + 1.85, ys[1]), 0.16, fill=False, ec=C_ARR, lw=1.6))
    ax.plot([x0 + 1.85], [ys[1]], marker="o", ms=3.5, color=C_ARR)
    ax.text(x0, ys[1] - 0.30, "둘 다에 수직인 방향", fontsize=ANNO,
            color="#555555", va="center")

    # 럭비공
    ax.text(x0, ys[2], "럭비공", fontsize=BASE, color=C_RUG, va="center")
    ax.text(x0 + 1.15, ys[2], "5개", fontsize=ANNO, color=C_RUG, va="center")
    ax.add_patch(Ellipse((x0 + 1.88, ys[2]), 0.52, 0.26, facecolor=C_RUG,
                         alpha=0.25, edgecolor=C_RUG, lw=1.3))
    ax.text(x0, ys[2] - 0.30, "남은 모양 정보", fontsize=ANNO,
            color="#555555", va="center")

    ax.text(0.5, 0.045, "3 × 3 = 9 = 1 + 3 + 5     점수(숫자)는 같은 종류끼리만 나온다",
            transform=ax.transAxes, fontsize=ANNO, ha="center", color="#333333")

    ax.set_title("화살표 둘을 곱하면 세 종류가 나온다")
    ax.set_xlim(-0.75, 4.75)
    ax.set_ylim(-1.35, 1.75)


# ---------------------------------------------------------------------------
def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0))
    ax = axes.ravel()

    panel_pencil(ax[0])
    panel_scalar_map(ax[1])
    panel_vector_map(ax[2])
    panel_ring_arrow(ax[3])
    panel_ring_rugby(ax[4])
    panel_product(ax[5])

    for a in ax:
        a.set_aspect("equal")
        a.axis("off")

    for letter, a in zip("abcdef", ax):
        a.text(0.0, 1.14, letter, transform=a.transAxes, fontweight="bold",
               fontsize=11, va="top", ha="left", fontfamily="DejaVu Sans")

    fig.tight_layout(w_pad=1.2, h_pad=2.4)
    out = "figures/03_concepts_equivariance.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

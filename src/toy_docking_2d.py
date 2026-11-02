"""2D 장난감 도킹 — 설계 노트 §9 1단계를 2차원으로 축소한 것.

학습 하나도 없이, 분자를 '다채널 함수'로 올리고 FFT 상관만으로 포즈를 찾는다.
이 스크립트가 보여주는 것:

  * 점수 = 채널쌍 결합계수를 가진 함수 내적의 합  (설계 노트 §2.1 의 이중선형형)
  * 전하 채널의 결합계수가 '음수'라는 것이 곧 상보성 (+ 는 - 를 찾는다)
  * 모든 평행이동 점수를 FFT 한 번으로 -> 회전만 바깥 루프
  * 그 argmax 가 곧 포즈. 따로 pose 예측기가 필요 없다.

점수 정의 (설계 노트 §2.1):

    S(theta, d) = w_contact * <shell_P , T_d R_theta occ_L>
                - w_clash   * <occ_P   , T_d R_theta occ_L>
                - w_elec    * <charge_P, T_d R_theta charge_L>

마지막 항의 음수 부호가 핵심이다. 같은 부호 전하가 겹치면 상관이 양수가 되고,
거기에 음수 계수가 붙어 벌점이 된다. 반대로 +/- 가 마주보면 보상이 된다.

실행: uv run python src/toy_docking_2d.py
"""

from __future__ import annotations

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, rotate

mpl.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

N = 128                 # 격자
PX = 0.45               # 격자 간격 (A 와 비슷한 축척이라고 생각하면 됨)

W_CONTACT = 2.0         # 접촉 보상
W_CLASH = 8.0           # 충돌 벌점 — 상호침투를 금지할 만큼 커야 한다
W_ELEC = 2.0            # 전하 상보성 (음수 결합계수로 들어감)

C_NEG = "#2166ac"
C_POS = "#b2182b"
C_LIG = "#2a9d8f"
C_MARK = "#e07a00"


# ---------------------------------------------------------------------------
# 기하 만들기
# ---------------------------------------------------------------------------
def disk(cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def gauss_at(cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N]
    return np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sigma ** 2)


# 포켓 경로: 위쪽 표면에서 안으로 들어가 한 번 꺾이는 채널.
# 꺾여 있으므로 올바른 '회전'을 찾아야만 들어맞는다.
CAVITY_PATH = [(26.0, 64.0, 12.0), (40.0, 72.0, 12.0), (55.0, 60.0, 11.0)]


def build_receptor() -> dict[str, np.ndarray]:
    body = disk(64, 64, 44)
    cavity = np.zeros_like(body)
    for cy, cx, r in CAVITY_PATH:
        cavity |= disk(cy, cx, r)
    mask = body & ~cavity

    occ = gaussian_filter(mask.astype(float), 1.6)
    occ /= occ.max()

    # 표면 껍질: 단백질 바로 바깥 한 겹. 리간드가 여기 얹히면 접촉 보상.
    shell = np.clip(gaussian_filter(mask.astype(float), 3.2) - mask, 0.0, None)
    shell /= shell.max()

    # 전하: 입구 쪽 벽에 음전하, 포켓 깊은 쪽 벽에 양전하.
    # 중요 — occ 로 마스킹하면 안 된다. 실제 정전기 퍼텐셜은 단백질 밖(= 포켓 공간)으로
    # 뻗어나가고 리간드는 그 공간에 앉는다. 마스킹하면 전기적 보상을 받으려면
    # 단백질을 파고들어야 하는 비물리적 점수가 되어 최적점이 표면 바깥으로 끌려간다.
    # 전하원은 포켓 '공간' 안(경로 위)에 둔다. APBS 류 퍼텐셜 맵이 포켓 안에서
    # 최대가 되는 것과 같다. 전하원이 벽 안쪽으로 물러나 있으면 리간드가 포켓에
    # 들어가지 않고 전하원 위에 떠 있는 포즈가 이겨버린다.
    charge = (-1.0 * gauss_at(CAVITY_PATH[0][0], CAVITY_PATH[0][1], 7.0)
              + 0.9 * gauss_at(CAVITY_PATH[2][0], CAVITY_PATH[2][1], 7.0))
    return {"mask": mask.astype(float), "occ": occ, "shell": shell, "charge": charge}


# 리간드: 꺾인 5원자 막대. 포켓 경로와 같은 굽힘을 갖도록 잡았다.
LIG_OFFSETS = [(-14.0, -2.0), (-7.0, 2.0), (0.0, 5.0), (7.0, 1.0), (14.0, -4.0)]
LIG_CHARGES = [+1.0, +0.3, 0.0, -0.3, -1.0]     # 한쪽 끝 +, 반대쪽 끝 -


def build_ligand() -> dict[str, np.ndarray]:
    cy, cx = N / 2.0, N / 2.0
    occ = np.zeros((N, N))
    charge = np.zeros((N, N))
    for (dy, dx), q in zip(LIG_OFFSETS, LIG_CHARGES):
        blob = gauss_at(cy + dy, cx + dx, 4.2)
        occ += blob
        charge += q * blob
    occ /= occ.max()
    return {"occ": occ, "charge": charge}


# ---------------------------------------------------------------------------
# 상관: 모든 평행이동을 한 번에
# ---------------------------------------------------------------------------
def xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C[k] = sum_n a[n+k] b[n].  fftshift 후에는 index 가 곧 리간드 중심 위치."""
    return np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))))


def score_map(rec: dict[str, np.ndarray], occ_L: np.ndarray, chg_L: np.ndarray,
              w_elec: float = W_ELEC) -> dict[str, np.ndarray]:
    """채널별 상관을 리간드 크기로 정규화해 O(1) 스케일로 맞춘 뒤 가중합.

    w_elec 를 0 으로 주면 전하 채널을 끈 ablation (형상만으로 포켓을 찾는지 확인).
    """
    n_occ = occ_L.sum()
    n_chg = np.abs(chg_L).sum() + 1e-12
    contact = xcorr(rec["shell"], occ_L) / n_occ
    clash = xcorr(rec["occ"], occ_L) / n_occ
    elec = xcorr(rec["charge"], chg_L) / n_chg
    total = W_CONTACT * contact - W_CLASH * clash - w_elec * elec
    return {"contact": contact, "clash": clash, "elec": elec, "total": total}


def place(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """리간드 장을 (dy, dx) 만큼 옮긴다. 중심이 (N/2+dy, N/2+dx) 로 간다."""
    return np.roll(field, (dy, dx), axis=(0, 1))


def brute_force_score(rec: dict[str, np.ndarray], occ_L: np.ndarray, chg_L: np.ndarray,
                      dy: int, dx: int) -> float:
    """같은 점수를 FFT 없이 직접. FFT 결과 검증용."""
    o = place(occ_L, dy, dx)
    c = place(chg_L, dy, dx)
    n_occ, n_chg = occ_L.sum(), np.abs(chg_L).sum() + 1e-12
    return float(W_CONTACT * np.sum(rec["shell"] * o) / n_occ
                 - W_CLASH * np.sum(rec["occ"] * o) / n_occ
                 - W_ELEC * np.sum(rec["charge"] * c) / n_chg)


# ---------------------------------------------------------------------------
# 탐색: 회전은 바깥 루프, 평행이동은 FFT 로 한 번에
# ---------------------------------------------------------------------------
def search(rec: dict[str, np.ndarray], lig: dict[str, np.ndarray],
           angles: np.ndarray, w_elec: float = W_ELEC) -> tuple[dict, list[float], float]:
    best = {"score": -np.inf}
    per_angle = []
    t0 = time.perf_counter()
    for ang in angles:
        occ_R = rotate(lig["occ"], ang, reshape=False, order=1, mode="constant", cval=0.0)
        chg_R = rotate(lig["charge"], ang, reshape=False, order=1, mode="constant", cval=0.0)
        maps = score_map(rec, occ_R, chg_R, w_elec=w_elec)
        k = int(np.argmax(maps["total"]))
        i, j = np.unravel_index(k, maps["total"].shape)
        s = float(maps["total"][i, j])
        per_angle.append(s)
        if s > best["score"]:
            best = {"score": s, "angle": float(ang), "i": int(i), "j": int(j),
                    "occ": occ_R, "charge": chg_R, "maps": maps}
    elapsed = time.perf_counter() - t0
    return best, per_angle, elapsed


def main() -> None:
    rec = build_receptor()
    lig = build_ligand()
    angles = np.arange(0, 360, 10, dtype=float)

    best, per_angle, elapsed = search(rec, lig, angles)
    dy, dx = best["i"] - N // 2, best["j"] - N // 2

    # FFT 상관이 직접 계산과 같은지 — 최적점과 무작위 5개 지점에서 확인
    checks = [(dy, dx)] + [(int(v), int(w)) for v, w in
                           np.random.default_rng(0).integers(-40, 40, size=(5, 2))]
    devs = []
    for cy, cx in checks:
        fft_val = float(best["maps"]["total"][cy + N // 2, cx + N // 2])
        bf_val = brute_force_score(rec, best["occ"], best["charge"], cy, cx)
        devs.append(abs(fft_val - bf_val))
    max_dev = max(devs)
    assert max_dev < 1e-9, f"FFT 상관이 직접 계산과 불일치: {max_dev:.2e}"

    def decompose(maps: dict[str, np.ndarray], i: int, j: int) -> dict[str, float]:
        return {
            "접촉": W_CONTACT * float(maps["contact"][i, j]),
            "충돌": -W_CLASH * float(maps["clash"][i, j]),
            "전하": -W_ELEC * float(maps["elec"][i, j]),
        }

    terms = decompose(best["maps"], best["i"], best["j"])

    # 진단: 손으로 예상한 포즈(회전 0도, 포켓 경로 중앙)와 비교해 본다.
    maps0 = score_map(rec, lig["occ"], lig["charge"])
    ref_i, ref_j = 40, 66
    ref_terms = decompose(maps0, ref_i, ref_j)
    print("--- 참고: 손으로 예상한 포즈 (0도, 포켓 중앙) ---")
    print(f"  center=({ref_i},{ref_j})  합={sum(ref_terms.values()):+.4f}  "
          + "  ".join(f"{k}={v:+.3f}" for k, v in ref_terms.items()))

    # 비용 비교: 회전 하나당 FFT 3회 vs 모든 평행이동을 직접 루프
    n_shifts = N * N
    t_one_bf = time.perf_counter()
    brute_force_score(rec, best["occ"], best["charge"], 0, 0)
    t_one_bf = time.perf_counter() - t_one_bf
    bf_total_est = t_one_bf * n_shifts * len(angles)

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    ax = axes.ravel()
    qmax = float(np.abs(rec["charge"]).max())

    def draw_molecule(a, occ, charge, boundary_color, title):
        im = a.imshow(charge, cmap="RdBu_r", vmin=-qmax, vmax=qmax, origin="upper")
        a.contour(occ, levels=[0.5], colors=boundary_color, linewidths=1.3)
        a.set_title(title)
        a.set_xticks([]); a.set_yticks([])
        return im

    # (a) 리셉터
    im_q = draw_molecule(ax[0], rec["occ"], rec["charge"], "#222222",
                         "리셉터: 형상(윤곽) + 전하(색)")
    for cy, cx, r in CAVITY_PATH:
        ax[0].add_patch(plt.Circle((cx, cy), r, fill=False, ec="#999999",
                                   lw=0.7, ls=":"))
    ax[0].annotate("포켓", xy=(72, 40), xytext=(104, 18), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))

    # (b) 리간드
    draw_molecule(ax[1], lig["occ"], lig["charge"], C_LIG,
                  "리간드: 양끝에 반대 전하")
    ax[1].set_xlim(34, 94); ax[1].set_ylim(94, 34)

    # (c) 점수 지도
    smap = best["maps"]["total"]
    slim = float(np.abs(smap).max())
    im_s = ax[2].imshow(smap, cmap="PuOr_r", vmin=-slim, vmax=slim, origin="upper")
    ax[2].contour(rec["occ"], levels=[0.5], colors="#222222", linewidths=0.8)
    ax[2].plot([best["j"]], [best["i"]], marker="o", ms=9, mfc="none",
               mec=C_MARK, mew=2.0)
    ax[2].annotate("최고점", xy=(best["j"], best["i"]), xytext=(best["j"] + 26, best["i"] - 24),
                   fontsize=8, color=C_MARK,
                   arrowprops=dict(arrowstyle="->", lw=0.9, color=C_MARK))
    ax[2].set_title(f"회전 {best['angle']:.0f}° 에서 모든 평행이동 점수")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    cb = fig.colorbar(im_s, ax=ax[2], fraction=0.046, pad=0.03)
    cb.set_label("점수 (0 = 상호작용 없음)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # (d) 최적 배치
    occ_best = place(best["occ"], dy, dx)
    chg_best = place(best["charge"], dy, dx)
    ax[3].imshow(rec["charge"], cmap="RdBu_r", vmin=-qmax, vmax=qmax, origin="upper")
    ax[3].contour(rec["occ"], levels=[0.5], colors="#222222", linewidths=1.3)
    ax[3].contourf(occ_best, levels=[0.5, 10], colors=[C_LIG], alpha=0.55)
    ax[3].contour(occ_best, levels=[0.5], colors=[C_LIG], linewidths=1.3)
    for sign, fld, col in (("+", chg_best, C_POS), ("-", -chg_best, C_NEG)):
        k = int(np.argmax(fld))
        yi, xi = np.unravel_index(k, fld.shape)
        ax[3].text(xi, yi, sign, color=col, fontsize=13, fontweight="bold",
                   ha="center", va="center", fontfamily="DejaVu Sans")
    ax[3].set_title("찾아낸 포즈: 점수의 argmax")
    ax[3].set_xticks([]); ax[3].set_yticks([])

    # (e) 회전각 스캔
    ax[4].plot(angles, per_angle, color="#8d99ae", lw=1.3, marker="o", ms=3)
    ax[4].plot([best["angle"]], [best["score"]], marker="o", ms=8, mfc="none",
               mec=C_MARK, mew=2.0)
    ax[4].axhline(0.0, color="#cccccc", lw=0.8, zorder=0)
    ax[4].set_title("각도마다 최고점 — 꺾인 포켓이라 방향이 중요")
    ax[4].set_xlabel("리간드 회전 (도)")
    ax[4].set_ylabel("그 각도의 최고 점수")
    ax[4].set_xticks([0, 90, 180, 270, 360])
    ax[4].annotate(f"{best['angle']:.0f}°", xy=(best["angle"], best["score"]),
                   xytext=(best["angle"] + 28, best["score"]), fontsize=8, color=C_MARK)
    ax[4].spines[["top", "right"]].set_visible(False)
    ax[4].margins(0.06)

    # (f) 항목 분해
    names = list(terms)
    vals = [terms[k] for k in names]
    ypos = np.arange(len(names))[::-1]
    ax[5].axvline(0.0, color="#999999", lw=0.9)
    for y, v in zip(ypos, vals):
        ax[5].plot([0, v], [y, y], color="#8d99ae", lw=1.2, zorder=1)
        ax[5].plot([v], [y], marker="o", ms=8, color=C_LIG if v > 0 else C_POS, zorder=2)
        ax[5].text(v + (0.03 if v > 0 else -0.03) * max(abs(min(vals)), abs(max(vals))) * 4,
                   y, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax[5].set_yticks(ypos); ax[5].set_yticklabels(names)
    ax[5].set_xlabel("최적 포즈에서의 기여")
    ax[5].set_title(f"점수 분해 (합 {best['score']:+.2f})")
    ax[5].spines[["top", "right", "left"]].set_visible(False)
    ax[5].margins(x=0.30, y=0.22)

    cb2 = fig.colorbar(im_q, ax=[ax[0], ax[1]], fraction=0.030, pad=0.02, location="bottom")
    cb2.set_label("전하 (파랑 음 · 빨강 양)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    for letter, a in zip("abcdef", ax):
        a.text(-0.08, 1.10, letter, transform=a.transAxes, fontweight="bold",
               fontsize=11, va="top", ha="left", fontfamily="DejaVu Sans")

    out = "figures/02_toy_docking_2d.png"
    fig.savefig(out)
    print(f"saved {out}")
    print(f"best angle={best['angle']:.0f} deg  center=({best['i']},{best['j']})  score={best['score']:+.4f}")
    print(f"terms: " + "  ".join(f"{k}={v:+.3f}" for k, v in terms.items()))
    print(f"fft_vs_bruteforce_max_dev={max_dev:.2e}")
    print(f"search time (36 rotations x FFT) = {elapsed:.3f} s")
    print(f"direct-loop estimate for same search = {bf_total_est:.1f} s  "
          f"({bf_total_est / elapsed:,.0f}x)")


if __name__ == "__main__":
    main()

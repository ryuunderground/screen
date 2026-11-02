"""푸리에 급수 네 가지 사실 — 이 프로젝트에 필요한 전부.

학부 해석학에서 한 '푸리에 급수'에서 더 나아갈 것이 거의 없다는 것을 수치로 확인하는 스크립트.

1. 직교기저 전개 = 좌표      -> 함수가 복잡해져도 계수 벡터 길이는 내가 정한 N
2. 파르스발(Plancherel)      -> 함수 내적(적분) = 계수 벡터 내적(점곱)
3. 이동 정리                 -> 평행이동 = 계수에 위상 곱. 크기 |a_n|은 불변
4. 상관 정리                 -> 모든 평행이동의 점수를 한 번에. O(N^2) -> O(N log N)

실행: uv run python src/fourier_basics.py
"""

from __future__ import annotations

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- 스타일: 폰트 크기 3단계(제목/주석/눈금), 한글 폰트, 장식 없음 -------------
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_SIMPLE = "#1f6feb"   # 단순 분자 / 원본
C_COMPLEX = "#d1495b"  # 복잡 분자 / 이동된 것
C_REF = "#8d99ae"      # 기준선, 보조
C_FFT = "#2a9d8f"      # FFT

RNG = np.random.default_rng(7)
NGRID = 512                      # 공간 격자점
X = np.linspace(0.0, 2.0 * np.pi, NGRID, endpoint=False)


# ---------------------------------------------------------------------------
# 1D "원자 밀도 프로파일": 주기 구간 위 가우시안 합. 분자를 함수로 올린 것의 1D 판.
# ---------------------------------------------------------------------------
def atom_profile(centers: np.ndarray, widths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """주기 경계를 지키는 가우시안 합. centers 는 [0, 2pi)."""
    out = np.zeros_like(X)
    for c, w, a in zip(centers, widths, weights):
        d = np.angle(np.exp(1j * (X - c)))          # 주기적 최단거리
        out += a * np.exp(-0.5 * (d / w) ** 2)
    return out


def random_profile(n_atoms: int, rng: np.random.Generator, width: float = 0.18) -> np.ndarray:
    """최대값 1로 정규화. 상대오차/파르스발 검증은 스케일에 무관하므로 영향 없음.

    width = 원자 하나의 가우시안 폭(rad). 실제 분자에서 원자 크기는 분자가 커져도
    변하지 않으므로, 두 분자를 비교할 때는 width 를 고정하고 개수만 바꾸는 것이 옳다.
    """
    p = atom_profile(
        centers=rng.uniform(0, 2 * np.pi, n_atoms),
        widths=np.full(n_atoms, width),
        weights=rng.uniform(0.5, 1.0, n_atoms),
    )
    return p / p.max()


# ---------------------------------------------------------------------------
# 사실 1 + 2: 계수 = 좌표, 그리고 적분이 점곱이 된다
# ---------------------------------------------------------------------------
def coeffs(f: np.ndarray) -> np.ndarray:
    """푸리에 계수 a_n = (1/N) sum f_k e^{-2pi i n k / N}. np.fft.fft 의 정규화 버전."""
    return np.fft.fft(f) / f.size


def reconstruct(a: np.ndarray, n_keep: int) -> np.ndarray:
    """저주파 n_keep 개 조화항만 남겨 복원 (= 급수 절단 = band limit)."""
    a_cut = np.zeros_like(a)
    a_cut[:n_keep + 1] = a[:n_keep + 1]
    a_cut[-n_keep:] = a[-n_keep:]              # 실함수이므로 음의 주파수도 짝으로
    return np.real(np.fft.ifft(a_cut) * a.size)


def inner_product_space(f: np.ndarray, g: np.ndarray) -> float:
    """<f,g> = integral f g dx, 공간에서 직접 수치적분."""
    dx = 2 * np.pi / f.size
    return float(np.sum(f * g) * dx)


def inner_product_coeff(f: np.ndarray, g: np.ndarray) -> float:
    """같은 값을 계수에서: 2pi * sum_n a_n conj(b_n).  <- 파르스발

    유도: f_k = sum_n a_n e^{2pi i n k/N} 이므로 sum_k f_k conj(g_k) = N sum_n a_n conj(b_n).
    한편 integral f g dx = (2pi/N) sum_k f_k g_k.  두 식을 합치면 계수 쪽 N 이 정확히 상쇄된다.
    """
    a, b = coeffs(f), coeffs(g)
    return float(np.real(2 * np.pi * np.sum(a * np.conj(b))))


# ---------------------------------------------------------------------------
# 사실 4: 상관 — 모든 평행이동의 겹침을 한 번에
# ---------------------------------------------------------------------------
def correlation_bruteforce(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """C[k] = sum_n f[n+k] g[n]  —  모든 이동 k 에 대해 루프로."""
    n = f.size
    out = np.empty(n)
    for k in range(n):
        out[k] = np.dot(np.roll(f, -k), g)
    return out


def correlation_fft(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """같은 것을 FFT 로: IFFT( FFT(f) * conj(FFT(g)) )."""
    return np.real(np.fft.ifft(np.fft.fft(f) * np.conj(np.fft.fft(g))))


def timing_scaling(sizes: list[int]) -> tuple[list[float], list[float]]:
    """격자 크기별 brute force / FFT 벽시계 시간."""
    t_brute, t_fft = [], []
    for n in sizes:
        f = RNG.standard_normal(n)
        g = RNG.standard_normal(n)
        t0 = time.perf_counter()
        correlation_bruteforce(f, g)
        t_brute.append(time.perf_counter() - t0)
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            correlation_fft(f, g)
        t_fft.append((time.perf_counter() - t0) / reps)
    return t_brute, t_fft


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def main() -> None:
    n_keep = 16                                  # 양쪽 다 이 차수로 자른다

    f_simple = random_profile(4, np.random.default_rng(1))
    f_complex = random_profile(22, np.random.default_rng(2))
    rec_simple = reconstruct(coeffs(f_simple), n_keep)
    rec_complex = reconstruct(coeffs(f_complex), n_keep)

    def rel_err(orig: np.ndarray, rec: np.ndarray) -> float:
        return float(np.linalg.norm(orig - rec) / np.linalg.norm(orig))

    err_s, err_c = rel_err(f_simple, rec_simple), rel_err(f_complex, rec_complex)
    dim = 2 * n_keep + 1                         # 실함수 계수 자유도

    # 진단: 절단 오차를 결정하는 것은 원자 '개수'인가 '폭'인가?
    print("--- 절단 오차 (n_keep=16) : 원자폭 x 원자수 ---")
    for width in (0.06, 0.12, 0.18, 0.30):
        row = []
        for n_at in (4, 22):
            f = random_profile(n_at, np.random.default_rng(5), width=width)
            row.append(rel_err(f, reconstruct(coeffs(f), n_keep)))
        print(f"width={width:.2f} rad  원자4={row[0]:7.2%}  원자22={row[1]:7.2%}")

    # 파르스발 검증: 무작위 함수쌍 150개
    pairs = []
    rng = np.random.default_rng(11)
    for _ in range(150):
        f = random_profile(int(rng.integers(3, 25)), rng)
        g = random_profile(int(rng.integers(3, 25)), rng)
        pairs.append((inner_product_space(f, g), inner_product_coeff(f, g)))
    ip_space, ip_coeff = map(np.array, zip(*pairs))
    # 각 쌍의 상대오차로 보면 '거의 겹치지 않는' 함수쌍(내적 ~1e-9)이 분모가 되어
    # 지표가 왜곡된다. 내적의 전체 스케일 대비 최대 절대편차로 보는 것이 옳다.
    parseval_dev = float(np.max(np.abs(ip_space - ip_coeff)) / np.max(np.abs(ip_space)))

    # 이동 불변성
    shift_px = 137
    f_orig = random_profile(8, np.random.default_rng(3))
    f_shift = np.roll(f_orig, shift_px)
    a_orig, a_shift = coeffs(f_orig), coeffs(f_shift)
    nmax_show = 24
    mag_dev = float(np.max(np.abs(np.abs(a_orig[:nmax_show]) - np.abs(a_shift[:nmax_show]))))

    # 상관
    g_small = random_profile(3, np.random.default_rng(4))
    corr_b = correlation_bruteforce(f_orig, g_small)
    corr_f = correlation_fft(f_orig, g_small)
    corr_dev = float(np.max(np.abs(corr_b - corr_f)) / np.max(np.abs(corr_b)))

    sizes = [128, 256, 512, 1024, 2048, 4096]
    t_brute, t_fft = timing_scaling(sizes)
    speedup = t_brute[-1] / t_fft[-1]

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))
    ax = axes.ravel()

    # (a) 계수 벡터 길이는 함수 복잡도와 무관
    offset = 1.7
    ax[0].plot(X, f_simple, color=C_SIMPLE, lw=1.6, label=f"원자 4개 (복원오차 {err_s:.1%})")
    ax[0].plot(X, rec_simple, color=C_SIMPLE, lw=1.0, ls="--")
    ax[0].plot(X, f_complex - offset, color=C_COMPLEX, lw=1.6, label=f"원자 22개 (복원오차 {err_c:.1%})")
    ax[0].plot(X, rec_complex - offset, color=C_COMPLEX, lw=1.0, ls="--")
    ax[0].set_title("원자가 늘어도 계수 벡터 길이는 그대로")
    ax[0].set_xlabel("위치 (rad)")
    ax[0].set_yticks([])
    ax[0].set_ylabel("밀도 (두 분자를 위아래로)")
    ax[0].set_ylim(-offset - 0.35, 1.72)
    ax[0].text(0.02, 0.015, f"둘 다 {n_keep}차에서 절단 → {dim}차원 · 실선 원본, 점선 복원",
               transform=ax[0].transAxes, va="bottom", fontsize=8)
    ax[0].legend(frameon=False, loc="upper center", ncol=1, fontsize=8,
                 handlelength=1.6, borderaxespad=0.1)

    # (b) 파르스발
    lim = [min(ip_space.min(), ip_coeff.min()), max(ip_space.max(), ip_coeff.max())]
    ax[1].plot(lim, lim, color=C_REF, lw=1.0, zorder=1)
    ax[1].scatter(ip_space, ip_coeff, s=14, color=C_SIMPLE, alpha=0.75, lw=0, zorder=2)
    ax[1].set_title("적분이 점곱이 된다")
    ax[1].set_xlabel(r"공간에서 수치적분  $\int f g\,dx$")
    ax[1].set_ylabel(r"계수에서 점곱  $2\pi\sum_n a_n \bar{b}_n$")
    ax[1].text(0.04, 0.95, f"함수쌍 150개\n최대 상대편차 {parseval_dev:.1e}",
               transform=ax[1].transAxes, va="top", fontsize=8)
    ax[1].margins(0.05)

    # (c) 공간에서는 다른 함수
    ax[2].plot(X, f_orig, color=C_SIMPLE, lw=1.6, label="원본")
    ax[2].plot(X, f_shift, color=C_COMPLEX, lw=1.6, label="평행이동")
    ax[2].set_title("공간에서는 서로 다른 함수")
    ax[2].set_xlabel("위치 (rad)")
    ax[2].set_ylabel("밀도")
    ax[2].legend(frameon=False, loc="upper right", fontsize=8)
    ax[2].margins(0.04)

    # (d) 계수 크기는 같다
    n_idx = np.arange(nmax_show)
    ax[3].plot(n_idx, np.abs(a_orig[:nmax_show]), color=C_SIMPLE, lw=1.4, marker="o", ms=4, label="원본")
    ax[3].plot(n_idx, np.abs(a_shift[:nmax_show]), color=C_COMPLEX, lw=0, marker="x", ms=6, label="평행이동")
    ax[3].set_title("계수 크기는 위치에 무관 — 불변 디스크립터")
    ax[3].set_xlabel("조화항 번호 $n$")
    ax[3].set_ylabel(r"$|a_n|$")
    ax[3].text(0.35, 0.95, f"최대 차이 {mag_dev:.1e}\n위치 정보는 위상에만",
               transform=ax[3].transAxes, va="top", fontsize=8)
    ax[3].legend(frameon=False, loc="center right", fontsize=8)
    ax[3].margins(0.04)

    # (e) 상관: 모든 이동의 점수를 한 번에
    shifts = np.arange(NGRID)
    ax[4].plot(shifts, corr_b, color=C_REF, lw=2.6, label="직접 루프 (이동마다)")
    ax[4].plot(shifts, corr_f, color=C_FFT, lw=1.1, label="FFT 한 번")
    k_best = int(np.argmax(corr_b))
    ax[4].plot([k_best], [corr_b[k_best]], marker="v", ms=7, color="#e07a00", lw=0)
    ax[4].annotate("최적 이동", xy=(k_best, corr_b[k_best]), xytext=(k_best + 60, corr_b[k_best] * 0.92),
                   fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.7, color="#e07a00"))
    ax[4].set_title("모든 평행이동의 겹침을 한 번에")
    ax[4].set_xlabel("이동량 (격자점)")
    ax[4].set_ylabel("겹침 점수")
    ax[4].text(0.04, 0.22, f"두 방법 최대 상대차 {corr_dev:.1e}",
               transform=ax[4].transAxes, fontsize=8)
    ax[4].legend(frameon=False, loc="upper left", fontsize=8)
    ax[4].margins(0.04)

    # (f) 비용 스케일링
    ms_brute = [t * 1e3 for t in t_brute]
    ms_fft = [t * 1e3 for t in t_fft]
    ax[5].plot(sizes, ms_brute, color=C_REF, lw=1.5, marker="o", ms=4)
    ax[5].plot(sizes, ms_fft, color=C_FFT, lw=1.5, marker="s", ms=4)
    ax[5].set_xscale("log", base=2)
    ax[5].set_yscale("log")
    ax[5].set_title("격자가 커질수록 벌어지는 비용")
    ax[5].set_xlabel("격자점 수 $N$")
    ax[5].set_ylabel("1회 계산 시간 (ms)")
    ax[5].set_xticks(sizes)
    ax[5].set_xticklabels([str(s) for s in sizes])
    ax[5].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax[5].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax[5].text(0.03, 0.97, r"직접 루프  $O(N^2)$", transform=ax[5].transAxes,
               va="top", ha="left", color=C_REF, fontsize=8)
    ax[5].text(0.03, 0.20, r"FFT  $O(N\log N)$", transform=ax[5].transAxes,
               va="top", ha="left", color=C_FFT, fontsize=8)
    ax[5].text(0.97, 0.04, f"N={sizes[-1]} 에서 {speedup:,.0f}배 차이",
               transform=ax[5].transAxes, va="bottom", ha="right", fontsize=8)
    ax[5].margins(0.10)

    for letter, a in zip("abcdef", ax):
        a.text(-0.14, 1.12, letter, transform=a.transAxes, fontweight="bold",
               fontsize=11, va="top", ha="left", fontfamily="DejaVu Sans")

    fig.tight_layout(w_pad=1.8, h_pad=2.0)
    out = "figures/01_fourier_intuition.png"
    fig.savefig(out)
    print(f"saved {out}")
    print(f"dim={dim} err_simple={err_s:.4f} err_complex={err_c:.4f}")
    print(f"parseval_max_rel_dev={parseval_dev:.3e}")
    print(f"shift_mag_max_dev={mag_dev:.3e}")
    print(f"corr_max_rel_dev={corr_dev:.3e} speedup_at_N{sizes[-1]}={speedup:.1f}")


if __name__ == "__main__":
    main()

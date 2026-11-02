"""경첩 회전을 계수 행렬곱으로 d번 할 때 오차가 얼마나 쌓이는가.

배경: 장을 유한 기저로 자르면 '원점을 지나지 않는 축 둘레의 회전'(= 경첩)을
정확히 표현할 수 없다. ESF 본문이 지적한 국소→전역 재전개 문제다.
단편 분해는 이 근사를 사슬 깊이만큼 연달아 적용하므로 오차가 누적된다.

묻는 것: 깊이 d에 대해 점수 오차가 얼마나 빨리 늘어나는가?
  천천히 늘면 -> 설계가 선다
  빨리 늘면   -> 깊이 상한이 생기고, 그게 곧 '토막 몇 개까지 허용되나'의 답

측정 방법:
  c_근사 = M(g_d) ... M(g_1) c_0      (계수에 행렬 d번)
  c_정답 = project(합성변환에서 직접 그린 장)
  둘의 장 오차와 점수 오차를 비교.

M(g)는 투영 연산자로 만든다 — 해당 기저에서 가능한 **최적** 선형 연산자이므로
여기서 나오는 오차는 하한이다. 실제 구현은 이보다 나쁘면 나빴지 좋을 수 없다.

ponytail: 점유 채널 하나만 쓴다. 기저 절단이 강체 운동을 얼마나 못 쫓아가는지는
          채널에 무관한 성질이라 하나면 답이 나온다.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from scipy.special import factorial, lpmv

from torsion_spectrum import OUT, RADII, load_complex, torsion_quadruples

GAUSS_SIGMA = 1.0  # 원자 blob 폭 (Å)


# --- 실수 구면조화 --------------------------------------------------------
def real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """실수 구면조화 Y_l^m. theta = 극각, phi = 방위각."""
    am = abs(m)
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - am) / factorial(l + am))
    p = lpmv(am, l, np.cos(theta))
    if m == 0:
        return norm * p
    return np.sqrt(2.0) * norm * p * (np.cos(am * phi) if m > 0 else np.sin(am * phi))


def build_basis(xyz: np.ndarray, lmax: int, nrad: int, rmax: float) -> np.ndarray:
    """격자점에서 기저 평가. (n_basis, n_grid) float32.

    psi_{j,l,m}(x) = exp(-(r-r_j)^2 / 2s^2) * Y_l^m(각도)
    """
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(np.divide(xyz[:, 2], r, out=np.zeros_like(r), where=r > 0), -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    centers = np.linspace(0.0, rmax, nrad)
    s = (centers[1] - centers[0]) if nrad > 1 else rmax / 2
    radial = np.exp(-((r[None, :] - centers[:, None]) ** 2) / (2 * s * s))

    ang = np.stack([real_sph_harm(l, m, theta, phi)
                    for l in range(lmax + 1) for m in range(-l, l + 1)])
    # 바깥은 잘라낸다 (박스 모서리에서 기저가 발산하는 걸 막음)
    cut = (r <= rmax).astype(np.float32)
    return (radial[:, None, :] * ang[None, :, :] * cut).reshape(-1, len(r)).astype(np.float32)


# --- 투영 ----------------------------------------------------------------
def gram_pinv(B: np.ndarray, dV: float, rtol=1e-6):
    """Gram 행렬과 그 유사역행렬.

    반경 가우시안들이 겹쳐서 G가 심하게 ill-conditioned하다. 그냥 solve를 쓰면
    행렬곱을 반복할 때 발산한다 (lmax=10에서 장오차 1371%로 터졌었다).
    고유분해로 작은 고유값을 잘라내서 안정화한다.
    """
    G = (B @ B.T).astype(np.float64) * dV
    w, V = np.linalg.eigh(G)
    keep = w > rtol * w.max()
    Gp = (V[:, keep] / w[keep]) @ V[:, keep].T
    return G, Gp, int(keep.sum())


def project(B: np.ndarray, Gp: np.ndarray, f: np.ndarray, dV: float) -> np.ndarray:
    """장 f를 기저에 투영해 계수를 얻는다."""
    return Gp @ ((B @ f.astype(np.float32)).astype(np.float64) * dV)


def transfer(B: np.ndarray, Gp: np.ndarray, xyz: np.ndarray, dV: float,
             R: np.ndarray, t: np.ndarray, lmax: int, nrad: int, rmax: float) -> np.ndarray:
    """강체 운동 g=(R,t)의 계수 연산자 M. 보간 없이 기저를 변환좌표에서 재평가.

    (g.psi)(x) = psi(g^-1 x) 이므로, 변환된 좌표에서 기저를 평가해 투영하면 된다.
    """
    inv = (xyz - t) @ R  # R^T (x - t)
    Bg = build_basis(inv, lmax, nrad, rmax)
    return Gp @ ((B @ Bg.T).astype(np.float64) * dV)


# --- 장 그리기 -----------------------------------------------------------
def render(xyz: np.ndarray, atoms: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """원자마다 가우시안 blob. 점유 채널."""
    f = np.zeros(len(xyz), dtype=np.float32)
    for a, rad in zip(atoms, radii):
        d2 = ((xyz - a) ** 2).sum(1)
        near = d2 < (4 * GAUSS_SIGMA + rad) ** 2
        f[near] += np.exp(-d2[near] / (2 * GAUSS_SIGMA**2)).astype(np.float32)
    return f


def axis_rotation(p: np.ndarray, u: np.ndarray, angle: float):
    """점 p를 지나고 방향 u인 축 둘레 회전. (R, t) 반환 — x -> Rx + t."""
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R, p - R @ p


# --- 사슬 추출 -----------------------------------------------------------
def torsion_chain(mol):
    """한 경로 위에 놓인 회전결합들을 뿌리에서 잎 순서로. [(pivot, axis, 움직이는 원자들)]."""
    quads = torsion_quadruples(mol)
    dm = Chem.GetDistanceMatrix(mol)
    # 가장 먼 두 원자를 잇는 경로를 사슬로 삼는다
    i0, j0 = np.unravel_index(np.argmax(dm), dm.shape)
    path = [a for a in range(mol.GetNumAtoms())
            if dm[i0, a] + dm[a, j0] == dm[i0, j0]]
    path.sort(key=lambda a: dm[i0, a])
    pset = set(path)

    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions())
    chain = []
    for (_, j, k, _) in quads:
        if j not in pset or k not in pset:
            continue
        # j가 뿌리(i0) 쪽, k가 잎 쪽이 되도록
        if dm[i0, j] > dm[i0, k]:
            j, k = k, j
        em = Chem.RWMol(mol)
        em.RemoveBond(j, k)
        frags = Chem.GetMolFrags(em.GetMol(), asMols=False, sanitizeFrags=False)
        moving = next((set(f) for f in frags if k in f), None)
        if moving is None or j in moving:
            continue
        chain.append((pos[j], pos[k] - pos[j], sorted(moving)))
    chain.sort(key=lambda c: len(c[2]), reverse=True)  # 뿌리 쪽(많이 움직임)부터
    return chain


# --- 실험 ----------------------------------------------------------------
def run(pdb_id="1HSG", ccd="MK1", lmax=8, nrad=8, rmax=12.0, spacing=0.7,
        angle_deg=40.0, max_depth=6):
    cache = OUT / "cache"
    mol, rec_xyz, rec_el = load_complex(pdb_id, ccd, cache)
    pos = np.array(mol.GetConformer().GetPositions())
    lig_radii = np.array([RADII.get(a.GetSymbol().upper(), 1.9) for a in mol.GetAtoms()])

    center = pos.mean(0)
    n = int(2 * rmax / spacing) + 1
    ax = np.linspace(-rmax, rmax, n)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    xyz = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    dV = (ax[1] - ax[0]) ** 3

    print(f"{pdb_id}/{ccd}  격자 {n}^3={len(xyz)}  "
          f"기저 {(lmax+1)**2 * nrad} (lmax={lmax}, nrad={nrad})")

    B = build_basis(xyz, lmax, nrad, rmax)
    _, Gp, rank = gram_pinv(B, dV)
    print(f"  Gram 유효 랭크 {rank}/{len(Gp)}")

    # 리셉터 장 -> 점수용 A_n = <psi_n, phi_P>
    rec_local = rec_xyz - center
    keep = np.linalg.norm(rec_local, axis=1) < rmax + 4
    rec_radii = np.array([RADII.get(e, 1.9) for e in rec_el])[keep]
    phi_P = render(xyz, rec_local[keep].astype(np.float32), rec_radii)
    A = (B @ phi_P).astype(np.float64) * dV

    chain = torsion_chain(mol)[:max_depth]
    if not chain:
        print("회전결합 사슬 없음")
        return None
    # 마지막 경첩 너머 원자들이 말단 토막
    frag = np.array(chain[-1][2])
    print(f"사슬 깊이 {len(chain)}, 말단 토막 원자 {len(frag)}개, "
          f"각 경첩 {angle_deg}° 회전")

    frag_pos = (pos[frag] - center).astype(np.float32)
    frag_rad = lig_radii[frag]
    c = project(B, Gp, render(xyz, frag_pos, frag_rad), dV)

    ang = np.deg2rad(angle_deg)
    rows = []
    Rc, tc = np.eye(3), np.zeros(3)  # 합성 변환 (정답용)
    for d, (p, u, _) in enumerate(chain, 1):
        R, t = axis_rotation((p - center).astype(np.float64), u.astype(np.float64), ang)
        M = transfer(B, Gp, xyz, dV, R, t, lmax, nrad, rmax)
        c = M @ c                                  # 근사: 계수에 행렬곱
        Rc, tc = R @ Rc, R @ tc + t                # 정답: 변환을 합성
        exact = render(xyz, (frag_pos @ Rc.T + tc).astype(np.float32), frag_rad)
        c_exact = project(B, Gp, exact, dV)

        approx_field = B.T @ c.astype(np.float32)
        ref_norm = np.sqrt((exact.astype(np.float64) ** 2).sum() * dV)
        rows.append(dict(
            depth=d,
            field_err=float(np.sqrt(((approx_field - exact).astype(np.float64) ** 2).sum() * dV) / ref_norm),
            basis_err=float(np.sqrt(((B.T @ c_exact.astype(np.float32) - exact).astype(np.float64) ** 2).sum() * dV) / ref_norm),
            score_approx=float(c @ A), score_exact=float(c_exact @ A),
            # 장 노름이 커지면 M이 수축사상이 아니라는 뜻 — 발산의 직접 증거
            norm_ratio=float(np.sqrt((approx_field.astype(np.float64) ** 2).sum() * dV) / ref_norm),
        ))
        r = rows[-1]
        r["score_err"] = abs(r["score_approx"] - r["score_exact"])
        print(f"  d={d}  장오차 {r['field_err']*100:7.2f}%  "
              f"(기저한계 {r['basis_err']*100:5.2f}%)  노름비 {r['norm_ratio']:6.3f}  "
              f"점수 {r['score_approx']:9.2f} vs {r['score_exact']:9.2f}  "
              f"오차 {r['score_err']:8.3f}")
    return rows


def basis_sweep(pdb_id="1HSG", ccd="MK1", rmax=12.0, spacing=0.7, max_depth=6,
                configs=((6, 6), (10, 8), (14, 10), (20, 12), (26, 14))):
    """기저를 얼마나 크게 해야 토막 장을 표현할 수 있는가.

    전사 연산자가 필요 없다 — 토막을 그려서 투영하고 되돌리면 끝.
    (첫 실행에서 누적 초과분이 2~6%뿐이고 기저한계가 전부를 지배했으므로,
     병목은 '운동'이 아니라 '표현'이다. 그걸 먼저 따로 잰다.)
    """
    cache = OUT / "cache"
    mol, rec_xyz, rec_el = load_complex(pdb_id, ccd, cache)
    pos = np.array(mol.GetConformer().GetPositions())
    lig_radii = np.array([RADII.get(a.GetSymbol().upper(), 1.9) for a in mol.GetAtoms()])
    center = pos.mean(0)

    n = int(2 * rmax / spacing) + 1
    ax = np.linspace(-rmax, rmax, n)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    xyz = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    dV = (ax[1] - ax[0]) ** 3

    rec_local = rec_xyz - center
    keep = np.linalg.norm(rec_local, axis=1) < rmax + 4
    rec_radii = np.array([RADII.get(e, 1.9) for e in rec_el])[keep]
    phi_P = render(xyz, rec_local[keep].astype(np.float32), rec_radii)

    chain = torsion_chain(mol)[:max_depth]  # run()과 같은 토막을 봐야 비교가 된다
    frag = np.array(chain[-1][2]) if chain else np.arange(len(pos))
    frag_pos = (pos[frag] - center).astype(np.float32)
    exact = render(xyz, frag_pos, lig_radii[frag])
    ref_norm = np.sqrt((exact.astype(np.float64) ** 2).sum() * dV)
    # 토막이 포켓 중심에서 얼마나 떨어져 있나 — 필요한 ell의 눈대중 척도
    rad = np.linalg.norm(frag_pos, axis=1)
    print(f"격자 {n}^3={len(xyz)}  토막 원자 {len(frag)}개  "
          f"중심거리 {rad.min():.1f}~{rad.max():.1f} Å")

    rows = []
    for lmax, nrad in configs:
        nb = (lmax + 1) ** 2 * nrad
        B = build_basis(xyz, lmax, nrad, rmax)
        _, Gp, rank = gram_pinv(B, dV)
        A = (B @ phi_P).astype(np.float64) * dV
        c = project(B, Gp, exact, dV)
        err = float(np.sqrt(((B.T @ c.astype(np.float32) - exact).astype(np.float64) ** 2).sum() * dV) / ref_norm)
        s_hat, s_ref = float(c @ A), float((exact.astype(np.float64) * phi_P).sum() * dV)
        rows.append(dict(lmax=lmax, nrad=nrad, n_basis=nb, rank=rank,
                         basis_err=err, score_err=abs(s_hat - s_ref), score_ref=s_ref))
        print(f"  lmax={lmax:2d} nrad={nrad:2d}  기저 {nb:5d} (랭크 {rank:5d})  "
              f"기저한계 {err*100:6.2f}%  점수 {s_hat:9.2f} vs {s_ref:9.2f} "
              f"({abs(s_hat-s_ref)/abs(s_ref)*100:5.2f}%)")
        del B, Gp
    return rows


def main():
    OUT.mkdir(exist_ok=True)

    print("=== 1부: 기저가 토막을 표현이나 하는가 (운동 없음) ===")
    sweep = basis_sweep()

    print("\n=== 2부: 경첩을 연달아 적용할 때 누적 ===")
    configs = [(6, 6), (10, 8), (14, 10), (20, 12)]
    results = {}
    for lmax, nrad in configs:
        print(f"\n--- lmax={lmax}, nrad={nrad} ---")
        try:
            r = run(lmax=lmax, nrad=nrad)
            if r:
                results[(lmax, nrad)] = r
                ex = [x["field_err"] / x["basis_err"] for x in r]
                print(f"  누적 초과분 (장오차/기저한계): {ex[0]:.3f} -> {ex[-1]:.3f}")
        except MemoryError:
            print("  메모리 부족 — 건너뜀")

    np.savez(OUT / "reexpansion_error.npz",
             sweep=np.array([(r["n_basis"], r["basis_err"], r["score_err"]) for r in sweep]),
             **{f"d_{k[0]}_{k[1]}": np.array([(r["depth"], r["field_err"], r["basis_err"], r["score_err"])
                                             for r in v]) for k, v in results.items()})

    if not results:
        print("결과 없음")
        return

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[2].loglog([r["n_basis"] for r in sweep], [r["basis_err"] for r in sweep], "s-", c="k")
    for r in sweep:
        ax[2].annotate(f"l={r['lmax']}", (r["n_basis"], r["basis_err"]), fontsize=7,
                       textcoords="offset points", xytext=(4, 4))
    ax[2].axhline(0.05, c="r", ls="--", lw=1, label="5%")
    ax[2].set(xlabel="basis size", ylabel="relative field error",
              title="basis limit vs basis size\n(no hinge motion)")
    for (lmax, nrad), rows in results.items():
        d = [r["depth"] for r in rows]
        lbl = f"lmax={lmax}, nrad={nrad}"
        ax[0].semilogy(d, [r["field_err"] for r in rows], "o-", label=lbl)
        ax[0].semilogy(d, [r["basis_err"] for r in rows], "--", alpha=0.4, lw=1)
        ax[1].semilogy(d, [max(r["score_err"], 1e-6) for r in rows], "o-", label=lbl)
    ax[0].set(xlabel="depth (hinges applied)", ylabel="relative field error",
              title="field error vs depth\n(dashed = basis limit, no accumulation)")
    # 점수는 점유 장의 겹침 적분 — kcal/mol 아님. 절대 문턱선을 긋지 않는다.
    ax[1].set(xlabel="depth (hinges applied)", ylabel="abs score error (overlap units)",
              title="score error vs depth")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "reexpansion_error.png", dpi=150)
    print(f"\n저장: {OUT / 'reexpansion_error.png'}")


if __name__ == "__main__":
    sys.exit(main())

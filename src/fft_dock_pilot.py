"""파일럿: 상관지형의 꼭대기가 결정 포즈를 찾아내는가 (강체, 형태 상보성).

설계 북극성(notes/design_north_star.md)의 "곡선"을 처음으로 진짜 계산한다.
리간드를 회전·평행이동시키며 겹침 점수를 FFT로 전부 그린 뒤, 꼭대기 포즈를
결정 포즈와 RMSD로 비교한다.

점수(형태 상보성, PIPER류):
  score(pose) = Σ lig·(리셉터 표면껍질)  −  W_clash · Σ lig·(리셉터 내부)
  둘 다 상호상관 → 평행이동 전부를 FFT 한 번에. 회전만 바깥 루프.

ponytail: 형태(점유)만. 전하·소수성 채널 없음 → 형태만으로 포즈가 안 잡히면
          그건 결론이 아니라 "채널이 더 필요하다"는 신호다 (채널은 v2).
          회전은 무작위 N개. 균일 SO(3) 격자는 커버리지 문제 생기면 업그레이드.
"""
import sys
import numpy as np
from scipy.signal import fftconvolve
from torsion_spectrum import OUT, RADII, load_complex

SIGMA = 0.8          # 리간드 blob 폭 (Å)
SKIN = 1.5           # 리셉터 표면껍질 두께 (Å)
W_CLASH = 8.0        # 내부 침투 벌점
W_POLAR = 3.0        # 극성 접촉 채널 가중치 (나중에 학습 대상)
POLAR_REACH = 3.5    # 리셉터 극성 원자 영향 반경 (Å)
SPACING = 1.0        # 격자 간격 (Å)
MARGIN = 7.0         # 리간드가 돌아다닐 여유 (Å)
POLAR = {"N", "O"}   # ponytail: 수소 없어 donor/acceptor 구분 불가 → 극성 접촉으로 뭉뚱


def rand_rotations(n, seed=0):
    """무작위 회전 N개 (균일 쿼터니언). 항등을 맨 앞에."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    R[0] = np.eye(3)
    return R


def render(shape, origin, coords, radii, sigma):
    """격자에 가우시안 점유. 원자마다 작은 창에만 더한다 (4σ 컷)."""
    f = np.zeros(shape, dtype=np.float32)
    cut = 4 * sigma
    for a, _ in zip(coords, radii):
        lo = np.floor((a - cut - origin) / SPACING).astype(int)
        hi = np.ceil((a + cut - origin) / SPACING).astype(int)
        lo = np.maximum(lo, 0); hi = np.minimum(hi, np.array(shape) - 1)
        if np.any(lo > hi):
            continue
        sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
        gx = origin[0] + np.arange(lo[0], hi[0] + 1) * SPACING
        gy = origin[1] + np.arange(lo[1], hi[1] + 1) * SPACING
        gz = origin[2] + np.arange(lo[2], hi[2] + 1) * SPACING
        d2 = ((gx - a[0])[:, None, None] ** 2 + (gy - a[1])[None, :, None] ** 2
              + (gz - a[2])[None, None, :] ** 2)
        f[sl] += np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
    return f


def ball_mask(shape, origin, coords, radii):
    """원자마다 vdW 반지름 공을 그린 합집합 (불리언)."""
    m = np.zeros(shape, dtype=bool)
    rmax = float(np.max(radii))
    for a, rad in zip(coords, radii):
        lo = np.floor((a - rad - origin) / SPACING).astype(int)
        hi = np.ceil((a + rad - origin) / SPACING).astype(int)
        lo = np.maximum(lo, 0); hi = np.minimum(hi, np.array(shape) - 1)
        if np.any(lo > hi):
            continue
        sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
        gx = origin[0] + np.arange(lo[0], hi[0] + 1) * SPACING
        gy = origin[1] + np.arange(lo[1], hi[1] + 1) * SPACING
        gz = origin[2] + np.arange(lo[2], hi[2] + 1) * SPACING
        d2 = ((gx - a[0])[:, None, None] ** 2 + (gy - a[1])[None, :, None] ** 2
              + (gz - a[2])[None, None, :] ** 2)
        m[sl] |= d2 <= rad * rad
    return m


def rec_core_skin(shape, origin, rec_xyz, rec_rad):
    """리셉터 내부(core = vdW 공)와 표면껍질(skin = 그 바깥 SKIN 두께 층)."""
    core = ball_mask(shape, origin, rec_xyz, rec_rad)
    outer = ball_mask(shape, origin, rec_xyz, rec_rad + SKIN)
    skin = outer & ~core
    return core.astype(np.float32), skin.astype(np.float32)


def correlate_shift(score_field, lig_field):
    """score와 lig의 상호상관 최대값과 그 평행이동(복셀)을 반환.

    corr(t) = Σ_x score[x] lig[x - t].  fftconvolve로 lig를 뒤집어 상관으로.
    """
    corr = fftconvolve(score_field, lig_field[::-1, ::-1, ::-1], mode="same")
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    center = np.array(corr.shape) // 2   # fftconvolve 'same' 영점 (짝·홀 모두)
    shift = (np.array(idx) - center)   # 복셀 단위, lig를 +shift 만큼 옮기면 최대
    return float(corr[idx]), shift


def correlate_sum(rec_fields, lig_fields):
    """여러 채널의 상관지형을 가중합한 뒤 최대점과 평행이동(복셀)을 반환.

    E(t) = Σ_ch w · (rec_ch ⋆ lig_ch)(t).  각 채널은 FFT 상관 하나.
    """
    total = None
    for (rec, w), lig in zip(rec_fields, lig_fields):
        c = w * fftconvolve(rec, lig[::-1, ::-1, ::-1], mode="same")
        total = c if total is None else total + c
    idx = np.unravel_index(np.argmax(total), total.shape)
    center = np.array(total.shape) // 2
    return float(total[idx]), np.array(idx) - center


def dock(pdb_id="1HSG", ccd="MK1", n_rot=300):
    mol, rec_xyz, rec_el = load_complex(pdb_id, ccd, OUT / "cache")
    lig = np.array(mol.GetConformer().GetPositions())
    lig_sym = [a.GetSymbol().upper() for a in mol.GetAtoms()]
    lig_rad = np.array([RADII.get(s, 1.9) for s in lig_sym])
    lig_polar = np.array([s in POLAR for s in lig_sym])
    rec_xyz = np.asarray(rec_xyz, float)
    rec_rad = np.array([RADII.get(e, 1.9) for e in rec_el])
    rec_polar_m = np.array([e in POLAR for e in rec_el])

    centroid = lig.mean(0)
    L = lig - centroid                              # 리간드 국소 좌표
    ext = np.abs(L).max() + MARGIN
    origin = centroid - ext
    n = int(2 * ext / SPACING) + 1
    shape = (n, n, n)

    # 리셉터 채널: 박스 근처만
    near = np.linalg.norm(rec_xyz - centroid, axis=1) < ext + 4
    core, skin = rec_core_skin(shape, origin, rec_xyz[near], rec_rad[near])
    shape_field = skin - W_CLASH * core
    # 극성 채널: 표면껍질 중 리셉터 극성원자 근처
    rp = rec_polar_m[near]
    polar_reach = ball_mask(shape, origin, rec_xyz[near][rp],
                            np.full(rp.sum(), POLAR_REACH)) if rp.any() else np.zeros(shape, bool)
    polar_skin = (skin.astype(bool) & polar_reach).astype(np.float32)
    rec_fields = [(shape_field, 1.0), (polar_skin, W_POLAR)]

    print(f"{pdb_id}/{ccd}  격자 {n}^3={n**3}  리셉터원자 {near.sum()}  "
          f"리간드원자 {len(lig)} (극성 {lig_polar.sum()})  회전 {n_rot}")

    R = rand_rotations(n_rot)
    best = (-np.inf, None, None)                     # (score, R, shift_Å)
    for i, Ri in enumerate(R):
        Lr = L @ Ri.T                                # 제자리 회전
        lig_occ = render(shape, origin, centroid + Lr, lig_rad, SIGMA)
        lig_pol = render(shape, origin, (centroid + Lr)[lig_polar], lig_rad[lig_polar], SIGMA) \
            if lig_polar.any() else np.zeros(shape, np.float32)
        s, shift = correlate_sum(rec_fields, [lig_occ, lig_pol])
        if s > best[0]:
            best = (s, Ri, shift * SPACING)

    s, Rb, tb = best
    pred = centroid + L @ Rb.T + tb                  # 예측 포즈
    rmsd = float(np.sqrt(((pred - lig) ** 2).sum(1).mean()))
    is_id = np.allclose(Rb, np.eye(3))
    print(f"  최고점 {s:9.1f}  RMSD {rmsd:6.2f} Å  "
          f"(평행이동 {np.linalg.norm(tb):.1f} Å, 회전 {'항등' if is_id else '비항등'})")
    return rmsd, s


def demo():
    """상관 평행이동 부호 자기검증: 알려진 양만큼 옮기고 되찾나."""
    rng = np.random.default_rng(1)
    shape = (24, 24, 24)
    blob = np.zeros(shape, np.float32)
    blob[12, 12, 12] = 1.0
    blob = fftconvolve(blob, np.ones((3, 3, 3), np.float32), mode="same")
    shifted = np.roll(blob, shift=(2, -3, 1), axis=(0, 1, 2))
    _, sh = correlate_shift(shifted, blob)           # blob를 +sh 옮기면 shifted에 맞음
    assert tuple(sh) == (2, -3, 1), f"상관 부호 틀림: {sh}"
    print("demo: 상관 평행이동 부호 OK")


def main():
    OUT.mkdir(exist_ok=True)
    demo()
    for pdb_id, ccd in [("1HSG", "MK1"), ("1STP", "BTN"), ("3ERT", "OHT"), ("1AQ1", "STU")]:
        try:
            dock(pdb_id, ccd)
        except Exception as e:
            print(f"  {pdb_id}/{ccd}: 실패 {e}")


if __name__ == "__main__":
    sys.exit(main())

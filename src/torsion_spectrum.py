"""torsion 하나를 0-360° 돌릴 때 리셉터-리간드 상호작용 점수의 푸리에 스펙트럼.

묻는 것: 점수의 torsion 의존성이 저차 모드 몇 개로 표현되는가?
  저차 몇 개면  -> 토러스 푸리에 (탈출구 a) 가능
  광대역이면    -> 단편 분해 (탈출구 b) 로 가야 함

점수는 Vina 상호작용 항(Trott & Olson 2010)을 직접 구현했다. Vina 바이너리와
pdbqt 파이프라인을 피하려는 것이고, 묻는 것이 절대 점수가 아니라 각도 의존성의
주파수 구조이므로 이 정도로 충분하다.

ponytail: 수소 없이 N/O를 donor/acceptor 양쪽으로 취급 -> hbond 항이 과대추정된다.
          절대 점수를 인용할 거면 PROPKA/PDB2PQR로 양성자화를 먼저 정해야 한다.
"""

import io
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms as rmt

RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).resolve().parent.parent / "out"

# --- Vina 파라미터 -------------------------------------------------------
# X-S 반지름 (Å)
RADII = {
    "C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1,
    "F": 1.5, "CL": 1.8, "BR": 2.0, "I": 2.2,
    "MG": 1.2, "MN": 1.2, "ZN": 1.2, "CA": 1.2, "FE": 1.2, "NA": 1.2, "K": 1.2,
}
W_GAUSS1, W_GAUSS2, W_REPULSION = -0.0356, -0.00516, 0.840
W_HYDROPHOBIC, W_HBOND = -0.0351, -0.587
CUTOFF = 8.0
HALOGENS = {"F", "CL", "BR", "I"}


def fetch(url: str, dest: Path) -> Path:
    """없으면 받아서 캐시한다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


# --- 원자 타이핑 ---------------------------------------------------------
def bonded_elements(coords: np.ndarray, elements: list[str]) -> list[set[str]]:
    """거리로 결합을 추정해 각 원자의 이웃 원소 집합을 돌려준다.

    단백질에 RDKit sanitize를 돌리지 않으려는 것. 중원자 결합은 1.95 Å 안쪽이다.
    """
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adj = (d < 1.95) & (d > 0.1)
    return [set(np.array(elements)[adj[i]]) for i in range(len(elements))]


def type_atoms(coords: np.ndarray, elements: list[str]):
    """(반지름, 소수성 플래그, hbond 플래그)."""
    nbrs = bonded_elements(coords, elements)
    radii = np.array([RADII.get(e, 1.9) for e in elements])
    hydrophobic = np.array([
        (e == "C" or e in HALOGENS) and not (nbr & {"N", "O"})
        for e, nbr in zip(elements, nbrs)
    ])
    hbond = np.array([e in ("N", "O") for e in elements])
    return radii, hydrophobic, hbond


# --- 점수 ----------------------------------------------------------------
def vina_inter(lig_xyz, lig_t, rec_xyz, rec_t, include_repulsion=True):
    """상호작용 점수. lig_t/rec_t = type_atoms() 출력."""
    lr, lh, lb = lig_t
    rr, rh, rb = rec_t
    r = np.linalg.norm(lig_xyz[:, None, :] - rec_xyz[None, :, :], axis=-1)
    mask = r < CUTOFF
    if not mask.any():
        return 0.0
    d = r - (lr[:, None] + rr[None, :])
    d = d[mask]

    e = W_GAUSS1 * np.exp(-((d / 0.5) ** 2)) + W_GAUSS2 * np.exp(-(((d - 3.0) / 2.0) ** 2))
    if include_repulsion:
        e = e + W_REPULSION * np.where(d < 0, d * d, 0.0)
    e = e + W_HYDROPHOBIC * np.clip(1.5 - d, 0, 1) * (lh[:, None] & rh[None, :])[mask]
    e = e + W_HBOND * np.clip(-d / 0.7, 0, 1) * (lb[:, None] & rb[None, :])[mask]
    return float(e.sum())


def self_clash(mol, xyz, path_min=4):
    """4결합 이상 떨어진 원자쌍의 최소 거리. 분자 내부 충돌 감지용."""
    dm = Chem.GetDistanceMatrix(mol)
    far = dm >= path_min
    if not far.any():
        return np.inf
    r = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return float(r[far].min())


# --- 구조 읽기 -----------------------------------------------------------
def load_complex(pdb_id: str, ccd: str, cache: Path):
    """(리간드 mol, 리셉터 좌표, 리셉터 원소) 반환."""
    pdb = fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb",
                cache / f"{pdb_id}.pdb").read_text()

    lig_lines, rec_xyz, rec_el = [], [], []
    seen_lig_key = None
    for line in pdb.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resn = line[17:20].strip()
        altloc = line[16]
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if element == "H" or element == "D":
            continue
        if altloc not in (" ", "A"):
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if resn == ccd:
            key = (line[21], line[22:27])  # chain + resseq: 사본이 여러 개일 수 있다
            if seen_lig_key is None:
                seen_lig_key = key
            if key == seen_lig_key:
                lig_lines.append(line)
            continue
        if resn in ("HOH", "DOD"):
            continue
        rec_xyz.append(xyz)
        rec_el.append(element)

    if not lig_lines:
        raise RuntimeError(f"{pdb_id}: 리간드 {ccd} 없음")

    block = "\n".join(lig_lines) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True, proximityBonding=True)
    if mol is None:
        raise RuntimeError(f"{pdb_id}: 리간드 PDB 파싱 실패")

    # 결합 차수를 RCSB ideal SDF에서 가져온다 (근접 결합은 전부 단일로 잡힌다)
    sdf = fetch(f"https://files.rcsb.org/ligands/download/{ccd}_ideal.sdf",
                cache / f"{ccd}_ideal.sdf").read_bytes()
    tmpl = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf), removeHs=True), None)
    if tmpl is not None:
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(tmpl, mol)
        except Exception as exc:  # 원자 수/연결 불일치
            print(f"  ! {pdb_id}/{ccd}: 결합차수 이식 실패 ({exc}); 단일결합으로 진행")
    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
    return mol, np.array(rec_xyz), rec_el


ROT_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def torsion_quadruples(mol):
    """회전가능 결합마다 (i, j, k, l). j-k가 회전축."""
    out = []
    for j, k in mol.GetSubstructMatches(ROT_SMARTS):
        i = next((a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors()
                  if a.GetIdx() != k), None)
        l = next((a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors()
                  if a.GetIdx() != j), None)
        if i is not None and l is not None:
            out.append((i, j, k, l))
    return out


# --- 스윕 ---------------------------------------------------------------
def sweep(mol, quad, rec_xyz, rec_t, step_deg=1.0):
    """torsion 한 바퀴. (각도, 전체점수, 반발제외점수, 자기충돌거리, 최대원자이동)."""
    mol = Chem.Mol(mol)  # 원본 보존
    conf = mol.GetConformer()
    lig_el = [a.GetSymbol().upper() for a in mol.GetAtoms()]
    lig_t = type_atoms(np.array(conf.GetPositions()), lig_el)  # 타입은 회전에 불변

    angles = np.arange(0.0, 360.0, step_deg)
    e_full, e_norep, clash = [], [], []
    ref = None
    max_shift = 0.0
    for a in angles:
        rmt.SetDihedralDeg(conf, *quad, float(a))
        xyz = np.array(conf.GetPositions())
        if ref is None:
            ref = xyz.copy()
        max_shift = max(max_shift, float(np.linalg.norm(xyz - ref, axis=1).max()))
        e_full.append(vina_inter(xyz, lig_t, rec_xyz, rec_t, True))
        e_norep.append(vina_inter(xyz, lig_t, rec_xyz, rec_t, False))
        clash.append(self_clash(mol, xyz))
    return angles, np.array(e_full), np.array(e_norep), np.array(clash), max_shift


def modes_for(signal: np.ndarray, frac: float) -> int:
    """DC를 뺀 파워의 frac을 담는 데 필요한 최저 푸리에 모드 수.

    주의: 분산 비율 기준이다. 충돌 봉우리가 분산을 독점하면 낙관적으로 나온다.
    판정에는 modes_for_abs를 쓸 것.
    """
    p = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    if p.sum() <= 0:
        return 0
    c = np.cumsum(p) / p.sum()
    return int(np.searchsorted(c, frac) + 1)


def truncate(signal: np.ndarray, k: int) -> np.ndarray:
    """모드 k까지만 남긴 재구성."""
    f = np.fft.rfft(signal)
    f[k + 1:] = 0
    return np.fft.irfft(f, n=len(signal))


def modes_for_abs(signal: np.ndarray, tol: float, mask: np.ndarray, kmax=60) -> int:
    """mask 영역에서 최대 절대오차가 tol 미만이 되는 최저 모드 수. 못하면 -1.

    이것이 실제 판정 기준이다. 도킹은 점수가 좋은 영역에서만 정확해야 하고
    (충돌 영역은 '나쁘다'는 것만 알면 된다), 필요한 정확도는 절대값이다.
    """
    if not mask.any():
        return -1
    for k in range(1, kmax + 1):
        if np.abs(truncate(signal, k) - signal)[mask].max() < tol:
            return k
    return -1


TARGETS = [("1HSG", "MK1"), ("1STP", "BTN"), ("3ERT", "OHT"), ("1AQ1", "STU")]


def main(targets=TARGETS):
    OUT.mkdir(exist_ok=True)
    cache = OUT / "cache"
    rows = []
    for pdb_id, ccd in targets:
        try:
            mol, rec_xyz, rec_el = load_complex(pdb_id, ccd, cache)
            rec_t = type_atoms(rec_xyz, rec_el)
            quads = torsion_quadruples(mol)
            print(f"{pdb_id}/{ccd}: 리간드 중원자 {mol.GetNumAtoms()}, "
                  f"리셉터 중원자 {len(rec_xyz)}, 회전결합 {len(quads)}")
            for q in quads:
                try:
                    ang, ef, en, cl, shift = sweep(mol, q, rec_xyz, rec_t)
                except Exception as exc:
                    print(f"  ! torsion {q} 건너뜀: {exc}")
                    continue
                # 물리적으로 접근 가능한 영역: 최저점 +5 이내, 자기충돌 없음
                acc = (ef <= ef.min() + 5.0) & (cl >= 2.5)
                rows.append(dict(
                    pdb=pdb_id, ccd=ccd, quad=q, ang=ang, e_full=ef, e_norep=en,
                    clash=cl, shift=shift, acc=acc,
                    k95=modes_for(ef, 0.95),
                    k05=modes_for_abs(ef, 0.5, acc),
                    k10=modes_for_abs(ef, 1.0, acc),
                    span=float(ef.max() - ef.min()),
                    acc_frac=float(acc.mean()),
                ))
                r = rows[-1]
                print(f"  torsion {q}: 진폭 {r['span']:7.2f}  이동 {shift:5.2f} Å  "
                      f"접근가능 {r['acc_frac']*100:4.0f}%  "
                      f"k95 {r['k95']:3d} | 오차<1.0 {r['k10']:3d}  "
                      f"오차<0.5 {r['k05']:3d}")
        except Exception as exc:
            print(f"{pdb_id}/{ccd}: 실패 - {exc}")

    if not rows:
        print("결과 없음")
        return

    def stat(key):
        v = np.array([r[key] for r in rows])
        ok = v[v > 0]
        fail = int((v < 0).sum())
        if len(ok) == 0:
            return f"전부 {60}모드로 불가 ({fail}/{len(v)})"
        return (f"중앙값 {np.median(ok):.0f}  사분위 {np.percentile(ok,25):.0f}-"
                f"{np.percentile(ok,75):.0f}  최대 {ok.max()}" +
                (f"  [60모드로 불가 {fail}/{len(v)}]" if fail else ""))

    span = np.array([r["span"] for r in rows])
    print(f"\n=== torsion {len(rows)}개 ===")
    print(f"점수 진폭          중앙값 {np.median(span):.1f}  최대 {span.max():.1f}")
    print(f"k95  (분산 95%)    {stat('k95')}   <- 낙관적, 판정에 쓰지 말 것")
    print(f"모드 (오차<1.0)    {stat('k10')}")
    print(f"모드 (오차<0.5)    {stat('k05')}   <- 판정 기준")
    print("\n판정: '오차<0.5' 중앙값이 ~6 이하면 토러스 푸리에(a), "
          "수십이거나 불가가 많으면 단편 분해(b).")

    np.savez_compressed(OUT / "torsion_sweeps.npz",
                        **{f"{r['pdb']}_{r['ccd']}_{'_'.join(map(str,r['quad']))}":
                           np.vstack([r["ang"], r["e_full"], r["e_norep"], r["clash"]])
                           for r in rows})
    plot(rows)
    print(f"\n저장: {OUT}")


def plot(rows):
    """패널 3개: 대표 스윕, 모드 수에 따른 절대오차, 판정 히스토그램."""
    worst = max(rows, key=lambda r: r["span"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    a0 = ax[0]
    a0.plot(worst["ang"], worst["e_full"], lw=1.2, c="C0", label="score")
    for k in (4, 8, 16):
        a0.plot(worst["ang"], truncate(worst["e_full"], k), lw=0.9, ls="--",
                label=f"{k} modes")
    acc = worst["acc"]
    if acc.any():
        a0.scatter(worst["ang"][acc], worst["e_full"][acc], s=6, c="green",
                   zorder=3, label="accessible")
    a0.set(xlabel="torsion angle (deg)", ylabel="interaction score",
           title=f"{worst['pdb']}/{worst['ccd']} "
                 f"bond {worst['quad'][1]}-{worst['quad'][2]}")
    a0.legend(fontsize=7)

    # 접근 가능 영역에서의 최대 절대오차 vs 모드 수
    ks = np.arange(1, 41)
    for r in rows:
        if not r["acc"].any():
            continue
        err = [np.abs(truncate(r["e_full"], k) - r["e_full"])[r["acc"]].max() for k in ks]
        ax[1].semilogy(ks, err, lw=0.7, alpha=0.45, c="C0")
    ax[1].axhline(0.5, c="r", ls="--", lw=1, label="0.5 kcal/mol")
    ax[1].set(xlabel="Fourier modes kept", ylabel="max abs error (accessible region)",
              title="truncation error per torsion")
    ax[1].legend(fontsize=8)

    k05 = np.array([r["k05"] for r in rows])
    k95 = np.array([r["k95"] for r in rows])
    hi = max(k05.max(), k95.max(), 10) + 3
    ax[2].hist([k95, np.where(k05 > 0, k05, hi - 1)],
               bins=np.arange(0, hi + 2, 2),
               label=["variance 95% (optimistic)", "abs err < 0.5"])
    ax[2].axvline(6, c="k", ls="--", lw=1)
    ax[2].set(xlabel="modes required", ylabel="torsions",
              title="left of dashed line = torus Fourier viable")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "torsion_spectrum.png", dpi=150)


if __name__ == "__main__":
    main([(a, b) for a, b in (t.split("/") for t in sys.argv[1:])] or TARGETS)

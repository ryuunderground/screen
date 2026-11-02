"""v6 training: same cluster-aware CV + frozen held-out eval as v1, feature-name agnostic.

Reads feature_names from the shards. Adds a feature correlation-matrix figure and VIF
table so we can SEE whether the new channels de-collinearized vs v1 (where all 5
channels collapsed onto the buried-volume axis). Labels = experimental pKd; splits from
LP-PDBBind; GroupKFold on seq cluster (no same-protein leakage).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1F3864"


def merge_shards(shards_dir):
    ids, X, names = [], [], None
    for f in sorted(Path(shards_dir).glob("feat_*.npz")):
        d = np.load(f, allow_pickle=True)
        if names is None:
            names = [str(x) for x in d["feature_names"]]
        if len(d["ids"]):
            ids.extend([str(x) for x in d["ids"]]); X.append(d["X"])
    X = np.vstack(X) if X else np.zeros((0, len(names or [])))
    return ids, X, names


def metrics(y, p):
    r = pearsonr(y, p)[0] if len(y) > 2 else np.nan
    rho = spearmanr(y, p)[0] if len(y) > 2 else np.nan
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return r, rho, rmse


def vif(X):
    """Variance inflation factor per column (std features). High => collinear."""
    out = []
    Xs = StandardScaler().fit_transform(X)
    for j in range(Xs.shape[1]):
        y = Xs[:, j]
        Z = np.delete(Xs, j, axis=1)
        # R^2 of regressing feature j on the rest
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - np.sum((y - Z @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        out.append(1.0 / max(1e-9, 1 - r2))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    ids, X, NAMES = merge_shards(a.shards_dir)
    feat = pd.DataFrame(X, columns=NAMES); feat["id"] = ids
    man = pd.read_csv(a.manifest)
    df = man.merge(feat, on="id", how="inner").dropna(subset=NAMES)
    np.savez(out / "v6_features.npz", ids=df["id"].values,
             X=df[NAMES].values, feature_names=np.array(NAMES))
    print(f"merged: {len(ids)}; joined w/ labels: {len(df)}")
    print(df["split"].value_counts().to_string())

    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    Xtr, ytr, gtr = tr[NAMES].values, tr["pKd"].values, tr["seq"].values

    # collinearity diagnostics on train features
    corr = np.corrcoef(StandardScaler().fit_transform(Xtr), rowvar=False)
    vifs = vif(Xtr)
    pd.DataFrame({"channel": NAMES, "VIF": vifs}).to_csv(out / "v6_vif.csv", index=False)
    print("\nVIF (>>5 = collinear):")
    print(pd.DataFrame({"channel": NAMES, "VIF": vifs}).to_string(index=False))

    gkf = GroupKFold(n_splits=5)
    oof_r = np.full(len(tr), np.nan); oof_g = np.full(len(tr), np.nan)
    fold_rows, coefs = [], []
    for k, (itr, ite) in enumerate(gkf.split(Xtr, ytr, gtr)):
        sc = StandardScaler().fit(Xtr[itr])
        rm = Ridge(alpha=1.0).fit(sc.transform(Xtr[itr]), ytr[itr])
        gm = GradientBoostingRegressor(random_state=0).fit(Xtr[itr], ytr[itr])
        oof_r[ite] = rm.predict(sc.transform(Xtr[ite]))
        oof_g[ite] = gm.predict(Xtr[ite])
        coefs.append(rm.coef_)
        rr = metrics(ytr[ite], oof_r[ite]); rg = metrics(ytr[ite], oof_g[ite])
        fold_rows.append(dict(fold=k, n=len(ite), ridge_r=rr[0], ridge_rmse=rr[2],
                              gbr_r=rg[0], gbr_rmse=rg[2]))
    folds = pd.DataFrame(fold_rows); folds.to_csv(out / "v6_cv_folds.csv", index=False)
    cv_r = metrics(ytr, oof_r); cv_g = metrics(ytr, oof_g)
    print("\nCV (cluster-aware GroupKFold on seq):")
    print(folds.to_string(index=False))
    print(f"Ridge OOF: r={cv_r[0]:.3f} rho={cv_r[1]:.3f} rmse={cv_r[2]:.3f}")
    print(f"GBR   OOF: r={cv_g[0]:.3f} rho={cv_g[1]:.3f} rmse={cv_g[2]:.3f}")

    sc = StandardScaler().fit(Xtr)
    ridge_full = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    gbr_full = GradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    results = {}
    for name, sub in [("val", va), ("test", te)]:
        if len(sub) == 0:
            continue
        Xs, ys = sub[NAMES].values, sub["pKd"].values
        pr = ridge_full.predict(sc.transform(Xs)); pg = gbr_full.predict(Xs)
        results[name] = dict(ridge=metrics(ys, pr), gbr=metrics(ys, pg), y=ys, pr=pr, pg=pg)
        print(f"{name}: Ridge r={results[name]['ridge'][0]:.3f} rmse={results[name]['ridge'][2]:.3f}"
              f" | GBR r={results[name]['gbr'][0]:.3f} rmse={results[name]['gbr'][2]:.3f}")

    full_w = ridge_full.coef_
    pd.DataFrame({"channel": NAMES, "cv_mean_coef": np.mean(coefs, axis=0),
                  "full_train_coef": full_w}).to_csv(out / "v6_channel_weights.csv", index=False)

    # ---- figures ----
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444"})

    # correlation matrix (the de-collinearization check)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(NAMES)
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "#222", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("v6 channel correlation (train)")
    fig.tight_layout(); fig.savefig(out / "fig_v6_corr.png", dpi=150); plt.close(fig)

    # CV + test scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panels = [("CV out-of-fold (train, Ridge)", ytr, oof_r, cv_r),
              ("Held-out test (frozen Ridge)",
               results.get("test", {}).get("y", np.array([])),
               results.get("test", {}).get("pr", np.array([])),
               results.get("test", {}).get("ridge", (np.nan,) * 3))]
    for ax, (ttl, y, p, m) in zip(axes, panels):
        if len(y) == 0:
            continue
        ax.scatter(y, p, s=6, alpha=0.3, color=NAVY, edgecolors="none")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#C00", lw=1)
        ax.set_xlabel("experimental pKd"); ax.set_ylabel("predicted pKd")
        ax.set_title(f"{ttl}\nr={m[0]:.2f}  rho={m[1]:.2f}  RMSE={m[2]:.2f}  (n={len(y)})")
    fig.tight_layout(); fig.savefig(out / "fig_v6_scatter.png", dpi=150); plt.close(fig)

    # channel weights
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(NAMES, full_w, color=NAVY); ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Ridge coefficient (standardized)")
    ax.set_title("v6 channel weights (frozen full-train Ridge)")
    fig.tight_layout(); fig.savefig(out / "fig_v6_channel_weights.png", dpi=150); plt.close(fig)

    summ = {"feature_names": NAMES, "n_total": len(df), "n_train": len(tr),
            "n_val": len(va), "n_test": len(te),
            "cv_ridge": cv_r, "cv_gbr": cv_g,
            "test_ridge": results.get("test", {}).get("ridge"),
            "test_gbr": results.get("test", {}).get("gbr"),
            "val_ridge": results.get("val", {}).get("ridge"),
            "vif": dict(zip(NAMES, vifs.tolist())),
            "channel_weights_full": dict(zip(NAMES, full_w.tolist()))}
    (out / "v6_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print("\nwrote figures + v6_summary.json to", out)


if __name__ == "__main__":
    main()

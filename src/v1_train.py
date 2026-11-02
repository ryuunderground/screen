"""v1 training: cluster-aware CV + frozen test eval + figures.

Merges shard npz -> features, joins manifest (pKd, split, seq).
CV: 5-fold GroupKFold on the LP 'train' split, groups = protein seq (cluster-aware,
no same-protein leakage across folds). Models: Ridge (primary; standardized weights =
channel importances) and GradientBoosting (comparison). Then refit on full train,
FREEZE, evaluate on LP 'test' (and 'val'). Metrics: Pearson r, Spearman rho, RMSE.
Figures (navy #1F3864): pred-vs-true scatter (CV + test), channel-weight bar, per-fold table.
"""
import argparse
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
FEATURE_NAMES = ["contact", "clash", "polar", "hydrophob", "elec"]


def merge_shards(shards_dir):
    ids, X = [], []
    for f in sorted(Path(shards_dir).glob("feat_*.npz")):
        d = np.load(f, allow_pickle=True)
        if len(d["ids"]):
            ids.extend([str(x) for x in d["ids"]]); X.append(d["X"])
    X = np.vstack(X) if X else np.zeros((0, len(FEATURE_NAMES)))
    return ids, X


def metrics(y, p):
    r = pearsonr(y, p)[0] if len(y) > 2 else np.nan
    rho = spearmanr(y, p)[0] if len(y) > 2 else np.nan
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return r, rho, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    ids, X = merge_shards(a.shards_dir)
    feat = pd.DataFrame(X, columns=FEATURE_NAMES); feat["id"] = ids
    man = pd.read_csv(a.manifest)
    df = man.merge(feat, on="id", how="inner").dropna(subset=FEATURE_NAMES)
    np.savez(out / "v1_features.npz", ids=df["id"].values,
             X=df[FEATURE_NAMES].values, feature_names=np.array(FEATURE_NAMES))
    print(f"merged features: {len(ids)}; joined with labels: {len(df)}")
    print(df["split"].value_counts().to_string())

    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    Xtr, ytr, gtr = tr[FEATURE_NAMES].values, tr["pKd"].values, tr["seq"].values

    # --- cluster-aware 5-fold CV on train ---
    gkf = GroupKFold(n_splits=5)
    oof_ridge = np.full(len(tr), np.nan)
    oof_gbr = np.full(len(tr), np.nan)
    fold_rows = []
    coefs = []
    for k, (itr, ite) in enumerate(gkf.split(Xtr, ytr, gtr)):
        sc = StandardScaler().fit(Xtr[itr])
        Ridge_m = Ridge(alpha=1.0).fit(sc.transform(Xtr[itr]), ytr[itr])
        gbr = GradientBoostingRegressor(random_state=0).fit(Xtr[itr], ytr[itr])
        pr = Ridge_m.predict(sc.transform(Xtr[ite]))
        pg = gbr.predict(Xtr[ite])
        oof_ridge[ite] = pr; oof_gbr[ite] = pg
        coefs.append(Ridge_m.coef_)
        rr = metrics(ytr[ite], pr); rg = metrics(ytr[ite], pg)
        fold_rows.append(dict(fold=k, n=len(ite),
                              ridge_r=rr[0], ridge_rho=rr[1], ridge_rmse=rr[2],
                              gbr_r=rg[0], gbr_rho=rg[1], gbr_rmse=rg[2]))
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "v1_cv_folds.csv", index=False)
    cv_r = metrics(ytr, oof_ridge)
    cv_g = metrics(ytr, oof_gbr)
    print("\nCV (out-of-fold, cluster-aware GroupKFold on seq):")
    print(folds.to_string(index=False))
    print(f"Ridge OOF: r={cv_r[0]:.3f} rho={cv_r[1]:.3f} rmse={cv_r[2]:.3f}")
    print(f"GBR   OOF: r={cv_g[0]:.3f} rho={cv_g[1]:.3f} rmse={cv_g[2]:.3f}")

    # --- refit on full train, freeze, eval held-out ---
    sc = StandardScaler().fit(Xtr)
    ridge_full = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    gbr_full = GradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    results = {}
    for name, sub in [("val", va), ("test", te)]:
        if len(sub) == 0:
            continue
        Xs, ys = sub[FEATURE_NAMES].values, sub["pKd"].values
        pr = ridge_full.predict(sc.transform(Xs))
        pg = gbr_full.predict(Xs)
        results[name] = dict(ridge=metrics(ys, pr), gbr=metrics(ys, pg),
                             y=ys, pr=pr)
        print(f"{name}: Ridge r={results[name]['ridge'][0]:.3f} "
              f"rho={results[name]['ridge'][1]:.3f} rmse={results[name]['ridge'][2]:.3f} | "
              f"GBR r={results[name]['gbr'][0]:.3f} rmse={results[name]['gbr'][2]:.3f}")

    channel_w = np.mean(coefs, axis=0)
    full_w = ridge_full.coef_
    pd.DataFrame({"channel": FEATURE_NAMES, "cv_mean_coef": channel_w,
                  "full_train_coef": full_w}).to_csv(out / "v1_channel_weights.csv", index=False)

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444"})

    # 1. CV + test scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (ttl, y, p, m) in zip(axes, [
            ("CV out-of-fold (train, Ridge)", ytr, oof_ridge, cv_r),
            ("Held-out test (frozen Ridge)",
             results.get("test", {}).get("y", np.array([])),
             results.get("test", {}).get("pr", np.array([])),
             results.get("test", {}).get("ridge", (np.nan, np.nan, np.nan)))]):
        if len(y) == 0:
            continue
        ax.scatter(y, p, s=6, alpha=0.3, color=NAVY, edgecolors="none")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#C00", lw=1)
        ax.set_xlabel("experimental pKd"); ax.set_ylabel("predicted pKd")
        ax.set_title(f"{ttl}\nr={m[0]:.2f}  rho={m[1]:.2f}  RMSE={m[2]:.2f}  (n={len(y)})")
    fig.tight_layout(); fig.savefig(out / "fig_v1_scatter.png", dpi=150); plt.close(fig)

    # 2. channel weights
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(FEATURE_NAMES, full_w, color=NAVY)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Ridge coefficient (standardized features)")
    ax.set_title("v1 channel weights (frozen full-train Ridge)")
    fig.tight_layout(); fig.savefig(out / "fig_v1_channel_weights.png", dpi=150); plt.close(fig)

    # 3. per-fold metric summary
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35; x = np.arange(len(folds))
    ax.bar(x - w / 2, folds.ridge_r, w, label="Ridge r", color=NAVY)
    ax.bar(x + w / 2, folds.gbr_r, w, label="GBR r", color="#7F9BC6")
    ax.set_xticks(x); ax.set_xticklabels([f"fold {i}" for i in folds.fold])
    ax.set_ylabel("Pearson r"); ax.set_title("Per-fold CV Pearson r"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_v1_folds.png", dpi=150); plt.close(fig)

    # save summary json-ish
    summ = {"n_total": len(df), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "cv_ridge": cv_r, "cv_gbr": cv_g,
            "test_ridge": results.get("test", {}).get("ridge"),
            "test_gbr": results.get("test", {}).get("gbr"),
            "val_ridge": results.get("val", {}).get("ridge"),
            "channel_weights_full": dict(zip(FEATURE_NAMES, full_w.tolist()))}
    import json
    (out / "v1_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print("\nwrote figures + v1_summary.json to", out)


if __name__ == "__main__":
    main()

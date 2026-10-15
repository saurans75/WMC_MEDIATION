"""
Reproducible analysis pipeline for:
"Testing Moderation Against Mediation in the Working Memory Capacity and
Anxiety Relationship: A Multimodal Psychophysiological Investigation"

Two-phase design: Phase 1 = dual n-back (DnB), Phase 2 = AVT (4 levels).

Input file:
    Master_Sheet_v4_grouped.xlsx   (sheet "Master_Data", 96 participants,
                                     one row per participant, all tasks
                                     and all eye-tracking / EDA metrics
                                     as separate columns)

Requires: pandas, numpy, scipy, statsmodels, patsy, bambi, pymc, arviz
    pip install pandas numpy scipy statsmodels patsy bambi pymc arviz

Produces: results_final.xlsx, containing every table referenced in the
manuscript and its supplementary materials.

Notes on naming convention (see Section 3.1 / Limitations of the paper):
the two-factor EDA solution is labelled by factor POSITION, not by which
raw indicator happens to load highest in a given re-run: Factor 1 is
always "EDA Phasic Magnitude" and Factor 2 is always "EDA Tonic Arousal".
An earlier draft of the analysis accidentally reported one Anxiety x WMC
interaction under the wrong one of these two labels; this script uses the
fixed, position-based convention throughout so that doesn't happen again.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest, chi2 as chi2dist, kurtosis, norm, shapiro, skew
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

SRC = "Master_Sheet_v4_grouped.xlsx"

# ================================================================
# PART 1: LOAD DATA, BUILD CONTINUOUS PREDICTORS
# ================================================================
df = pd.read_excel(SRC, sheet_name="Master_Data")

df["Anxiety_z"] = (df["TraitanxietyScores"] - df["TraitanxietyScores"].mean()) / df["TraitanxietyScores"].std()
df["WMC_z"] = (df["WMC_dprime_avg"] - df["WMC_dprime_avg"].mean()) / df["WMC_dprime_avg"].std()
df["DnB_perf_z"] = (df["DnB_performance"] - df["DnB_performance"].mean()) / df["DnB_performance"].std()

# ================================================================
# PART 2: RESHAPE TO LONG FORMAT (DnB + AVT levels 1-4)
# ================================================================
eye_suffixes = ["TTFF", "FFD", "TFD", "NOF", "AFD", "SSD", "SSV", "MSD", "MSA", "MSV", "TSD", "PSV", "MPD", "SPD"]
eda_suffixes = ["EDA_Responses", "Mean_EDA_Amplitude", "EDA_Response_Proportion", "Tonic_SCL_Mean", "Phasic_AUC"]
task_prefixes = {"AVT_L1": "AVTL1", "AVT_L2": "AVTL2", "AVT_L3": "AVTL3", "AVT_L4": "AVTL4"}

cols = df.columns.tolist()


def find_col(prefix, suffix):
    """Match a column by prefix + suffix, tolerant of the single/double
    underscore inconsistency between the DnB and AVT column sets."""
    candidates = [c for c in cols if c.startswith(prefix) and c[len(prefix):].lstrip("_") == suffix]
    if not candidates:
        raise KeyError(f"no column found for prefix={prefix!r}, suffix={suffix!r}")
    return sorted(candidates, key=len)[0]


long_rows = []
for task, prefix in task_prefixes.items():
    sub = pd.DataFrame(
        {
            "Participant_ID": df["Participant_ID"].values,
            "Anxiety_z": df["Anxiety_z"].values,
            "WMC_z": df["WMC_z"].values,
            "task": task,
        }
    )
    for suf in eye_suffixes:
        sub["eye_" + suf] = df[find_col(prefix, suf)].values
    for suf in eda_suffixes:
        sub["eda_" + suf] = df[find_col(prefix, suf)].values
    sub["TCT"] = df[find_col(prefix, "TCT")].values
    long_rows.append(sub)

dnb_row = pd.DataFrame(
    {
        "Participant_ID": df["Participant_ID"].values,
        "Anxiety_z": df["Anxiety_z"].values,
        "WMC_z": df["WMC_z"].values,
        "task": "DnB",
        "DnB_perf_z": df["DnB_perf_z"].values,
    }
)
for suf in eye_suffixes:
    dnb_row["eye_" + suf] = df[find_col("DnB", suf)].values
for suf in eda_suffixes:
    dnb_row["eda_" + suf] = df[find_col("DnB", suf)].values
dnb_row["TCT"] = np.nan

long = pd.concat([dnb_row] + long_rows, ignore_index=True)

# ================================================================
# PART 3: EFA DIMENSION REDUCTION
# (TFD excluded from the eye set: redundant with NOF x AFD, r = .97)
# ================================================================


def kmo_bartlett(X):
    X = np.asarray(X)
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    dof = p * (p - 1) / 2
    pval = 1 - chi2dist.cdf(chi2, dof)
    Rinv = np.linalg.pinv(R)
    D = np.diag(1 / np.sqrt(np.diag(Rinv)))
    Q = D @ Rinv @ D
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(R, 0)
    kmo = np.sum(R ** 2) / (np.sum(R ** 2) + np.sum(Q ** 2))
    return chi2, dof, pval, kmo


def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
        R = u @ vt
        d_new = np.sum(s)
        if d_new < d * (1 + tol) and i > 1:
            break
        d = d_new
    return Phi @ R, R


def pca_efa(X, n_factors, colnames):
    X = np.asarray(X, dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0, ddof=1)
    Z = (X - mean) / std
    R = np.corrcoef(Z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order][:n_factors], eigvecs[:, order][:, :n_factors]
    loadings = eigvecs * np.sqrt(eigvals)
    rot_loadings, _ = varimax(loadings)
    for j in range(n_factors):
        idx = np.argmax(np.abs(rot_loadings[:, j]))
        if rot_loadings[idx, j] < 0:
            rot_loadings[:, j] *= -1
    communalities = np.sum(rot_loadings ** 2, axis=1)
    Rinv = np.linalg.pinv(R)
    W = Rinv @ rot_loadings
    return {
        "loadings": pd.DataFrame(rot_loadings, index=colnames, columns=[f"Factor{i + 1}" for i in range(n_factors)]),
        "communalities": pd.Series(communalities, index=colnames),
        "weights": W,
        "mean": mean,
        "std": std,
    }


def apply_scores(fit, X_new):
    Z = (np.asarray(X_new, dtype=float) - fit["mean"]) / fit["std"]
    return pd.DataFrame(Z @ fit["weights"], columns=fit["loadings"].columns)


eye_cols_efa = ["eye_TTFF", "eye_NOF", "eye_SSD", "eye_SSV", "eye_MSD", "eye_MSA", "eye_MSV", "eye_TSD", "eye_PSV"]
eda_cols = [c for c in long.columns if c.startswith("eda_")]

eye_fit = pca_efa(long[eye_cols_efa].values, 3, [c.replace("eye_", "") for c in eye_cols_efa])
eda_fit = pca_efa(long[eda_cols].values, 2, [c.replace("eda_", "") for c in eda_cols])
eye_scores = apply_scores(eye_fit, long[eye_cols_efa].values)
eda_scores = apply_scores(eda_fit, long[eda_cols].values)

# Factor -> construct labels are fixed by factor position (see module
# docstring), not re-derived from which raw item loads highest each time
# this is run.
long["Eye_Scanning"] = eye_scores["Factor1"].values        # TTFF, NOF, TSD
long["Eye_SaccKinematic"] = eye_scores["Factor2"].values    # MSD, MSA, MSV
long["Eye_SaccPeakSkew"] = eye_scores["Factor3"].values     # SSV, PSV
long["EDA_PhasicMag"] = eda_scores["Factor1"].values
long["EDA_TonicArousal"] = eda_scores["Factor2"].values
long["Eye_AFD_z"] = (long["eye_AFD"] - long["eye_AFD"].mean()) / long["eye_AFD"].std()
long["Pupil_MPD_z"] = (long["eye_MPD"] - long["eye_MPD"].mean()) / long["eye_MPD"].std()
long["Pupil_SPD_z"] = (long["eye_SPD"] - long["eye_SPD"].mean()) / long["eye_SPD"].std()

DVs = [
    "Eye_Scanning", "Eye_SaccKinematic", "Eye_SaccPeakSkew", "Eye_AFD_z",
    "Pupil_MPD_z", "Pupil_SPD_z", "EDA_PhasicMag", "EDA_TonicArousal",
]

kmo_eye = kmo_bartlett(long[eye_cols_efa])
kmo_eda = kmo_bartlett(long[eda_cols])
print(f"Eye KMO/Bartlett:  chi2={kmo_eye[0]:.1f}, df={kmo_eye[1]:.0f}, p={kmo_eye[2]:.4g}, KMO={kmo_eye[3]:.3f}")
print(f"EDA KMO/Bartlett:  chi2={kmo_eda[0]:.1f}, df={kmo_eda[1]:.0f}, p={kmo_eda[2]:.4g}, KMO={kmo_eda[3]:.3f}")

# ================================================================
# PART 4: PHASE 1 (DnB) FREQUENTIST MODELS
# ================================================================
dnb = long[long["task"] == "DnB"].copy()
dnb_results = []
for dv in ["DnB_perf_z"] + DVs:
    d = dnb.dropna(subset=[dv, "Anxiety_z", "WMC_z"])
    full = smf.ols(f"{dv} ~ Anxiety_z * WMC_z", d).fit()
    for term in ["Anxiety_z", "WMC_z", "Anxiety_z:WMC_z"]:
        others = [t for t in ["Anxiety_z", "WMC_z"] if t != term]
        reduced_formula = (
            f"{dv} ~ " + " + ".join(others) + " + Anxiety_z:WMC_z"
            if term != "Anxiety_z:WMC_z"
            else f"{dv} ~ Anxiety_z + WMC_z"
        )
        reduced = smf.ols(reduced_formula, d).fit()
        f2 = (full.rsquared - reduced.rsquared) / (1 - full.rsquared) if full.rsquared < 1 else np.nan
        dnb_results.append(
            {"DV": dv, "term": term, "coef": full.params[term], "se": full.bse[term], "p": full.pvalues[term], "Cohens_f2": f2}
        )
dnb_df = pd.DataFrame(dnb_results)
print("\nDnB frequentist models:")
print(dnb_df.round(4).to_string())

# ================================================================
# PART 5: PHASE 2 (AVT, 4 levels) FREQUENTIST LMMs + LRT effect sizes + post-hoc
# ================================================================


def fit_ml(formula, data, groups, reml=True):
    for method in ["lbfgs", "cg", "bfgs", "powell", "nm"]:
        try:
            mdf = smf.mixedlm(formula, data, groups=groups).fit(reml=reml, method=method, maxiter=300)
            if mdf.converged or method == "nm":
                return mdf
        except Exception:
            continue
    return None


def marginal_r2(mdf, data):
    fe_pred = mdf.predict(exog=data)
    var_fixed = np.var(fe_pred, ddof=1)
    var_random = float(mdf.cov_re.iloc[0, 0]) if mdf.cov_re.shape[0] > 0 else 0.0
    return var_fixed / (var_fixed + var_random + mdf.scale)


def run_avt_family(tasks, ref, modelname, dvs):
    all_coef, all_lrt, all_posthoc = [], [], []
    for dv in ["TCT_z"] + dvs:
        if dv == "TCT_z":
            d = long[long["task"].isin(tasks)].dropna(subset=["TCT", "Anxiety_z", "WMC_z"]).copy()
            d["TCT_z"] = (d["TCT"] - d.groupby("task")["TCT"].transform("mean")) / d.groupby("task")["TCT"].transform("std")
        else:
            d = long[long["task"].isin(tasks)].dropna(subset=[dv, "Anxiety_z", "WMC_z"]).copy()
        d["task"] = pd.Categorical(d["task"], categories=tasks)
        T = f"C(task, Treatment(reference='{ref}'))"
        formula_full = f"{dv} ~ {T}*Anxiety_z*WMC_z"
        mdf = fit_ml(formula_full, d, d["Participant_ID"], reml=True)
        if mdf is None:
            continue
        for term, coef, se, p in zip(mdf.params.index, mdf.params.values, mdf.bse.values, mdf.pvalues.values):
            if term != "Group Var":
                all_coef.append({"model": modelname, "DV": dv, "term": term, "coef": coef, "se": se, "p": p})

        forms = {
            "full": formula_full,
            "no_3way": f"{dv} ~ {T}*Anxiety_z + {T}*WMC_z + Anxiety_z*WMC_z",
            "no_taskXanx": f"{dv} ~ {T} + Anxiety_z + WMC_z + {T}:WMC_z + Anxiety_z:WMC_z",
            "no_taskXwmc": f"{dv} ~ {T} + Anxiety_z + WMC_z + {T}:Anxiety_z + Anxiety_z:WMC_z",
            "no_anxXwmc": f"{dv} ~ {T} + Anxiety_z + WMC_z + {T}:Anxiety_z + {T}:WMC_z",
        }
        fits = {k: fit_ml(f, d, d["Participant_ID"], reml=False) for k, f in forms.items()}
        if not any(v is None for v in fits.values()):
            r2 = {k: marginal_r2(v, d) for k, v in fits.items()}
            df3 = len(fits["full"].params) - len(fits["no_3way"].params)
            chi2_3 = 2 * (fits["full"].llf - fits["no_3way"].llf)
            p3 = 1 - chi2dist.cdf(max(chi2_3, 0), df3) if df3 > 0 else np.nan
            f2_3 = (r2["full"] - r2["no_3way"]) / (1 - r2["full"]) if r2["full"] < 1 else np.nan
            all_lrt.append(
                {"model": modelname, "DV": dv, "term_family": "Level x Anxiety x WMC (3-way)", "LRT_chi2": chi2_3, "df": df3, "LRT_p": p3, "Cohens_f2": f2_3}
            )
            for key, label in [
                ("no_taskXanx", "Level x Anxiety"),
                ("no_taskXwmc", "Level x WMC"),
                ("no_anxXwmc", "Anxiety x WMC"),
            ]:
                dfx = len(fits["no_3way"].params) - len(fits[key].params)
                chi2x = 2 * (fits["no_3way"].llf - fits[key].llf)
                px = 1 - chi2dist.cdf(max(chi2x, 0), dfx) if dfx > 0 else np.nan
                f2x = (r2["no_3way"] - r2[key]) / (1 - r2["no_3way"]) if r2["no_3way"] < 1 else np.nan
                all_lrt.append(
                    {"model": modelname, "DV": dv, "term_family": label, "LRT_chi2": chi2x, "df": dfx, "LRT_p": px, "Cohens_f2": f2x}
                )

        grid = pd.DataFrame({"task": tasks, "Anxiety_z": 0.0, "WMC_z": 0.0})
        grid["task"] = pd.Categorical(grid["task"], categories=tasks)
        preds = mdf.predict(exog=grid)
        cov = mdf.cov_params()
        _, Xg = patsy.dmatrices(formula_full.replace(f"{dv} ~", "1 ~"), grid, return_type="dataframe")
        for i, j in itertools.combinations(range(len(tasks)), 2):
            contrast = Xg.iloc[i] - Xg.iloc[j]
            common = [c for c in contrast.index if c in cov.index]
            diff = preds.iloc[i] - preds.iloc[j]
            var_diff = contrast[common].values @ cov.loc[common, common].values @ contrast[common].values
            se_diff = np.sqrt(max(var_diff, 0))
            z = diff / se_diff if se_diff > 0 else np.nan
            pval = 2 * (1 - norm.cdf(abs(z))) if se_diff > 0 else np.nan
            all_posthoc.append({"model": modelname, "DV": dv, "contrast": f"{tasks[i]} vs {tasks[j]}", "diff": diff, "se": se_diff, "p": pval})
    return pd.DataFrame(all_coef), pd.DataFrame(all_lrt), pd.DataFrame(all_posthoc)


avt_coef, avt_lrt, avt_posthoc = run_avt_family(["AVT_L1", "AVT_L2", "AVT_L3", "AVT_L4"], "AVT_L1", "AVT_4Levels", DVs)
print("\nPhase 2 (AVT) frequentist models complete.")

# ================================================================
# PART 6: BAYESIAN REPLICATION (bambi / PyMC) -- Phase 1 and Phase 2
# ================================================================
import arviz as az
import bambi as bmb

dnb_bayes_results = []
for dv in ["DnB_perf_z"] + DVs:
    d = dnb.dropna(subset=[dv, "Anxiety_z", "WMC_z"])
    model = bmb.Model(f"{dv} ~ Anxiety_z * WMC_z", d, family="gaussian")
    fitted = model.fit(draws=1000, tune=1000, chains=4, cores=4, progressbar=False, random_seed=42)
    summ = az.summary(fitted, var_names=["Anxiety_z", "WMC_z", "Anxiety_z:WMC_z"], ci_prob=0.95)
    summ["DV"] = dv
    summ["term"] = summ.index
    dnb_bayes_results.append(summ.reset_index(drop=True))
dnb_bayes = pd.concat(dnb_bayes_results, ignore_index=True)

avt = long[long["task"].isin(["AVT_L1", "AVT_L2", "AVT_L3", "AVT_L4"])].copy()
avt["task"] = pd.Categorical(avt["task"], categories=["AVT_L1", "AVT_L2", "AVT_L3", "AVT_L4"])
avt_bayes_results = []
for dv in DVs:
    d = avt.dropna(subset=[dv, "Anxiety_z", "WMC_z"])
    model = bmb.Model(f"{dv} ~ task * Anxiety_z * WMC_z + (1|Participant_ID)", d, family="gaussian")
    fitted = model.fit(draws=800, tune=800, chains=2, cores=2, progressbar=False, random_seed=42)
    summ = az.summary(fitted, var_names=["Anxiety_z", "WMC_z"], filter_vars="like", ci_prob=0.95)
    if summ["r_hat"].max() > 1.01:
        fitted = model.fit(draws=3000, tune=3000, chains=4, cores=4, target_accept=0.95, progressbar=False, random_seed=123)
        summ = az.summary(fitted, var_names=["Anxiety_z", "WMC_z"], filter_vars="like", ci_prob=0.95)
    summ["DV"] = dv
    summ["term"] = summ.index
    avt_bayes_results.append(summ.reset_index(drop=True))
avt_bayes = pd.concat(avt_bayes_results, ignore_index=True)
print("Bayesian replication complete. Max r_hat:", avt_bayes["r_hat"].max())

# ================================================================
# PART 7: MEDIATION -- moderated mediation (bootstrap + Bayesian) and
# bidirectional simple mediation (Bayesian), Phase 1 (DnB)
# ================================================================
mediators = ["EDA_TonicArousal", "Eye_Scanning", "Pupil_SPD_z"]


def fit_paths(d):
    out = {}
    for m in mediators:
        am = smf.ols(f"{m} ~ Anxiety_z * WMC_z", d).fit()
        bm = smf.ols(f"DnB_perf_z ~ Anxiety_z + WMC_z + {m}", d).fit()
        out[m] = {
            "a3": am.params["Anxiety_z:WMC_z"],
            "b": bm.params[m],
            "index_mod_med": am.params["Anxiety_z:WMC_z"] * bm.params[m],
        }
    return out


dnb_mm = dnb.dropna(subset=["DnB_perf_z", "Anxiety_z", "WMC_z"] + mediators)
point_est = fit_paths(dnb_mm)
np.random.seed(1)
boot_idx = {m: [] for m in mediators}
for _ in range(2000):
    samp = dnb_mm.sample(n=len(dnb_mm), replace=True)
    try:
        res = fit_paths(samp)
        for m in mediators:
            boot_idx[m].append(res[m]["index_mod_med"])
    except Exception:
        continue
print("\nModerated mediation (bootstrap):")
for m in mediators:
    ci = np.percentile(boot_idx[m], [2.5, 97.5])
    print(f"  {m}: index={point_est[m]['index_mod_med']:.4f}, 95% CI=[{ci[0]:.4f},{ci[1]:.4f}]")

bayes_mm_draws = {}
for m in mediators:
    d = dnb.dropna(subset=[m, "Anxiety_z", "WMC_z", "DnB_perf_z"])
    am = bmb.Model(f"{m} ~ Anxiety_z * WMC_z", d, family="gaussian")
    afit = am.fit(draws=2000, tune=2000, chains=4, cores=4, progressbar=False, random_seed=42)
    a3_draws = afit.posterior["Anxiety_z:WMC_z"].values.flatten()
    bm = bmb.Model(f"DnB_perf_z ~ Anxiety_z + WMC_z + {m}", d, family="gaussian")
    bfit = bm.fit(draws=2000, tune=2000, chains=4, cores=4, progressbar=False, random_seed=43)
    b_draws = bfit.posterior[m].values.flatten()
    bayes_mm_draws[m] = {"a3": a3_draws, "b": b_draws}

rng = np.random.RandomState(0)
print("\nModerated mediation (Bayesian):")
for m in mediators:
    a3, b = bayes_mm_draws[m]["a3"].copy(), bayes_mm_draws[m]["b"].copy()
    rng.shuffle(a3)
    rng.shuffle(b)
    idx = a3 * b
    ci = np.percentile(idx, [2.5, 97.5])
    pdir = max((idx > 0).mean(), (idx < 0).mean())
    print(f"  {m}: mean={idx.mean():.4f}, 95% CI=[{ci[0]:.4f},{ci[1]:.4f}], P(direction)={pdir:.3f}")

m_a1 = bmb.Model("WMC_z ~ Anxiety_z", dnb_mm, family="gaussian")
a1_draws = m_a1.fit(draws=2000, tune=2000, chains=4, cores=4, progressbar=False, random_seed=1).posterior["Anxiety_z"].values.flatten()
m_a2 = bmb.Model("Anxiety_z ~ WMC_z", dnb_mm, family="gaussian")
a2_draws = m_a2.fit(draws=2000, tune=2000, chains=4, cores=4, progressbar=False, random_seed=2).posterior["WMC_z"].values.flatten()

bidirectional_results = []
for dv in ["DnB_perf_z"] + DVs:
    d = dnb.dropna(subset=[dv])
    mb = bmb.Model(f"{dv} ~ Anxiety_z + WMC_z", d, family="gaussian")
    bfit = mb.fit(draws=2000, tune=2000, chains=4, cores=4, progressbar=False, random_seed=3)
    b_wmc = bfit.posterior["WMC_z"].values.flatten()
    b_anx = bfit.posterior["Anxiety_z"].values.flatten()
    for label, a_draws, b_draws in [("Anxiety->WMC->DV", a1_draws, b_wmc), ("WMC->Anxiety->DV", a2_draws, b_anx)]:
        a_s, b_s = a_draws.copy(), b_draws.copy()
        rng.shuffle(a_s)
        rng.shuffle(b_s)
        idx = a_s * b_s
        ci = np.percentile(idx, [2.5, 97.5])
        bidirectional_results.append(
            {"DV": dv, "path": label, "mean": idx.mean(), "ci_lo": ci[0], "ci_hi": ci[1], "P_direction": max((idx > 0).mean(), (idx < 0).mean())}
        )
bidirectional_df = pd.DataFrame(bidirectional_results)
n_credible = ((bidirectional_df["ci_lo"] > 0) | (bidirectional_df["ci_hi"] < 0)).sum()
print(f"\nBidirectional mediation complete. Credible effects: {n_credible} / {len(bidirectional_df)}")

# ================================================================
# PART 8: RESIDUAL DIAGNOSTICS (all 18 primary models)
# ================================================================
diag_records = []
for dv in ["DnB_perf_z"] + DVs:
    d = dnb.dropna(subset=[dv, "Anxiety_z", "WMC_z"])
    m = smf.ols(f"{dv} ~ Anxiety_z*WMC_z", d).fit()
    resid = m.resid.values
    _, sh_p = shapiro(resid)
    bp = sm.OLS(resid ** 2, sm.add_constant(m.fittedvalues)).fit()
    X = d[["Anxiety_z", "WMC_z"]].copy()
    X["AnxWMC"] = X["Anxiety_z"] * X["WMC_z"]
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    diag_records.append({"model": "DnB", "DV": dv, "Shapiro_p": sh_p, "skew": skew(resid), "kurtosis": kurtosis(resid), "BreuschPagan_p": bp.f_pvalue, "max_VIF": max(vifs)})

for dv in ["TCT_z"] + DVs:
    if dv == "TCT_z":
        d = avt.dropna(subset=["TCT", "Anxiety_z", "WMC_z"]).copy()
        d["TCT_z"] = (d["TCT"] - d.groupby("task")["TCT"].transform("mean")) / d.groupby("task")["TCT"].transform("std")
    else:
        d = avt.dropna(subset=[dv, "Anxiety_z", "WMC_z"]).copy()
    formula = f"{dv} ~ C(task, Treatment(reference='AVT_L1'))*Anxiety_z*WMC_z"
    mdf = fit_ml(formula, d, d["Participant_ID"], reml=True)
    if mdf is None:
        continue
    resid, fitted = mdf.resid.values, mdf.fittedvalues.values
    _, sh_p = shapiro(resid)
    bp = sm.OLS(resid ** 2, sm.add_constant(fitted)).fit()
    _, X = patsy.dmatrices(formula, d, return_type="dataframe")
    X = X.drop(columns=["Intercept"])
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    diag_records.append({"model": "AVT_4Levels", "DV": dv, "Shapiro_p": sh_p, "skew": skew(resid), "kurtosis": kurtosis(resid), "BreuschPagan_p": bp.f_pvalue, "max_VIF": max(vifs)})

diag = pd.DataFrame(diag_records)
print(
    f"\nResidual diagnostics: {(diag['Shapiro_p'] < 0.05).sum()}/{len(diag)} non-normal, "
    f"{(diag['BreuschPagan_p'] < 0.05).sum()}/{len(diag)} heteroscedastic, max VIF={diag['max_VIF'].max():.2f}"
)

# ================================================================
# PART 9: MULTIPLE-COMPARISONS / FDR-OMISSION JUSTIFICATION
# ================================================================
n_tested = len(dnb_df) + len(avt_coef[~avt_coef["term"].str.contains("Intercept")])
n_sig = (dnb_df["p"] < 0.05).sum() + (avt_coef[~avt_coef["term"].str.contains("Intercept")]["p"] < 0.05).sum()
bt = binomtest(int(n_sig), n_tested, 0.05, alternative="greater")
print(f"\nFDR justification: {n_tested} terms tested, {n_sig} significant, {n_tested * 0.05:.1f} expected by chance, binomial p={bt.pvalue:.2e}")

# ================================================================
# PART 10: EXPORT ALL RESULTS
# ================================================================
with pd.ExcelWriter("results_final.xlsx", engine="openpyxl") as writer:
    long.to_excel(writer, sheet_name="Long_data_with_factors", index=False)
    eye_fit["loadings"].assign(communality=eye_fit["communalities"]).to_excel(writer, sheet_name="EFA_Eye_Loadings")
    eda_fit["loadings"].assign(communality=eda_fit["communalities"]).to_excel(writer, sheet_name="EFA_EDA_Loadings")
    dnb_df.to_excel(writer, sheet_name="DnB_Frequentist", index=False)
    dnb_bayes.to_excel(writer, sheet_name="DnB_Bayesian", index=False)
    avt_coef.to_excel(writer, sheet_name="AVT_Frequentist", index=False)
    avt_lrt.to_excel(writer, sheet_name="AVT_LRT_EffectSizes", index=False)
    avt_bayes.to_excel(writer, sheet_name="AVT_Bayesian", index=False)
    avt_posthoc.to_excel(writer, sheet_name="AVT_PostHoc", index=False)
    bidirectional_df.to_excel(writer, sheet_name="Mediation_Bidirectional", index=False)
    diag.to_excel(writer, sheet_name="Residual_Diagnostics", index=False)

print("\nDone. See results_final.xlsx")

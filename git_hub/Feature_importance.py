"""
WMC Feature Importance Analysis
=================================
Computes permutation-based feature importance using Random Forest.
Produces:
  1. Cumulative — Eye-Tracking Only          (FIG_cumulative_eye_only.png)
  2. Cumulative — Eye-Tracking + EDA         (FIG_cumulative_eye_eda.png)
  3. Side-by-side cumulative comparison      (FIG_cumulative_4task_side_by_side.png)
  4. 4-panel master figure (A/B/C/D)         (FIG_master_4panel.png)
  5. Per-task — Eye-Tracking Only  (8 tasks) (FIG_pertask_eye_only.png)
  6. Per-task — Eye-Tracking + EDA (8 tasks) (FIG_pertask_eye_eda.png)
  7. CSVs: cumulative_importance_eye_only.csv
           cumulative_importance_eye_eda.csv

Usage
-----
    python wmc_feature_importance.py

Set DATA_FILE and OUTPUT_DIR in the CONFIG block below.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  –  edit only this block
# ─────────────────────────────────────────────────────────────
DATA_FILE   = "Data_set_paper.xlsx"
OUTPUT_DIR  = "/Users/Saurabh/Documents/Python_analysis data set paper/Outcome"
LABEL_COL   = "WMC_Label"
TASK_COL    = "Task"
ID_COL      = "Participant_ID"
RANDOM_SEED = 42
N_REPEATS   = 50       # permutation repeats — matches FIG_master_4panel title

# Eye-tracking features
EYE_FEATURES = ["TCT","TTFF","FFD","TFD","NOF","AFD",
                 "SSD","SSV","MSD","MSA","MSV","TSD","PSV","MPD","SPD"]
# EDA features
EDA_FEATURES = ["EDA_Responses","Mean_EDA_Amplitude","EDA_Proportion"]
# Combined
ALL_FEATURES = EYE_FEATURES + EDA_FEATURES

# Colour scheme — matches your figures
COL_EYE = "#2E7D32"   # dark green
COL_EDA = "#1565C0"   # dark blue
BG_COL  = "#F7F7F7"   # light grey panel background
# ─────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  DEMO DATA GENERATOR  — remove once real data is available
# ═══════════════════════════════════════════════════════════════
def generate_demo_data(n_participants=96, seed=42):
    rng   = np.random.default_rng(seed)
    tasks = ["BVT", "DnB", "AVT_L1", "AVT_L2", "AVT_L3",
             "SVST_L1", "SVST_L2", "SVST_L3"]
    labels = np.array([1]*49 + [0]*47)
    rows   = []
    for pid, label in enumerate(labels):
        for task in tasks:
            shift = 0.4 if label == 1 else -0.4
            row = {ID_COL: f"P{pid+1:03d}", TASK_COL: task, LABEL_COL: label}
            for feat in EYE_FEATURES:
                row[feat] = rng.normal(loc=50 + shift*5, scale=15)
            for feat in EDA_FEATURES:
                row[feat] = rng.normal(loc=1.5 + shift*0.3, scale=0.5)
            rows.append(row)
    df = pd.DataFrame(rows)
    print("  [demo] synthetic data:", df.shape,
          "| balance:", df[LABEL_COL].value_counts().to_dict())
    return df


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def make_rf_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200,
                                          random_state=RANDOM_SEED,
                                          n_jobs=-1)),
    ])


def build_Xy(sub_df, labels_map, feature_list, agg="mean"):
    """Aggregate features per participant then return X, y arrays."""
    feat_df = (sub_df.groupby(ID_COL)[feature_list]
                     .agg(agg)
                     .reset_index())
    feat_df[LABEL_COL] = feat_df[ID_COL].map(labels_map)
    feat_df = feat_df.dropna(subset=[LABEL_COL])
    X = feat_df[feature_list].values.astype(float)
    y = feat_df[LABEL_COL].values.astype(int)
    return X, y


def compute_importance(pipe, X, y, feature_list):
    """
    Fit pipeline, run permutation importance, return:
      mean_norm  — normalised mean (sum = 1, negatives clipped to 0)
      std_norm   — SD normalised by same factor for consistent error bars
    """
    pipe.fit(X, y)
    r        = permutation_importance(pipe, X, y,
                                       n_repeats=N_REPEATS,
                                       random_state=RANDOM_SEED,
                                       n_jobs=-1)
    mean_raw = r.importances_mean
    std_raw  = r.importances_std
    clipped  = np.clip(mean_raw, 0, None)
    total    = clipped.sum() + 1e-9
    mean_norm = clipped / total
    std_norm  = std_raw / total
    return mean_norm, std_norm


def save_importance_csv(mean_norm, std_norm, feature_list, out_path):
    types = ["eda" if f in EDA_FEATURES else "eye" for f in feature_list]
    df = pd.DataFrame({
        "Feature":         feature_list,
        "Type":            types,
        "Mean_Importance": mean_norm,
        "SD_Importance":   std_norm,
    })
    df = df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}")
    return df


# ═══════════════════════════════════════════════════════════════
#  CORE DRAW FUNCTION — single axes panel
# ═══════════════════════════════════════════════════════════════
def _draw_importance_ax(ax, mean_norm, std_norm, feature_list, title,
                        show_legend=True):
    """
    Horizontal importance bar chart on a given axes object.
    Features sorted ascending so most important is at the top.
    """
    order   = np.argsort(mean_norm)           # ascending → top = most important
    feats_o = np.array(feature_list)[order]
    mean_o  = mean_norm[order]
    std_o   = std_norm[order]
    colors  = [COL_EDA if f in EDA_FEATURES else COL_EYE for f in feats_o]
    y_pos   = np.arange(len(order))

    ax.set_facecolor(BG_COL)
    ax.barh(y_pos, mean_o, xerr=std_o, color=colors,
            edgecolor="white", linewidth=0.3,
            capsize=2, error_kw={"elinewidth": 0.8, "ecolor": "black"})

    # Value labels at bar end
    for i, (v, s) in enumerate(zip(mean_o, std_o)):
        if v > 0 or s > 0:
            ax.text(v + s + 0.003, i, f"±{v:.3f}",
                    va="center", ha="left", fontsize=6.5, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats_o, fontsize=8)
    ax.set_xlabel("Normalised Permutation Importance", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.axvline(0, color="#999999", linewidth=0.6)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    if show_legend:
        patches = [mpatches.Patch(color=COL_EYE, label="Eye-Tracking Feature"),
                   mpatches.Patch(color=COL_EDA, label="EDA Feature")]
        ax.legend(handles=patches, loc="lower right",
                  fontsize=7, framealpha=0.7)


# ═══════════════════════════════════════════════════════════════
#  FIG 1 & 2  — individual cumulative figures
# ═══════════════════════════════════════════════════════════════
def plot_cumulative_single(mean_norm, std_norm, feature_list,
                           suptitle, subtitle, out_path):
    n = len(feature_list)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.38)))
    _draw_importance_ax(ax, mean_norm, std_norm, feature_list,
                        subtitle, show_legend=True)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ═══════════════════════════════════════════════════════════════
#  FIG 3  — side-by-side cumulative (Eye-Only | Eye+EDA)
# ═══════════════════════════════════════════════════════════════
def plot_cumulative_side_by_side(eye_mean, eye_std, eye_feats,
                                  eda_mean, eda_std, eda_feats,
                                  out_path):
    n_max = max(len(eye_feats), len(eda_feats))
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                    figsize=(18, max(6, n_max * 0.38)))
    fig.suptitle(
        "Cumulative Feature Importance for WMC Classification\n"
        "(Averaged Across All Four Tasks: BVT, DnB, AVT, SVST)",
        fontsize=13, fontweight="bold"
    )
    _draw_importance_ax(ax1, eye_mean, eye_std, eye_feats,
                         "Eye-Tracking Features Only", show_legend=False)
    _draw_importance_ax(ax2, eda_mean, eda_std, eda_feats,
                         "Eye-Tracking + EDA Features", show_legend=False)

    patches = [mpatches.Patch(color=COL_EYE, label="Eye-Tracking Feature"),
               mpatches.Patch(color=COL_EDA, label="EDA Feature")]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ═══════════════════════════════════════════════════════════════
#  FIG 4  — 4-panel master figure  (A/B cumulative, C/D best task)
# ═══════════════════════════════════════════════════════════════
def plot_master_4panel(eye_cum_mean, eye_cum_std, eye_feats,
                        eda_cum_mean, eda_cum_std, eda_feats,
                        best_task_name,
                        eye_best_mean, eye_best_std,
                        eda_best_mean, eda_best_std,
                        out_path):
    n_max = max(len(eye_feats), len(eda_feats))
    row_h = max(5, n_max * 0.30)

    fig, axes = plt.subplots(2, 2, figsize=(18, row_h * 2 + 2))
    fig.suptitle(
        "Feature Importance for WMC Classification\n"
        f"Random Forest Permutation Importance (n_repeats={N_REPEATS})"
        " — Error Bars = ±1 SD",
        fontsize=12, fontweight="bold"
    )

    _draw_importance_ax(axes[0, 0], eye_cum_mean, eye_cum_std, eye_feats,
                         "A  Cumulative — Eye-Tracking Only")
    _draw_importance_ax(axes[0, 1], eda_cum_mean, eda_cum_std, eda_feats,
                         "B  Cumulative — Eye-Tracking + EDA")
    _draw_importance_ax(axes[1, 0], eye_best_mean, eye_best_std, eye_feats,
                         f"C  Best Task ({best_task_name}) — Eye-Tracking Only")
    _draw_importance_ax(axes[1, 1], eda_best_mean, eda_best_std, eda_feats,
                         f"D  Best Task ({best_task_name}) — Eye-Tracking + EDA")

    patches = [mpatches.Patch(color=COL_EYE, label="Eye-Tracking Feature"),
               mpatches.Patch(color=COL_EDA, label="EDA Feature")]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ═══════════════════════════════════════════════════════════════
#  FIG 5 & 6  — per-task 2×4 grid
# ═══════════════════════════════════════════════════════════════
def plot_pertask_grid(task_imp_dict, feature_list, mode_label, out_path):
    """
    task_imp_dict: {task_name: (mean_norm, std_norm)}
    Produces a 2×4 grid — one subplot per task.
    """
    tasks   = list(task_imp_dict.keys())
    ncols   = 4
    nrows   = 2
    n_feats = len(feature_list)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5,
                                       nrows * max(5, n_feats * 0.28)))
    fig.suptitle(
        f"Per-Task Feature Importance — {mode_label}\n"
        "WMC Classification (Random Forest Permutation Importance)",
        fontsize=12, fontweight="bold"
    )

    for idx, task in enumerate(tasks):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        mean_n, std_n = task_imp_dict[task]
        _draw_importance_ax(ax, mean_n, std_n, feature_list,
                             f"{task} — {mode_label}",
                             show_legend=False)

    # Hide unused axes
    for i in range(len(tasks), nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    patches = [mpatches.Patch(color=COL_EYE, label="Eye-Tracking Feature"),
               mpatches.Patch(color=COL_EDA, label="EDA Feature")]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/5] Loading data …")
    if os.path.exists(DATA_FILE):
        df_raw = pd.read_excel(DATA_FILE, engine="openpyxl")
        print(f"  Loaded: {df_raw.shape}")
    else:
        print(f"  '{DATA_FILE}' not found → using SYNTHETIC DEMO DATA.")
        df_raw = generate_demo_data()

    required = [ID_COL, LABEL_COL, TASK_COL] + ALL_FEATURES
    missing  = [c for c in required if c not in df_raw.columns]
    if missing:
        print(f"\n  ⚠  Missing columns: {missing}")
        print("  Available columns:", list(df_raw.columns))
        return

    labels_map    = df_raw.groupby(ID_COL)[LABEL_COL].first().to_dict()
    tasks_present = sorted(df_raw[TASK_COL].unique())
    print(f"  Tasks found:   {tasks_present}")
    print(f"  Participants:  {df_raw[ID_COL].nunique()}")
    print(f"  Label balance: "
          f"{df_raw.groupby(ID_COL)[LABEL_COL].first().value_counts().to_dict()}")

    # ── 2. Cumulative matrices ─────────────────────────────────
    print("\n[2/5] Building cumulative feature matrices …")
    X_eye_cum, y_eye_cum = build_Xy(df_raw, labels_map, EYE_FEATURES)
    X_eda_cum, y_eda_cum = build_Xy(df_raw, labels_map, ALL_FEATURES)

    # ── 3. Cumulative importance ───────────────────────────────
    print("\n[3/5] Computing cumulative permutation importance …")

    print("  Eye-Only cumulative …")
    eye_cum_mean, eye_cum_std = compute_importance(
        make_rf_pipeline(), X_eye_cum, y_eye_cum, EYE_FEATURES)

    print("  Eye+EDA cumulative …")
    eda_cum_mean, eda_cum_std = compute_importance(
        make_rf_pipeline(), X_eda_cum, y_eda_cum, ALL_FEATURES)

    # Save CSVs  (matching exact column names from your uploaded files)
    save_importance_csv(
        eye_cum_mean, eye_cum_std, EYE_FEATURES,
        os.path.join(OUTPUT_DIR, "cumulative_importance_eye_only.csv")
    )
    save_importance_csv(
        eda_cum_mean, eda_cum_std, ALL_FEATURES,
        os.path.join(OUTPUT_DIR, "cumulative_importance_eye_eda.csv")
    )

    # ── 4. Per-task importance ─────────────────────────────────
    print("\n[4/5] Computing per-task permutation importance …")

    eye_task_imp = {}    # {task: (mean_norm, std_norm)}
    eda_task_imp = {}
    eye_task_acc = {}    # for best-task selection

    for task in tasks_present:
        sub = df_raw[df_raw[TASK_COL] == task]
        if sub[ID_COL].nunique() < 6:
            print(f"  Skipping {task} — only {sub[ID_COL].nunique()} participants")
            continue
        print(f"  {task} …")

        # Eye-only
        Xe, ye = build_Xy(sub, labels_map, EYE_FEATURES)
        m_e, s_e = compute_importance(make_rf_pipeline(), Xe, ye, EYE_FEATURES)
        eye_task_imp[task] = (m_e, s_e)

        # Track RF train accuracy for best-task selection
        pipe_acc = make_rf_pipeline()
        pipe_acc.fit(Xe, ye)
        eye_task_acc[task] = accuracy_score(ye, pipe_acc.predict(Xe))

        # Eye+EDA
        Xa, ya = build_Xy(sub, labels_map, ALL_FEATURES)
        m_a, s_a = compute_importance(make_rf_pipeline(), Xa, ya, ALL_FEATURES)
        eda_task_imp[task] = (m_a, s_a)

    best_task = max(eye_task_acc, key=eye_task_acc.get)
    print(f"\n  Best task: {best_task}  "
          f"(RF accuracy = {eye_task_acc[best_task]*100:.1f}%)")

    eye_best_mean, eye_best_std = eye_task_imp[best_task]
    eda_best_mean, eda_best_std = eda_task_imp[best_task]

    # ── 5. Generate all figures ────────────────────────────────
    print("\n[5/5] Generating figures …")

    # FIG 1 — cumulative eye-only
    plot_cumulative_single(
        eye_cum_mean, eye_cum_std, EYE_FEATURES,
        suptitle=(
            "Cumulative Feature Importance — Eye-Tracking Only\n"
            "(Averaged Across All Tasks: BVT, DnB, AVT-L1/L2/L3, SVST-L1/L2/L3)"
        ),
        subtitle="Eye-Tracking Features Only",
        out_path=os.path.join(OUTPUT_DIR, "FIG_cumulative_eye_only.png")
    )

    # FIG 2 — cumulative eye+EDA
    plot_cumulative_single(
        eda_cum_mean, eda_cum_std, ALL_FEATURES,
        suptitle=(
            "Cumulative Feature Importance — Eye-Tracking + EDA\n"
            "(Averaged Across All Tasks: BVT, DnB, AVT-L1/L2/L3, SVST-L1/L2/L3)"
        ),
        subtitle="Eye-Tracking + EDA Features",
        out_path=os.path.join(OUTPUT_DIR, "FIG_cumulative_eye_eda.png")
    )

    # FIG 3 — side-by-side cumulative
    plot_cumulative_side_by_side(
        eye_cum_mean, eye_cum_std, EYE_FEATURES,
        eda_cum_mean, eda_cum_std, ALL_FEATURES,
        out_path=os.path.join(OUTPUT_DIR, "FIG_cumulative_4task_side_by_side.png")
    )

    # FIG 4 — 4-panel master figure
    plot_master_4panel(
        eye_cum_mean, eye_cum_std, EYE_FEATURES,
        eda_cum_mean, eda_cum_std, ALL_FEATURES,
        best_task_name=best_task,
        eye_best_mean=eye_best_mean, eye_best_std=eye_best_std,
        eda_best_mean=eda_best_mean, eda_best_std=eda_best_std,
        out_path=os.path.join(OUTPUT_DIR, "FIG_master_4panel.png")
    )

    # FIG 5 — per-task Eye-Only 2×4 grid
    plot_pertask_grid(
        eye_task_imp, EYE_FEATURES,
        mode_label="Eye-Tracking Only",
        out_path=os.path.join(OUTPUT_DIR, "FIG_pertask_eye_only.png")
    )

    # FIG 6 — per-task Eye+EDA 2×4 grid
    plot_pertask_grid(
        eda_task_imp, ALL_FEATURES,
        mode_label="Eye-Tracking + EDA",
        out_path=os.path.join(OUTPUT_DIR, "FIG_pertask_eye_eda.png")
    )

    # Console summary — top 5 per cumulative condition
    print(f"\n{'═'*65}")
    print("  TOP 5 — CUMULATIVE EYE-TRACKING ONLY")
    print(f"{'═'*65}")
    eye_df = pd.DataFrame({"Feature": EYE_FEATURES,
                            "Mean":    eye_cum_mean,
                            "SD":      eye_cum_std})
    print(eye_df.sort_values("Mean", ascending=False)
               .head(5).to_string(index=False))

    print(f"\n{'═'*65}")
    print("  TOP 5 — CUMULATIVE EYE-TRACKING + EDA")
    print(f"{'═'*65}")
    eda_df = pd.DataFrame({"Feature": ALL_FEATURES,
                            "Mean":    eda_cum_mean,
                            "SD":      eda_cum_std})
    print(eda_df.sort_values("Mean", ascending=False)
               .head(5).to_string(index=False))

    print(f"\n{'═'*65}")
    print(f"  ✓  All outputs saved to  →  {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()

"""
Step 2: Multi-Study Harmonization and Cross-Validation Feature Selection
Integrates ASD blood transcriptomic cohorts (GSE26415, GSE111175, GSE42133),
generates exploratory PCA and UMAP plots before and after harmonization,
and builds Leave-One-Dataset-Out (LODO) cross-validation splits using
network topological consensus hub selection and stability resampling.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import umap
except ImportError:
    umap = None

# Study list and filtering parameters
STUDIES = ["GSE26415", "GSE111175", "GSE42133"]
RANDOM_STATE = 42
VAR_CUTOFF = 1e-8
CORR_CUTOFF = 0.90
N_REPEATS = 50
STABILITY_CUTOFF = 0.40

# Directory paths
OUT_DIR = Path("step2_split_featsel")
LODO_DIR = OUT_DIR / "lodo_folds"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"

for d in [OUT_DIR, LODO_DIR, PLOT_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Visual styling
STUDY_COLORS = {"GSE26415": "#E74C3C", "GSE111175": "#3498DB", "GSE42133": "#2ECC71"}
DIAG_COLORS = {"ASD": "#E63946", "Control": "#2A9D8F"}
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False
})


# -------------------------------------------------------------------------
# 1. Load expression matrices and align common gene space
# -------------------------------------------------------------------------
def load_and_align_cohorts():
    """Load study expression profiles, log-transform GSE26415, and intersect common genes."""
    expr_dict = {}
    meta_dict = {}
    gene_sets = []

    for study_id in STUDIES:
        expr_path = Path(f"{study_id}_data/processed/expression_gene_level.csv")
        meta_path = Path(f"{study_id}_data/processed/sample_metadata.csv")

        expr = pd.read_csv(expr_path, index_col=0)
        meta = pd.read_csv(meta_path, index_col=0)

        # Select samples with confirmed diagnosis (ASD: 1, Control: 0)
        valid_samples = meta[meta["label"].isin([0, 1])].index
        common = [s for s in expr.columns if s in valid_samples]
        expr = expr[common].copy()
        meta = meta.loc[common].copy()

        # Check linear vs log2 scale (GSE26415 is on raw intensity)
        if study_id == "GSE26415" and float(np.nanmax(expr.values)) > 50:
            print(f"Log2 transforming raw intensities for {study_id}")
            expr = np.log2(np.clip(expr.astype(float), 0, None) + 1.0)

        expr_dict[study_id] = expr
        meta_dict[study_id] = meta
        gene_sets.append(set(expr.index))

    # Intersect common gene symbols
    common_genes = sorted(list(set.intersection(*gene_sets)))
    print(f"Intersected {len(common_genes)} common genes across cohorts: {', '.join(STUDIES)}")

    # Merge into a single sample-by-gene matrix
    rows, studies, labels, sample_names = [], [], [], []
    for study_id in STUDIES:
        sub = expr_dict[study_id].loc[common_genes].T
        meta = meta_dict[study_id]
        for sid in sub.index:
            sample_names.append(sid)
            studies.append(study_id)
            labels.append(int(meta.loc[sid, "label"]))
            rows.append(sub.loc[sid].values)

    merged_df = pd.DataFrame(rows, index=sample_names, columns=common_genes)
    study_series = pd.Series(studies, index=sample_names, name="study")
    label_series = pd.Series(labels, index=sample_names, name="label")

    n_asd = int((label_series == 1).sum())
    n_ctrl = int((label_series == 0).sum())
    print(f"Total merged cohort: {merged_df.shape[0]} samples (ASD: {n_asd}, Control: {n_ctrl}) x {merged_df.shape[1]} genes")

    merged_df.to_csv(OUT_DIR / "merged_expression_before_harmonization.csv")
    return merged_df, study_series, label_series, common_genes


# -------------------------------------------------------------------------
# 2. Exploratory PCA and UMAP plots before and after harmonization
# -------------------------------------------------------------------------
def plot_pca_umap(merged_expr, study_series, label_series):
    """Generates Figure 6 (PCA before/after) and Figure 7 (UMAP before/after)."""
    scaler = StandardScaler()
    scaled_before = scaler.fit_transform(merged_expr)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_before = pca.fit_transform(scaled_before)

    # Simple mean-centering by study for exploratory cross-study visualization
    adj_rows = []
    for s in STUDIES:
        m = (study_series == s).values
        mat = scaled_before[m]
        centered = mat - np.mean(mat, axis=0)
        adj_rows.append(pd.DataFrame(centered, index=merged_expr.index[m]))
    adj_df = pd.concat(adj_rows).loc[merged_expr.index]
    adj_df.to_csv(OUT_DIR / "merged_expression_after_harmonization.csv")

    pca_after = pca.fit_transform(adj_df.values)

    # Figure 6: PCA comparison
    for stage, coords, fname in [("before", pca_before, "Figure_6_PCA_before.png"),
                                 ("after", pca_after, "Figure_6_PCA_after.png")]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        for s in STUDIES:
            m = (study_series == s).values
            ax1.scatter(coords[m, 0], coords[m, 1], c=STUDY_COLORS[s], label=f"{s} (n={m.sum()})", s=45, alpha=0.85, edgecolors="white", lw=0.5)
        ax1.set_title(f"PCA by Cohort ({stage.capitalize()} Harmonization)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("PC 1", fontsize=10, fontweight="bold")
        ax1.set_ylabel("PC 2", fontsize=10, fontweight="bold")
        ax1.legend(frameon=True, facecolor="white", fontsize=9.5)

        for grp, lbl, col in [(0, "Control", DIAG_COLORS["Control"]), (1, "ASD", DIAG_COLORS["ASD"])]:
            m = (label_series == grp).values
            ax2.scatter(coords[m, 0], coords[m, 1], c=col, label=f"{lbl} (n={m.sum()})", s=45, alpha=0.85, edgecolors="white", lw=0.5)
        ax2.set_title(f"PCA by Diagnosis ({stage.capitalize()} Harmonization)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("PC 1", fontsize=10, fontweight="bold")
        ax2.set_ylabel("PC 2", fontsize=10, fontweight="bold")
        ax2.legend(frameon=True, facecolor="white", fontsize=9.5)

        plt.suptitle(f"Figure 6. Principal Component Analysis {stage.capitalize()} Harmonization", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / fname, bbox_inches="tight")
        plt.close()

    # Figure 7: UMAP comparison
    if umap is not None:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=RANDOM_STATE)
        umap_before = reducer.fit_transform(scaled_before)
        umap_after = reducer.fit_transform(adj_df.values)

        for stage, coords, fname in [("before", umap_before, "Figure_7_UMAP_before.png"),
                                     ("after", umap_after, "Figure_7_UMAP_after.png")]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

            for s in STUDIES:
                m = (study_series == s).values
                ax1.scatter(coords[m, 0], coords[m, 1], c=STUDY_COLORS[s], label=f"{s} (n={m.sum()})", s=45, alpha=0.85, edgecolors="white", lw=0.5)
            ax1.set_title(f"UMAP by Cohort ({stage.capitalize()} Harmonization)", fontsize=11, fontweight="bold")
            ax1.set_xlabel("UMAP 1", fontsize=10, fontweight="bold")
            ax1.set_ylabel("UMAP 2", fontsize=10, fontweight="bold")
            ax1.legend(frameon=True, facecolor="white", fontsize=9.5)

            for grp, lbl, col in [(0, "Control", DIAG_COLORS["Control"]), (1, "ASD", DIAG_COLORS["ASD"])]:
                m = (label_series == grp).values
                ax2.scatter(coords[m, 0], coords[m, 1], c=col, label=f"{lbl} (n={m.sum()})", s=45, alpha=0.85, edgecolors="white", lw=0.5)
            ax2.set_title(f"UMAP by Diagnosis ({stage.capitalize()} Harmonization)", fontsize=11, fontweight="bold")
            ax2.set_xlabel("UMAP 1", fontsize=10, fontweight="bold")
            ax2.set_ylabel("UMAP 2", fontsize=10, fontweight="bold")
            ax2.legend(frameon=True, facecolor="white", fontsize=9.5)

            plt.suptitle(f"Figure 7. UMAP Visualization {stage.capitalize()} Harmonization", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / fname, bbox_inches="tight")
            plt.close()


# -------------------------------------------------------------------------
# 3. Network topological consensus hub candidate selection
# -------------------------------------------------------------------------
def get_topological_candidate_hubs(training_studies, common_gene_set):
    """Extracts genes ranked in the top 50 across at least 2 topological metrics (MCC, Degree, Betweenness)."""
    hub_dir = Path("step2_hubgene_analysis")
    gene_votes = {}

    for study_id in training_studies:
        files = {
            "MCC": hub_dir / f"{study_id}_hub_gene_MCC.csv",
            "degree": hub_dir / f"{study_id}_hub_gene_degree.csv",
            "betweenness": hub_dir / f"{study_id}_hub_gene_betweenness.csv"
        }

        study_top50 = {}
        for m, fpath in files.items():
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            col = next((c for c in ["HGNC_Symbol", "Name", "gene", "Gene", "name", "Symbol"] if c in df.columns), None)
            if col:
                genes = df[col].dropna().astype(str).str.strip().head(50).tolist()
                study_top50[m] = set(genes)

        all_study_genes = set.union(*study_top50.values()) if study_top50 else set()
        for g in all_study_genes:
            if g in common_gene_set:
                votes = sum(int(g in study_top50[m]) for m in study_top50)
                if votes >= 2:
                    gene_votes[g] = gene_votes.get(g, 0) + 1

    candidate_hubs = sorted(list(gene_votes.keys()))
    print(f"  Consensus candidate hubs in training studies ({', '.join(training_studies)}): {len(candidate_hubs)} genes")
    return candidate_hubs


# -------------------------------------------------------------------------
# 4. Variance, correlation filtering, and stability selection
# -------------------------------------------------------------------------
def run_stability_selection(X_train, y_train, candidate_genes):
    """Filters near-zero variance and high correlation, then resamples 50 times using Welch's t-test."""
    df = X_train[candidate_genes].copy()

    # Low-variance filter
    vars_ = df.var(axis=0)
    var_genes = vars_[vars_ > VAR_CUTOFF].index.tolist()
    df = df[var_genes]

    # Correlation filter (drop one gene of pairs with |r| > 0.90)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > CORR_CUTOFF)]
    uncorr_genes = [g for g in var_genes if g not in drop_cols]

    # 50 repeats of StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=0.25, random_state=RANDOM_STATE)
    counts = {g: 0 for g in uncorr_genes}
    X_mat = df[uncorr_genes].values
    y_vec = y_train.values

    for train_idx, _ in cv.split(X_mat, y_vec):
        x_sub = X_mat[train_idx]
        y_sub = y_vec[train_idx]
        asd_sub = x_sub[y_sub == 1]
        ctrl_sub = x_sub[y_sub == 0]

        t_stat, _ = stats.ttest_ind(asd_sub, ctrl_sub, equal_var=False)
        top_idx = np.argsort(np.abs(t_stat))[::-1][:min(30, len(uncorr_genes))]
        for idx in top_idx:
            counts[uncorr_genes[idx]] += 1

    # Keep genes selected above frequency threshold
    stable_genes = [g for g, cnt in counts.items() if (cnt / N_REPEATS) >= STABILITY_CUTOFF]
    if len(stable_genes) < 5:
        stable_genes = sorted(counts, key=counts.get, reverse=True)[:15]

    print(f"  Stable biomarkers retained: {len(stable_genes)} genes (selection frequency >= {STABILITY_CUTOFF*100:.0f}%)")
    return stable_genes


# -------------------------------------------------------------------------
# 5. Raincloud plots for biomarker expression distributions
# -------------------------------------------------------------------------
def plot_rainclouds(fold_results):
    """Supplementary Figure S1: Raincloud plots of biomarker expression distributions in test cohorts."""
    panel_map = {"GSE26415": "A", "GSE111175": "B", "GSE42133": "C"}

    for held_out, data in fold_results.items():
        sub_letter = panel_map.get(held_out, "A")
        genes = data["stable_genes"][:min(12, len(data["stable_genes"]))]
        X_test = data["X_test"][genes]
        y_test = data["y_test"]

        rows = []
        for g in genes:
            for sid, val in X_test[g].items():
                rows.append({
                    "Gene": g,
                    "Expression": float(val),
                    "Phenotype": "ASD" if y_test.loc[sid] == 1 else "Control"
                })
        df_plot = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=df_plot, x="Gene", y="Expression", hue="Phenotype", split=True,
                       inner=None, alpha=0.45, palette={"ASD": "#E63946", "Control": "#2A9D8F"}, ax=ax, cut=0)
        sns.boxplot(data=df_plot, x="Gene", y="Expression", hue="Phenotype", width=0.25,
                    boxprops=dict(alpha=0.8, zorder=2), showfliers=False, palette={"ASD": "#E63946", "Control": "#2A9D8F"}, ax=ax, dodge=True)
        sns.stripplot(data=df_plot, x="Gene", y="Expression", hue="Phenotype", dodge=True,
                      jitter=0.12, size=3.5, alpha=0.7, palette={"ASD": "#8B0000", "Control": "#004D40"}, ax=ax, zorder=3)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Phenotype", loc="upper right", frameon=True, facecolor="white")
        ax.set_title(f"Supplementary Figure S1{sub_letter}. Biomarker Expression Distributions (Held-Out: {held_out})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Consensus Biomarker Gene", fontsize=11, fontweight="bold")
        ax.set_ylabel("Standardized Expression (Z-score)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"Supplementary_Figure_S1{sub_letter}.png")
        plt.close()


# -------------------------------------------------------------------------
# 6. Main execution: build LODO folds and export matrices
# -------------------------------------------------------------------------
def main():
    print("Starting Step 2 multi-study harmonization and feature selection...")

    # Load and merge cohorts
    merged_expr, study_series, label_series, common_genes = load_and_align_cohorts()
    common_gene_set = set(common_genes)

    # Pre- and post-harmonization plots
    plot_pca_umap(merged_expr, study_series, label_series)

    # Leave-One-Dataset-Out cross-validation splits
    table_9_rows = []
    fold_results = {}

    for held_out in STUDIES:
        print(f"\nConstructing LODO split (Held-out cohort: {held_out})...")
        train_studies = [s for s in STUDIES if s != held_out]

        train_mask = (study_series != held_out).values
        test_mask = (study_series == held_out).values

        X_train_raw = merged_expr.loc[train_mask].copy()
        y_train = label_series.loc[train_mask].copy()
        X_test_raw = merged_expr.loc[test_mask].copy()
        y_test = label_series.loc[test_mask].copy()

        # Fit scaling on training data only
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), index=X_train_raw.index, columns=X_train_raw.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), index=X_test_raw.index, columns=X_test_raw.columns)

        # Consensus hub candidate selection and stability selection
        hubs = get_topological_candidate_hubs(train_studies, common_gene_set)
        stable_genes = run_stability_selection(X_train_scaled, y_train, hubs)

        # Save fold matrices
        X_tr = X_train_scaled[stable_genes].copy()
        X_te = X_test_scaled[stable_genes].copy()

        fold_dir = LODO_DIR / f"LODO_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        X_tr.to_csv(fold_dir / "X_train.csv")
        X_te.to_csv(fold_dir / "X_test.csv")
        y_train.to_csv(fold_dir / "y_train.csv")
        y_test.to_csv(fold_dir / "y_test.csv")
        pd.DataFrame({"gene": stable_genes}).to_csv(fold_dir / "stable_biomarker_panel.csv", index=False)

        fold_results[held_out] = {
            "X_test": X_te,
            "y_test": y_test,
            "stable_genes": stable_genes
        }

        table_9_rows.append({
            "Held-Out Cohort": held_out,
            "Training Cohorts": ", ".join(train_studies),
            "Training Samples": len(y_train),
            "Testing Samples": len(y_test),
            "Candidate Topological Hubs": len(hubs),
            "Stable Genes": len(stable_genes)
        })

    # Save Table 9
    table_9 = pd.DataFrame(table_9_rows)
    table_9.to_csv(TABLE_DIR / "Table_9_LODO_Fold_Configuration_and_Biomarkers.csv", index=False)
    print("Saved Table 9 (LODO fold configuration and biomarkers)")

    # Generate Supplementary Figure S1
    plot_rainclouds(fold_results)

    print("\nStep 2 feature selection completed successfully.")


if __name__ == "__main__":
    main()

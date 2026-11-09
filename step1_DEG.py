"""
Step 1: Differential Expression Analysis (limma) and Cross-Study Meta-Analysis
Analyzes gene expression differences between ASD and Neurotypical Controls
across microarray cohorts (GSE26415, GSE111175, GSE25507, GSE42133) and
combines evidence using directional Fisher's meta-analysis.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, combine_pvalues
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA

# Cohorts and statistical thresholds
ALL_STUDIES = ["GSE26415", "GSE111175", "GSE25507", "GSE42133"]
PRIMARY_STUDIES = ["GSE26415", "GSE111175", "GSE42133"]
FDR_CUTOFF = 0.05
FC_CUTOFF = 0.50

# Output directories
OUT_DIR = Path("step1_DEG")
DEG_DIR = OUT_DIR / "deg"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"

for d in [DEG_DIR, PLOT_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Visual styling
PALETTE = {
    "ASD": "#E63946",
    "Control": "#2A9D8F",
    "Up": "#E63946",
    "Down": "#1D70B8",
    "NS": "#CBD5E1"
}
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False
})


# -------------------------------------------------------------------------
# 1. Load expression data and metadata
# -------------------------------------------------------------------------
def load_study(study_id):
    """Load gene-level expression matrix and sample diagnostic labels (0=Control, 1=ASD)."""
    study_dir = Path(f"{study_id}_data") / "processed"
    expr_file = study_dir / "expression_gene_level.csv"
    meta_file = study_dir / "sample_metadata.csv"

    if not expr_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Missing data files for {study_id}: {expr_file} / {meta_file}")

    expr = pd.read_csv(expr_file, index_col=0)
    meta = pd.read_csv(meta_file, index_col=0)

    # Retain samples with confirmed binary diagnosis
    valid_samples = meta[meta["label"].isin([0, 1])].index
    common = [s for s in expr.columns if s in valid_samples]
    
    expr = expr[common].copy()
    y = meta.loc[common, "label"].astype(int).copy()

    n_ctrl = int((y == 0).sum())
    n_asd = int((y == 1).sum())
    print(f"Loaded {study_id}: {expr.shape[0]} genes, {len(common)} samples (ASD: {n_asd}, Control: {n_ctrl})")
    return expr, y


# -------------------------------------------------------------------------
# 2. Scale checking and quantile normalization
# -------------------------------------------------------------------------
def quantile_normalize(df):
    """Quantile normalizes sample columns to have matching empirical distributions."""
    vals = df.values.astype(float)
    order = np.argsort(vals, axis=0)
    sorted_vals = np.take_along_axis(vals, order, axis=0)
    mean_dist = np.nanmean(sorted_vals, axis=1)
    ranks = np.argsort(order, axis=0)
    return pd.DataFrame(np.take(mean_dist, ranks), index=df.index, columns=df.columns)


def preprocess_study(expr):
    """Checks whether values are raw or log2, transforms if needed, and applies quantile normalization."""
    vals = expr.values.astype(float)
    p99 = float(np.nanpercentile(vals, 99))
    max_val = float(np.nanmax(vals))
    med_val = float(np.nanmedian(vals))
    min_val = float(np.nanmin(vals))

    is_raw = (p99 > 30.0) or (max_val > 50.0) or (med_val > 20.0 and min_val >= 0)
    data = expr.astype(float).copy()

    if is_raw:
        print("  Applying log2(x + 1) transformation for linear scale values")
        data = np.log2(np.clip(data, 0, None) + 1.0)

    data = quantile_normalize(data)
    return data


# -------------------------------------------------------------------------
# 3. Differential expression analysis using limma
# -------------------------------------------------------------------------
def run_limma(expr, y, study_id):
    """Runs empirical Bayes linear model in R limma (model: ~ group, reference: Control)."""
    tmp_dir = DEG_DIR / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    expr_file = tmp_dir / f"{study_id}_expr.csv"
    meta_file = tmp_dir / f"{study_id}_meta.csv"
    out_file = tmp_dir / f"{study_id}_limma.csv"
    r_file = tmp_dir / f"limma_{study_id}.R"

    expr.to_csv(expr_file)
    meta_df = pd.DataFrame({
        "sample": y.index,
        "group": y.map({0: "Control", 1: "ASD"}).values
    }).set_index("sample")
    meta_df.to_csv(meta_file)

    r_script = f"""
    suppressPackageStartupMessages(library(limma))
    expr <- as.matrix(read.csv("{expr_file.as_posix()}", row.names=1, check.names=FALSE))
    meta <- read.csv("{meta_file.as_posix()}", row.names=1, check.names=FALSE)
    meta$group <- factor(meta$group, levels=c("Control", "ASD"))

    common <- intersect(colnames(expr), rownames(meta))
    expr <- expr[, common, drop=FALSE]
    meta <- meta[common, , drop=FALSE]

    design <- model.matrix(~ group, data=meta)
    fit <- lmFit(expr, design)
    fit <- eBayes(fit)
    tt <- topTable(fit, coef="groupASD", number=Inf, sort.by="none")

    mean_asd <- rowMeans(expr[, rownames(meta)[meta$group == "ASD"], drop=FALSE], na.rm=TRUE)
    mean_ctrl <- rowMeans(expr[, rownames(meta)[meta$group == "Control"], drop=FALSE], na.rm=TRUE)

    res <- data.frame(
        gene = rownames(tt),
        mean_ASD = mean_asd[rownames(tt)],
        mean_Control = mean_ctrl[rownames(tt)],
        log2FC = mean_asd[rownames(tt)] - mean_ctrl[rownames(tt)],
        t_stat = tt$t,
        p_value = tt$P.Value,
        adj_p_val = tt$adj.P.Val,
        stringsAsFactors=FALSE
    )
    write.csv(res, "{out_file.as_posix()}", row.names=FALSE)
    """
    r_file.write_text(r_script, encoding="utf-8")

    proc = subprocess.run(["Rscript", str(r_file)], capture_output=True, text=True)
    if proc.returncode != 0 or not out_file.exists():
        raise RuntimeError(f"R limma failed for {study_id}: {proc.stderr}")

    deg_df = pd.read_csv(out_file)

    # Classify regulation direction based on FDR < 0.05
    deg_df["neg_log10_fdr"] = -np.log10(deg_df["adj_p_val"].clip(lower=1e-300))
    is_sig = deg_df["adj_p_val"] < FDR_CUTOFF
    deg_df["regulation"] = "NS"
    deg_df.loc[is_sig & (deg_df["log2FC"] > 0), "regulation"] = "Up"
    deg_df.loc[is_sig & (deg_df["log2FC"] < 0), "regulation"] = "Down"

    deg_df.sort_values("adj_p_val", inplace=True)
    deg_df.reset_index(drop=True, inplace=True)
    return deg_df


# -------------------------------------------------------------------------
# 4. Cross-study directional Fisher meta-analysis
# -------------------------------------------------------------------------
def run_meta_analysis(studies, label):
    """Combines per-study p-values using directional Fisher's method."""
    print(f"Running {label} ({', '.join(studies)})...")
    dfs = []
    for s in studies:
        f = DEG_DIR / s / "DEG_full_results.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)[["gene", "p_value", "adj_p_val", "log2FC"]].copy()
        d.columns = ["gene", f"p_value_{s}", f"adj_p_val_{s}", f"log2FC_{s}"]
        dfs.append(d)

    if len(dfs) < 2:
        return None

    meta = dfs[0]
    for d in dfs[1:]:
        meta = meta.merge(d, on="gene", how="outer")

    def combine_gene(row):
        p_up, p_dn, fc_list = [], [], []
        for s in studies:
            p = row.get(f"p_value_{s}")
            fc = row.get(f"log2FC_{s}")
            if pd.notna(p) and pd.notna(fc):
                p = max(float(p), 1e-300)
                fc_list.append(float(fc))
                p_up.append(p / 2.0 if fc > 0 else 1.0 - p / 2.0)
                p_dn.append(p / 2.0 if fc < 0 else 1.0 - p / 2.0)

        if len(fc_list) < 2:
            return pd.Series([np.nan, np.nan, "NS"])

        _, meta_up = combine_pvalues(p_up, method="fisher")
        _, meta_dn = combine_pvalues(p_dn, method="fisher")

        meta_p = min(meta_up, meta_dn) * 2.0
        meta_p = min(meta_p, 1.0)
        mean_fc = float(np.mean(fc_list))
        direction = "Up" if meta_up < meta_dn else "Down"
        return pd.Series([meta_p, mean_fc, direction])

    res = meta.apply(combine_gene, axis=1)
    res.columns = ["meta_p_value", "meta_log2FC", "meta_direction"]
    meta = pd.concat([meta, res], axis=1)

    valid = meta["meta_p_value"].notna()
    meta.loc[valid, "meta_adj_p_val"] = multipletests(meta.loc[valid, "meta_p_value"], method="fdr_bh")[1]
    meta["is_meta_sig"] = meta["meta_adj_p_val"] < FDR_CUTOFF
    meta.sort_values("meta_adj_p_val", inplace=True)
    meta.reset_index(drop=True, inplace=True)
    return meta


# -------------------------------------------------------------------------
# 5. Publication figures (Figures 2 - 5)
# -------------------------------------------------------------------------
def plot_volcano(deg_df, study_id):
    """Figure 2: Volcano plot highlighting up- and down-regulated DEGs."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for reg, col, label, sz in [("NS", PALETTE["NS"], "Not Significant", 14),
                               ("Down", PALETTE["Down"], "Downregulated", 30),
                               ("Up", PALETTE["Up"], "Upregulated", 30)]:
        m = deg_df["regulation"] == reg
        ax.scatter(deg_df.loc[m, "log2FC"], deg_df.loc[m, "neg_log10_fdr"],
                   c=col, s=sz, alpha=0.75, label=f"{label} (n={m.sum():,})", edgecolors="none")

    ax.axhline(-np.log10(FDR_CUTOFF), color="#64748B", linestyle="--", lw=1.0)
    ax.set_xlabel("Log2 Fold Change (ASD vs Control)", fontsize=11, fontweight="bold")
    ax.set_ylabel("-Log10 Adjusted P-value (FDR)", fontsize=11, fontweight="bold")
    ax.set_title(f"Figure 2. Differentially Expressed Genes: {study_id}", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"Figure_2_volcano_{study_id}.png")
    plt.close()


def plot_pca(expr, y, study_id):
    """Figure 3: PCA plot of sample-level transcriptomic variance."""
    pca = PCA(n_components=2, random_state=42)
    top_genes = expr.var(axis=1).nlargest(1000).index
    coords = pca.fit_transform(expr.loc[top_genes].T)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for grp, lbl, col in [(0, "Control", PALETTE["Control"]), (1, "ASD", PALETTE["ASD"])]:
        m = y.values == grp
        ax.scatter(coords[m, 0], coords[m, 1], c=col, label=f"{lbl} (n={m.sum()})", s=65, alpha=0.85, edgecolors="white", lw=0.6)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Figure 3. PCA Transcriptomic Structure: {study_id}", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"Figure_3_PCA_{study_id}.png")
    plt.close()


def plot_heatmap(expr, y, deg_df, study_id):
    """Figure 4: Standardized expression heatmap for top DEGs."""
    sig_genes = deg_df[deg_df["adj_p_val"] < FDR_CUTOFF]["gene"].head(50).tolist()
    if len(sig_genes) < 5:
        sig_genes = deg_df.nsmallest(50, "adj_p_val")["gene"].tolist()

    sub_expr = expr.loc[sig_genes].apply(zscore, axis=1)
    col_order = y.sort_values().index
    sub_expr = sub_expr[col_order]

    col_colors = y.loc[col_order].map({0: PALETTE["Control"], 1: PALETTE["ASD"]})
    g = sns.clustermap(sub_expr, col_colors=col_colors, cmap="vlag", center=0, col_cluster=False,
                       figsize=(10, 8), yticklabels=True, xticklabels=False, cbar_kws={"label": "Z-Score"})
    g.fig.suptitle(f"Figure 4. Top DEG Standardized Expression: {study_id}", fontsize=12, fontweight="bold", y=1.02)
    plt.savefig(PLOT_DIR / f"Figure_4_heatmap_{study_id}.png", bbox_inches="tight")
    plt.close()


def plot_bar(deg_df, study_id):
    """Figure 5: Bar plot of top upregulated and downregulated DEGs."""
    top_up = deg_df[deg_df["regulation"] == "Up"].head(10)
    top_dn = deg_df[deg_df["regulation"] == "Down"].head(10)
    top_genes = pd.concat([top_up, top_dn]).sort_values("log2FC")

    if top_genes.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = top_genes["regulation"].map({"Up": PALETTE["Up"], "Down": PALETTE["Down"]})
    ax.barh(top_genes["gene"], top_genes["log2FC"], color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Log2 Fold Change (ASD vs Control)", fontsize=11, fontweight="bold")
    ax.set_title(f"Figure 5. Top Significant DEGs: {study_id}", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"Figure_5_bar_{study_id}.png")
    plt.close()


# -------------------------------------------------------------------------
# 6. Main execution
# -------------------------------------------------------------------------
def main():
    print("Starting Step 1 differential expression analysis...")
    summary_rows = []

    for study in ALL_STUDIES:
        print(f"\nProcessing {study}...")
        expr, y = load_study(study)
        expr_norm = preprocess_study(expr)
        deg_df = run_limma(expr_norm, y, study)

        # Save per-study DEG files
        study_out = DEG_DIR / study
        study_out.mkdir(parents=True, exist_ok=True)
        deg_df.to_csv(study_out / "DEG_full_results.csv", index=False)
        deg_df[deg_df["adj_p_val"] < FDR_CUTOFF].to_csv(study_out / "DEG_significant.csv", index=False)
        deg_df[deg_df["regulation"] == "Up"].to_csv(study_out / "DEG_upregulated.csv", index=False)
        deg_df[deg_df["regulation"] == "Down"].to_csv(study_out / "DEG_downregulated.csv", index=False)

        # Generate figures
        plot_volcano(deg_df, study)
        plot_pca(expr_norm, y, study)
        plot_heatmap(expr_norm, y, deg_df, study)
        plot_bar(deg_df, study)

        n_sig = int((deg_df["adj_p_val"] < FDR_CUTOFF).sum())
        n_up = int((deg_df["regulation"] == "Up").sum())
        n_down = int((deg_df["regulation"] == "Down").sum())
        summary_rows.append({
            "GEO Cohort Dataset": study,
            "Total Genes Tested": len(deg_df),
            "Significant DEGs (FDR < 0.05)": n_sig,
            "Upregulated in ASD": n_up,
            "Downregulated in ASD": n_down
        })
        print(f"  {study} complete: {len(deg_df)} genes, {n_sig} significant DEGs (Up: {n_up}, Down: {n_down})")

    # Table 5: Differential expression summary
    table_5 = pd.DataFrame(summary_rows)
    table_5.to_csv(TABLE_DIR / "Table_5_Differential_Expression_Summary.csv", index=False)
    print("Saved Table 5 (Differential expression summary)")

    # Meta-analyses
    meta_primary = run_meta_analysis(PRIMARY_STUDIES, label="Primary Meta-Analysis (GSE26415, GSE111175, GSE42133)")
    meta_sensitivity = run_meta_analysis(ALL_STUDIES, label="4-Cohort Sensitivity Meta-Analysis")

    if meta_primary is not None and meta_sensitivity is not None:
        meta_primary.to_csv(DEG_DIR / "Primary_Meta_Analysis_Results.csv", index=False)
        meta_sensitivity.to_csv(DEG_DIR / "Sensitivity_4Study_Meta_Analysis_Results.csv", index=False)

        table_6 = pd.DataFrame([
            {
                "Meta-Analysis Configuration": "Primary Meta-Analysis",
                "Cohorts Included": ", ".join(PRIMARY_STUDIES),
                "Total Samples (N)": 287,
                "Meta-Significant DEGs (FDR < 0.05)": int(meta_primary["is_meta_sig"].sum())
            },
            {
                "Meta-Analysis Configuration": "4-Study Sensitivity Meta-Analysis",
                "Cohorts Included": ", ".join(ALL_STUDIES),
                "Total Samples (N)": 433,
                "Meta-Significant DEGs (FDR < 0.05)": int(meta_sensitivity["is_meta_sig"].sum())
            }
        ])
        table_6.to_csv(TABLE_DIR / "Table_6_Meta_Analysis_Summary.csv", index=False)
        print("Saved Table 6 (Meta-analysis summary)")

        table_7 = pd.DataFrame([
            {"Metric": "Consistent Upregulated", "3-Cohort Count": int((meta_primary["is_meta_sig"] & (meta_primary["meta_direction"] == "Up")).sum())},
            {"Metric": "Consistent Downregulated", "3-Cohort Count": int((meta_primary["is_meta_sig"] & (meta_primary["meta_direction"] == "Down")).sum())}
        ])
        table_7.to_csv(TABLE_DIR / "Table_7_Directional_Consistency.csv", index=False)
        print("Saved Table 7 (Directional consistency summary)")

    # Supplementary Tables S3, S4, S5
    for study_id, fname in [("GSE26415", "Supplementary_Table_S3_GSE26415_DEGs.csv"),
                            ("GSE111175", "Supplementary_Table_S4_GSE111175_DEGs.csv"),
                            ("GSE42133", "Supplementary_Table_S5_GSE42133_DEGs.csv")]:
        f_sig = DEG_DIR / study_id / "DEG_significant.csv"
        if f_sig.exists():
            pd.read_csv(f_sig).to_csv(TABLE_DIR / fname, index=False)
            print(f"Saved {fname}")

    print("\nStep 1 analysis completed successfully.")


if __name__ == "__main__":
    main()

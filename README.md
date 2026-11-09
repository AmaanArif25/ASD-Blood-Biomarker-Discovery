# Integrated Transcriptomic Meta-Analysis and Machine Learning Reveal Reliable Blood Biomarkers for Autism Spectrum Disorder

[![Research](https://img.shields.io/badge/Research-Computational%20Biology-blue)](https://github.com)
[![Analysis](https://img.shields.io/badge/Analysis-Transcriptomics-green)](https://github.com)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Explainable%20AI-orange)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Autism Spectrum Disorder (ASD) is a heterogeneous neurodevelopmental condition characterized by substantial molecular and biological variability. Although transcriptomic studies have reported numerous ASD-associated gene expression changes, identifying reproducible molecular signatures across independent cohorts remains challenging.

This project presents an integrated computational framework combining cross-study transcriptomic meta-analysis, protein-protein interaction (PPI) network analysis, and explainable machine learning to prioritize reproducible blood-based gene expression biomarkers associated with ASD.

Rather than relying on differential expression findings from individual studies, the workflow emphasizes cross-cohort concordance, independent dataset evaluation, biologically informed biomarker prioritization, and leakage-controlled machine learning.

The primary objective is to identify candidate molecular signatures that remain informative across independent transcriptomic datasets and provide insights into the biological processes potentially associated with ASD.

## Research Objectives

The study addresses the following research questions.

1. Which blood transcriptomic alterations are reproducibly associated with ASD across independent microarray cohorts?
2. Can cross-study meta-analysis distinguish consistent molecular signals from cohort-specific differential expression?
3. Which candidate genes occupy central positions in ASD-associated protein interaction networks?
4. Can machine learning identify compact gene expression panels that discriminate ASD samples from controls across independent datasets?
5. Which molecular features contribute most consistently to predictive performance, and what biological processes do these genes represent?

## Datasets

Four publicly available transcriptomic datasets from the NCBI Gene Expression Omnibus (GEO) were investigated.

| GEO accession | Dataset |
|---|---|
| GSE26415 | ASD-associated blood transcriptomic study |
| GSE111175 | ASD-associated blood transcriptomic study |
| GSE25507 | ASD-associated blood transcriptomic study |
| GSE42133 | ASD-associated blood transcriptomic study |

All datasets underwent phenotype harmonization, quality assessment, probe-to-gene mapping, normalization, and differential expression analysis.

GSE25507 was retained for sensitivity analyses but excluded from the primary machine learning analysis because of its limited differential-expression signal and substantial discordance with the other cohorts.

The primary machine learning analysis included three datasets comprising 287 samples, including 140 ASD samples and 147 controls, with 9,755 intersecting genes.

Original expression datasets are available through NCBI GEO using their respective accession numbers.

## Computational Methodology

### 1. Transcriptomic Data Processing

A standardized preprocessing workflow was implemented to improve comparability across independent microarray studies.

The processing strategy included:

- GEO expression and phenotype data retrieval.
- Sample-level phenotype harmonization.
- Quality control and expression data assessment.
- Probe-to-gene annotation and mapping.
- Dataset-specific normalization.
- Differential gene expression analysis using the limma framework.

### 2. Differential Gene Expression and Meta-Analysis

Differential expression analysis was conducted independently within each cohort.

Cross-study concordance was subsequently evaluated using a directional Fisher's method-based meta-analysis to prioritize genes showing consistent directional changes across studies.

This approach was designed to reduce dependence on individual cohort findings and identify more reproducible ASD-associated transcriptomic signals.

### 3. Protein-Protein Interaction Network Analysis

Candidate genes were examined within protein-protein interaction networks to investigate their potential functional relationships and network-level importance.

Network prioritization incorporated multiple topological measures, including:

- Maximal Clique Centrality (MCC).
- Degree centrality.
- Betweenness centrality.

These complementary measures were used to prioritize potential hub genes and identify molecular candidates for further investigation.

### 4. Machine Learning-Based Biomarker Identification

Five classification algorithms were evaluated.

- Logistic Regression.
- Elastic Net Logistic Regression.
- Linear Support Vector Machine.
- Random Forest.
- Extreme Gradient Boosting (XGBoost).

To assess generalization across independent transcriptomic cohorts, a Leave-One-Dataset-Out (LODO) validation strategy was implemented.

For each validation fold, one dataset was held out for independent evaluation while the remaining datasets were used for model development.

Importantly, feature selection, hyperparameter optimization, and classification-threshold selection were restricted to the training data within each LODO fold to minimize information leakage.

Compact biomarker panels containing approximately 17–19 genes per fold were evaluated.

### 5. Model Evaluation and Explainability

Predictive performance was assessed using complementary classification metrics, including:

- Receiver Operating Characteristic Area Under the Curve (ROC-AUC).
- Precision-Recall Area Under the Curve (PR-AUC).

Cross-fold feature recurrence and model interpretability were used to prioritize candidate biomarkers with consistent predictive contributions.

### 6. Functional Interpretation

Prioritized genes were examined through functional enrichment and biological interpretation to investigate their involvement in molecular processes potentially associated with ASD.

The analysis highlighted biological themes involving:

- Cytosolic and mitochondrial translation.
- Ribosomal function.
- Protein homeostasis and proteostasis.
- Wnt signaling.
- Mitochondrial bioenergetics.

## Key Findings

The integrated analysis prioritized several candidate blood transcriptomic biomarkers, including:

**CSNK1G2, SETD1B, CBL, RPS14, MRPL34, MRPL22, SPTAN1, MYH9, and PSMA6.**

The machine learning framework achieved cross-cohort discrimination of up to:

| Performance metric | Result |
|---|---:|
| ROC-AUC | 0.760 |
| PR-AUC | 0.777 |
| Biomarker panel size | 17–19 genes per fold |
| Primary ML samples | 287 |

These findings suggest that integrating transcriptomic concordance, network topology, and leakage-controlled machine learning can help prioritize candidate molecular signatures across heterogeneous ASD cohorts.

The identified biomarkers are exploratory and should not be interpreted as clinically validated diagnostic markers.

## Repository Contents

This repository contains the computational scripts and supporting documentation associated with the study.

The code is organized according to the analytical workflow, covering transcriptomic preprocessing, differential expression, cross-study meta-analysis, network analysis, machine learning, and functional interpretation.

The exact file structure and execution instructions should be updated to reflect the scripts included in this repository.

## Reproducibility and Data Availability

The original transcriptomic datasets are publicly accessible through the NCBI Gene Expression Omnibus (GEO) under the following accession numbers:

- GSE26415
- GSE111175
- GSE25507
- GSE42133

Supplementary figures, figure captions, supplementary tables, and table captions are deposited in Zenodo.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.24089826

The DOI corresponds to the supplementary materials associated with this research.

Analysis scripts are provided in this GitHub repository to support methodological transparency and computational reproducibility.

## Citation

If you use this workflow, computational framework, or associated findings in your research, please cite the manuscript.

Arif, A., Emmert, S., Boeckmann, L., & Singh, K. P. (2026). *Integrated Transcriptomic Meta-Analysis and Machine Learning Reveal Reliable Blood Biomarkers for Autism Spectrum Disorder.* Manuscript submitted / preprint details to be updated upon publication.

Please replace the citation status with the final bioRxiv or journal citation once available.

## Authors

**Amaan Arif**  
Integral University, Lucknow, Uttar Pradesh, India

**Steffen Emmert**  
University Medical Center Rostock, Germany

**Lars Boeckmann**  
University Medical Center Rostock, Germany

**Krishna P. Singh**  
University Medical Center Rostock, Germany

## Acknowledgements

We acknowledge the researchers who generated and made the publicly available GEO transcriptomic datasets accessible to the research community.

## License

This repository is intended to be distributed under the MIT License, subject to approval by the manuscript authors.

## Disclaimer

The computational findings and candidate biomarkers reported in this repository are intended for research purposes. They do not constitute clinically validated diagnostic biomarkers for Autism Spectrum Disorder. Independent validation in larger and clinically diverse cohorts is necessary.

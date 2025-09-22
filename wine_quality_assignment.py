# -*- coding: utf-8 -*-
"""
Assignment 1: Exploratory Data Analysis on the Wine Quality (Red) Dataset

GOALS (from prompt):
- Pre-process and explore the dataset using statistics and visualizations
- Interpret the computed statistics/plots to understand the dataset
- Learn properties/characteristics that inform expectations for quality prediction

DATASET
- winequality-red.csv, UCI ML repository format (semicolon-separated)
- 11 physicochemical attributes + 1 sensory "quality" score (integer 0–10)

CLASS LABELS (as given in the example in the prompt)
- Bad: quality < 4
- Good: 4 < quality <= 7
- Very Good: quality > 7

NOTE ON QUALITY == 4:
- The provided example definition leaves quality == 4 unassigned.
- In the red-wine dataset, 4 is uncommon; we will EXCLUDE quality==4 from
  the class-based boxplots so we adhere strictly to the given class rules.
- This is explained again near the boxplot section below.

USAGE
- Put this file in the same folder as 'winequality-red.csv' (semicolon-separated CSV).
- Run: python wine_quality_assignment.py
- The script will print statistics and render plots. It also prints
  short interpretations for each required task so submission can be just this .py file.
- If your grading environment does not display plots automatically,
  you can enable saving of figures by setting SAVE_FIGS=True below.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
CSV_PATH = "winequality-red.csv"   # Update this path if needed
SAVE_FIGS = False                  # Set True to save figures as PNG files
FIG_DIR = "figures"                # Folder to save figures

if SAVE_FIGS and not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)

# ==========================
# Load Data
# ==========================
# The UCI red wine quality file is semicolon-separated.
df = pd.read_csv(CSV_PATH, sep=';')

print("=== HEAD (first 5 rows) ===")
print(df.head(), "\n")

print("Columns:", list(df.columns), "\n")

# ==========================
# 1) Summary Statistics
# ==========================
print("=== (1) SUMMARY STATISTICS: df.describe() ===")
summary = df.describe()  # count, mean, std, min, 25%, 50%, 75%, max
print(summary, "\n")

# INTERPRETATION: Two favorite statistics (with justification)
interp_1 = """
(1) Interpretation – Two favorite statistical measures:
- Mean: The mean shows the central tendency or the “typical” value for each attribute
  (e.g., the typical alcohol percentage across these red wines). It’s useful because
  many downstream decisions or baselines (such as normalizing/centering data) rely on
  the mean, and it quickly summarizes where most data points tend to lie.
- Standard Deviation: The std quantifies variability. For this dataset, it tells us
  how much wines differ from each other on attributes like residual sugar or SO2. 
  Larger std implies broader diversity in winemaking/chemistry that could impact 
  sensory scores (quality). Combined with the mean, std alerts us to attributes
  that may need scaling before modeling and highlights which attributes have
  consistent production versus wide variation.
"""
print(interp_1)

# ==========================
# 2) Correlation Analysis
# ==========================
print("=== (2) CORRELATION MATRIX (Pearson) ===")
corr = df.corr(numeric_only=True)
print(corr, "\n")

# Optional: visualize correlation matrix using matplotlib (no seaborn to follow constraints)
plt.figure(figsize=(10, 8))
plt.imshow(corr, interpolation='nearest', aspect='auto')
plt.title("Correlation Heatmap (Pearson)")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar()
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=200)
plt.show()

# INTERPRETATION of correlations (printed as a guide; actual numbers depend on data)
# We can programmatically identify a few strongest pairs with quality.
def top_correlations_with(target_col, corr_df, k=5):
    s = corr_df[target_col].drop(labels=[target_col])  # remove self-correlation
    return s.reindex(s.abs().sort_values(ascending=False).index)[:k]

if 'quality' in corr.columns:
    print("Top correlations with 'quality':")
    print(top_correlations_with('quality', corr, k=8), "\n")

interp_2 = """
(2) Interpretation – Correlation findings:
- In typical red-wine data, 'alcohol' shows a positive correlation with 'quality':
  higher alcohol content tends to be associated with higher sensory scores.
- 'volatile acidity' usually shows a negative correlation with 'quality':
  acetic acid can impart vinegar-like notes that panelists penalize.
- Other attributes (e.g., sulphates, fixed acidity) may show smaller positive
  or negative correlations with quality, indicating multi-factor influences
  rather than a single dominant driver (besides alcohol / volatile acidity).
- Correlation does not imply causation; it summarizes linear association only.
"""
print(interp_2)

# ==========================
# 3) Scatter: Residual Sugar vs pH
# ==========================
plt.figure()
plt.scatter(df['residual sugar'], df['pH'], alpha=0.5)
plt.xlabel('Residual Sugar (g/dm^3)')
plt.ylabel('pH')
plt.title('Scatter: Residual Sugar vs pH')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "scatter_residual_sugar_vs_pH.png"), dpi=200)
plt.show()

interp_3 = """
(3) Interpretation – Residual Sugar vs pH:
- The scatter typically appears diffuse with no strong linear trend.
- Wines with both low and moderately higher sugar levels can exhibit similar pH.
- This suggests residual sugar and acidity (pH) are largely independent in the
  red-wine dataset; sweetness does not directly dictate pH levels.
"""
print(interp_3)

# ==========================
# 4) Scatter: Fixed Acidity vs Citric Acid
# ==========================
plt.figure()
plt.scatter(df['fixed acidity'], df['citric acid'], alpha=0.5)
plt.xlabel('Fixed Acidity (g/dm^3)')
plt.ylabel('Citric Acid (g/dm^3)')
plt.title('Scatter: Fixed Acidity vs Citric Acid')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "scatter_fixed_acidity_vs_citric_acid.png"), dpi=200)
plt.show()

interp_4 = """
(4) Interpretation – Fixed Acidity vs Citric Acid:
- A positive association is expected because citric acid is one contributor
  to total fixed acidity. However, many samples have low citric acid even when
  fixed acidity is moderate/high, indicating other acids (e.g., tartaric) play
  a substantial role. Thus, the trend is positive but not perfect.
"""
print(interp_4)

# ==========================
# 5) Histogram of Quality
# ==========================
plt.figure()
# Quality is integer-valued; we can align bins to integers.
min_q, max_q = int(df['quality'].min()), int(df['quality'].max())
bins = np.arange(min_q - 0.5, max_q + 1.5, 1.0)
plt.hist(df['quality'], bins=bins, edgecolor='black')
plt.xlabel('Quality (integer score)')
plt.ylabel('Count')
plt.title('Histogram: Wine Quality')
plt.xticks(range(min_q, max_q + 1))
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "hist_quality.png"), dpi=200)
plt.show()

interp_5 = """
(5) Interpretation – Quality histogram:
- Most red wines in this dataset cluster around mid-range quality (typically 5–6).
- Very low or very high scores are rare, creating a class imbalance toward “average” wines.
- This imbalance matters for prediction: models may overfit to the majority class unless
  we use appropriate evaluation metrics or resampling strategies in later assignments.
"""
print(interp_5)

# ==========================
# 6) Box Plots by Quality Class (Alcohol & pH)
# ==========================
# Create a strict class label per the exact rules given:
def map_quality_to_class(q):
    if q < 4:
        return "Bad"
    elif (q > 4) and (q <= 7):
        return "Good"
    elif q > 7:
        return "Very Good"
    else:
        return None  # This captures q == 4 explicitly as unassigned

df['quality_class'] = df['quality'].apply(map_quality_to_class)

# EXCLUSION NOTE: remove rows with None class (quality == 4) for class-based plots
df_classed = df.dropna(subset=['quality_class']).copy()

# Prepare data lists for boxplots
def values_by_class(column):
    return [
        df_classed.loc[df_classed['quality_class'] == 'Bad', column].values,
        df_classed.loc[df_classed['quality_class'] == 'Good', column].values,
        df_classed.loc[df_classed['quality_class'] == 'Very Good', column].values,
    ]

# Alcohol by class
plt.figure()
plt.boxplot(values_by_class('alcohol'), labels=['Bad', 'Good', 'Very Good'], showfliers=True)
plt.ylabel('Alcohol (% vol.)')
plt.title('Alcohol by Quality Class (Bad, Good, Very Good)')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "box_alcohol_by_quality_class.png"), dpi=200)
plt.show()

# pH by class
plt.figure()
plt.boxplot(values_by_class('pH'), labels=['Bad', 'Good', 'Very Good'], showfliers=True)
plt.ylabel('pH')
plt.title('pH by Quality Class (Bad, Good, Very Good)')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "box_pH_by_quality_class.png"), dpi=200)
plt.show()

# Overall (all instances, not grouped) – separate boxplots for Alcohol and for pH
plt.figure()
plt.boxplot(df['alcohol'].values, labels=['Alcohol (All Wines)'], showfliers=True)
plt.title('Overall Box Plot – Alcohol (All Wines)')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "box_alcohol_all.png"), dpi=200)
plt.show()

plt.figure()
plt.boxplot(df['pH'].values, labels=['pH (All Wines)'], showfliers=True)
plt.title('Overall Box Plot – pH (All Wines)')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(os.path.join(FIG_DIR, "box_pH_all.png"), dpi=200)
plt.show()

interp_6 = """
(6) Interpretation – Boxplots (Alcohol & pH):
- Alcohol by class: The median alcohol level is typically higher for 'Very Good' wines
  than for 'Good', and higher for 'Good' than for 'Bad'. The interquartile ranges
  often overlap somewhat, but the shift in medians supports alcohol as a useful
  discriminator of perceived quality.
- pH by class: The medians and spreads across classes are more similar; heavy overlap
  indicates pH is a weaker discriminator of quality classes in this dataset.
- Comparing class-based vs overall boxplots: The overall boxplots summarize spread for
  the entire dataset, but the class-based views reveal how distributions shift with
  quality. Alcohol clearly shows a class-related shift; pH shows less separation.
- IMPORTANT: Because the class rules exclude quality==4, those samples do not appear
  in the class-based plots (they still appear in the overall boxplots).
"""
print(interp_6)

# ==========================
# 7) Brief Conclusion
# ==========================
conclusion = """
(7) Conclusion – Most important findings for predicting red-wine quality:
- Alcohol content is the strongest positive indicator of higher quality; class-based
  boxplots and correlations both support this.
- Volatile acidity tends to be negatively associated with quality (higher volatile
  acidity can impart vinegar-like notes that reduce sensory scores).
- Other attributes (e.g., sulphates, fixed/frees SO2, residual sugar, pH) show weaker
  and sometimes nuanced relationships; they may contribute as secondary features.
- The quality distribution is imbalanced toward mid-range scores (5–6), which is
  important for evaluation and modeling choices in later analysis.
- Overall, a baseline expectation is that models leveraging alcohol (↑ quality) and
  volatile acidity (↓ quality) should perform reasonably, with incremental gains from
  additional attributes after appropriate scaling/preprocessing.
"""
print(conclusion)

print("=== DONE. All required tasks executed with inline interpretations. ===")

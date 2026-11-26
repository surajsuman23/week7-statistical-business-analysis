"""
Week 7 - Statistical Business Analysis

Reusable functions for descriptive statistics, distribution checks, correlation,
hypothesis testing, confidence intervals and regression. The notebook
(statistical_analysis.ipynb) imports this module, and `python stats_analysis.py`
regenerates every output (charts, results text file and PDF report).

All paths are relative to this file, so the project runs on any machine.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
BUSINESS_DATA = BASE_DIR / "business_data.csv"
SALES_DATA = BASE_DIR / "data" / "sales_data.csv"
ALPHA = 0.05
RANDOM_SEED = 42

BUSINESS_COLUMNS = [
    "CustomerID", "Tenure", "MonthlyCharges", "TotalCharges", "Contract",
    "PaymentMethod", "PaperlessBilling", "SeniorCitizen", "Churn",
]
SALES_COLUMNS = ["Date", "Product", "Quantity", "Price", "Customer_ID", "Region", "Total_Sales"]
BUSINESS_NUMERIC = ["Tenure", "MonthlyCharges", "TotalCharges"]
SALES_NUMERIC = ["Quantity", "Price", "Total_Sales"]

PALETTE = {"Retained": "#2E86AB", "Churned": "#C73E1D"}
sns.set_theme(style="whitegrid", context="notebook")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger("stats_analysis")


class DataValidationError(Exception):
    """Raised when an input file cannot be used for the analysis."""


# --------------------------------------------------------------------------- #
# 1. LOAD & VALIDATE
# --------------------------------------------------------------------------- #
def _read_csv(path, required_columns):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Data file is empty: {path}") from exc
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name}: missing required columns {missing}")
    if df.empty:
        raise DataValidationError(f"{path.name}: has headers but no rows")
    return df


def load_business_data(path=BUSINESS_DATA):
    """Load the 500-customer business dataset and coerce types."""
    df = _read_csv(path, BUSINESS_COLUMNS)
    for col in BUSINESS_NUMERIC + ["SeniorCitizen", "Churn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["CustomerID", "Contract", "PaymentMethod", "PaperlessBilling"]:
        df[col] = df[col].astype(str).str.strip()
    before = len(df)
    df = df.dropna(subset=BUSINESS_NUMERIC + ["Churn"]).drop_duplicates("CustomerID")
    if not df["Churn"].isin([0, 1]).all():
        raise DataValidationError("Churn must be 0 or 1")
    if (df[BUSINESS_NUMERIC] < 0).any().any():
        raise DataValidationError("Tenure and charges cannot be negative")
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d invalid/duplicate business rows", dropped)
    df["Status"] = df["Churn"].map({0: "Retained", 1: "Churned"})
    return df.reset_index(drop=True)


def load_sales_data(path=SALES_DATA):
    """Load the 100-order sales dataset."""
    df = _read_csv(path, SALES_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in SALES_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date"] + SALES_NUMERIC)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    if df.empty:
        raise DataValidationError("No valid sales rows")
    return df.reset_index(drop=True)


def total_charges_consistency(df, tolerance=0.10):
    """
    Compare TotalCharges with Tenure x MonthlyCharges (what a customer should
    have paid in total). Returns a summary dict and the per-row ratio.
    """
    expected = df["Tenure"] * df["MonthlyCharges"]
    ratio = df["TotalCharges"] / expected.replace(0, np.nan)
    within = (ratio - 1).abs() <= tolerance
    summary = {
        "rows": int(len(df)),
        "within_tolerance": int(within.sum()),
        "pct_within_tolerance": round(100 * within.mean(), 1),
        "median_ratio": round(float(ratio.median()), 3),
        "min_ratio": round(float(ratio.min()), 3),
        "max_ratio": round(float(ratio.max()), 3),
        "corr_total_vs_expected": (round(float(df["TotalCharges"].corr(expected)), 3)
                                   if expected.std() > 0 and df["TotalCharges"].std() > 0 else float("nan")),
    }
    return summary, ratio


# --------------------------------------------------------------------------- #
# 2. DESCRIPTIVE STATISTICS & DISTRIBUTIONS
# --------------------------------------------------------------------------- #
def descriptive_stats(df, columns):
    """Mean, median, mode, spread and shape statistics for each column."""
    rows = {}
    for col in columns:
        s = df[col].dropna()
        rows[col] = {
            "count": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "mode": s.mode().iloc[0],
            "std": s.std(ddof=1),
            "variance": s.var(ddof=1),
            "min": s.min(),
            "q1": s.quantile(0.25),
            "q3": s.quantile(0.75),
            "iqr": s.quantile(0.75) - s.quantile(0.25),
            "max": s.max(),
            "skewness": stats.skew(s, bias=False),
            "kurtosis": stats.kurtosis(s, bias=False),
            "cv_%": 100 * s.std(ddof=1) / s.mean() if s.mean() else np.nan,
        }
    return pd.DataFrame(rows).T


def normality_tests(df, columns, alpha=ALPHA):
    """Shapiro-Wilk and D'Agostino-Pearson tests for each column."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        sw = stats.shapiro(s)
        dp = stats.normaltest(s)
        rows.append({
            "variable": col,
            "shapiro_W": sw.statistic,
            "shapiro_p": sw.pvalue,
            "dagostino_K2": dp.statistic,
            "dagostino_p": dp.pvalue,
            "normal_at_5%": bool(sw.pvalue > alpha and dp.pvalue > alpha),
        })
    return pd.DataFrame(rows).set_index("variable")


# --------------------------------------------------------------------------- #
# 3. CORRELATION
# --------------------------------------------------------------------------- #
def correlation_table(df, columns, method="pearson"):
    """All pairwise correlations with p-values (pearson or spearman)."""
    func = stats.pearsonr if method == "pearson" else stats.spearmanr
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            res = func(df[a], df[b])
            r = float(res.statistic)
            rows.append({
                "var_1": a, "var_2": b, "r": r, "p_value": float(res.pvalue),
                "strength": correlation_strength(r),
                "significant": bool(res.pvalue < ALPHA),
            })
    return pd.DataFrame(rows)


def correlation_strength(r):
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "negligible"


# --------------------------------------------------------------------------- #
# 4. HYPOTHESIS TESTS
# --------------------------------------------------------------------------- #
@dataclass
class TestResult:
    name: str
    test: str
    h0: str
    h1: str
    statistic: float
    p_value: float
    effect_size: float
    effect_label: str
    conclusion: str
    details: dict = field(default_factory=dict)
    alpha: float = ALPHA

    @property
    def reject_h0(self):
        return self.p_value < self.alpha


def cohens_d(a, b):
    """Cohen's d using the pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def eta_squared(groups):
    allv = np.concatenate([np.asarray(g, float) for g in groups])
    grand = allv.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_total = ((allv - grand) ** 2).sum()
    return float(ss_between / ss_total)


def _effect_word(value, small, medium, large):
    v = abs(value)
    if v >= large:
        return "large"
    if v >= medium:
        return "medium"
    if v >= small:
        return "small"
    return "negligible"


def welch_ttest(df, value, group, a, b, name):
    """Welch's two-sample t-test with a Mann-Whitney U check."""
    x = df.loc[df[group] == a, value]
    y = df.loc[df[group] == b, value]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each group needs at least 2 observations")
    t = stats.ttest_ind(x, y, equal_var=False)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided")
    d = cohens_d(x, y)
    reject = t.pvalue < ALPHA
    conclusion = (
        f"{'Reject' if reject else 'Fail to reject'} H0. Mean {value} is {x.mean():.1f} for {a} vs "
        f"{y.mean():.1f} for {b} (difference {x.mean() - y.mean():+.1f}); "
        f"effect size is {_effect_word(d, 0.2, 0.5, 0.8)}. "
        f"Mann-Whitney U {'agrees' if (mw.pvalue < ALPHA) == reject else 'disagrees'} (p = {fmt_p(mw.pvalue)})."
    )
    return TestResult(
        name=name, test="Welch two-sample t-test",
        h0=f"Mean {value} is the same for {a} and {b} customers",
        h1=f"Mean {value} differs between {a} and {b} customers",
        statistic=float(t.statistic), p_value=float(t.pvalue),
        effect_size=float(d), effect_label="Cohen's d", conclusion=conclusion,
        details={"n_" + a: len(x), "n_" + b: len(y), "mean_" + a: x.mean(), "mean_" + b: y.mean(),
                 "df": float(t.df), "mann_whitney_U": float(mw.statistic), "mann_whitney_p": float(mw.pvalue)},
    )


def chi_square_test(df, row, col, name):
    """Chi-square test of independence between two categorical variables."""
    table = pd.crosstab(df[row], df[col])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    v = cramers_v(table)
    reject = p < ALPHA
    rates = ""
    if set(table.columns) == {0, 1}:
        pct = (table[1] / table.sum(axis=1) * 100).round(1)
        rates = " Churn rate: " + ", ".join(f"{k} {val}%" for k, val in pct.items()) + "."
    conclusion = (
        f"{'Reject' if reject else 'Fail to reject'} H0. {row} and {col} are "
        f"{'associated' if reject else 'not shown to be associated'} "
        f"(Cramer's V = {v:.2f}, {_effect_word(v, 0.1, 0.3, 0.5)}).{rates}"
    )
    return TestResult(
        name=name, test="Chi-square test of independence",
        h0=f"{row} and {col} are independent", h1=f"{row} and {col} are associated",
        statistic=float(chi2), p_value=float(p), effect_size=v, effect_label="Cramer's V",
        conclusion=conclusion,
        details={"dof": int(dof), "min_expected_count": float(expected.min()),
                 "table": table.to_dict()},
    )


def anova_test(df, value, group, name):
    """One-way ANOVA with Levene's test and a Kruskal-Wallis check."""
    groups = [g[value].to_numpy() for _, g in df.groupby(group)]
    if len(groups) < 2:
        raise ValueError("ANOVA needs at least 2 groups")
    f = stats.f_oneway(*groups)
    kw = stats.kruskal(*groups)
    lev = stats.levene(*groups)
    eta = eta_squared(groups)
    reject = f.pvalue < ALPHA
    means = df.groupby(group)[value].mean().round(1)
    conclusion = (
        f"{'Reject' if reject else 'Fail to reject'} H0. Group means: "
        + ", ".join(f"{k} {v:,.1f}" for k, v in means.items())
        + f". eta^2 = {eta:.3f} ({_effect_word(eta, 0.01, 0.06, 0.14)}). "
        f"Kruskal-Wallis {'agrees' if (kw.pvalue < ALPHA) == reject else 'disagrees'} (p = {fmt_p(kw.pvalue)})."
    )
    return TestResult(
        name=name, test="One-way ANOVA",
        h0=f"Mean {value} is equal across all {group} groups",
        h1=f"At least one {group} group has a different mean {value}",
        statistic=float(f.statistic), p_value=float(f.pvalue), effect_size=eta,
        effect_label="eta squared", conclusion=conclusion,
        details={"groups": int(len(groups)), "levene_p": float(lev.pvalue),
                 "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue)},
    )


def run_hypothesis_tests(biz, sales):
    """The six hypothesis tests reported in the project."""
    return [
        welch_ttest(biz, "Tenure", "Status", "Churned", "Retained",
                    "T1: Do churned customers have shorter tenure?"),
        welch_ttest(biz, "MonthlyCharges", "Status", "Churned", "Retained",
                    "T2: Do churned customers pay higher monthly charges?"),
        chi_square_test(biz, "Contract", "Churn", "T3: Is churn related to contract type?"),
        chi_square_test(biz, "PaymentMethod", "Churn", "T4: Is churn related to payment method?"),
        anova_test(biz, "MonthlyCharges", "Contract", "T5: Do monthly charges differ by contract type?"),
        anova_test(sales, "Total_Sales", "Product", "T6: Does order value differ by product?"),
    ]


def bonferroni(results, alpha=ALPHA):
    """Bonferroni-adjusted threshold and which tests still pass it."""
    threshold = alpha / len(results)
    return threshold, [r.name for r in results if r.p_value < threshold]


# --------------------------------------------------------------------------- #
# 5. CONFIDENCE INTERVALS
# --------------------------------------------------------------------------- #
def mean_ci(values, confidence=0.95):
    """t-based confidence interval for a mean."""
    s = pd.Series(values).dropna().astype(float)
    n = len(s)
    if n < 2:
        raise ValueError("Need at least 2 values for a confidence interval")
    mean = s.mean()
    se = s.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    moe = t_crit * se
    return {"estimate": mean, "lower": mean - moe, "upper": mean + moe, "margin_of_error": moe, "n": n, "method": "t"}


def proportion_ci(successes, n, confidence=0.95):
    """Wilson score interval for a proportion (better than normal approx. for small p)."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid counts for a proportion")
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return {"estimate": p, "lower": centre - half, "upper": centre + half,
            "margin_of_error": half, "n": n, "method": "Wilson"}


def bootstrap_ci(values, stat=np.mean, confidence=0.95, n_boot=10_000, seed=RANDOM_SEED):
    """Percentile bootstrap confidence interval."""
    arr = np.asarray(pd.Series(values).dropna(), float)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    est = stat(arr)
    return {"estimate": est, "lower": lo, "upper": hi, "margin_of_error": (hi - lo) / 2,
            "n": len(arr), "method": f"bootstrap ({n_boot:,} resamples)"}


def key_confidence_intervals(biz, sales):
    churned = int(biz["Churn"].sum())
    return {
        "Mean monthly charges": mean_ci(biz["MonthlyCharges"]),
        "Mean tenure (months)": mean_ci(biz["Tenure"]),
        "Churn rate": proportion_ci(churned, len(biz)),
        "Mean order value (t)": mean_ci(sales["Total_Sales"]),
        "Mean order value (bootstrap)": bootstrap_ci(sales["Total_Sales"]),
        "Median order value (bootstrap)": bootstrap_ci(sales["Total_Sales"], stat=np.median),
    }


# --------------------------------------------------------------------------- #
# 6. REGRESSION
# --------------------------------------------------------------------------- #
def fit_ols(df, formula):
    """Fit an OLS model with statsmodels and return the fitted results."""
    return smf.ols(formula, data=df).fit()


def regression_summary(model):
    coefs = pd.DataFrame({
        "coef": model.params, "std_err": model.bse, "t": model.tvalues, "p_value": model.pvalues,
        "ci_low": model.conf_int()[0], "ci_high": model.conf_int()[1],
    })
    return {
        "formula": model.model.formula, "n": int(model.nobs), "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj), "f_pvalue": float(model.f_pvalue),
        "rmse": float(np.sqrt(model.mse_resid)), "coefficients": coefs,
    }


def run_regressions(biz, sales):
    logged = sales.assign(log_total=np.log(sales["Total_Sales"]), log_qty=np.log(sales["Quantity"]),
                          log_price=np.log(sales["Price"]))
    return {
        "sales_linear": fit_ols(sales, "Total_Sales ~ Quantity + Price"),
        "sales_loglog": fit_ols(logged, "log_total ~ log_qty + log_price"),
        "charges_linear": fit_ols(biz, "TotalCharges ~ Tenure + MonthlyCharges"),
    }


# --------------------------------------------------------------------------- #
# 7. VISUALIZATIONS
# --------------------------------------------------------------------------- #
def _thousands(axis):
    """Show large numbers as 150K instead of 150000."""
    axis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}K"))


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_distributions(biz, sales, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    series = [(biz, "Tenure", "months"), (biz, "MonthlyCharges", "per month"),
              (biz, "TotalCharges", "total"), (sales, "Total_Sales", "order value")]
    for ax, (df, col, unit) in zip(axes.flat, series):
        sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB", edgecolor="white")
        ax.axvline(df[col].mean(), color="#C73E1D", ls="--", label=f"mean {df[col].mean():,.0f}")
        ax.axvline(df[col].median(), color="#F18F01", ls=":", lw=2, label=f"median {df[col].median():,.0f}")
        ax.set_title(f"{col} ({unit})", fontweight="bold")
        if df[col].max() > 50_000:
            _thousands(ax.xaxis)
        ax.legend(fontsize=8)
    fig.suptitle("Distributions with KDE", fontsize=14, fontweight="bold")
    return _save(fig, out_dir, "01_distributions.png")


def plot_qq(biz, sales, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    series = [(biz, "Tenure"), (biz, "MonthlyCharges"), (biz, "TotalCharges"), (sales, "Total_Sales")]
    for ax, (df, col) in zip(axes.flat, series):
        stats.probplot(df[col], dist="norm", plot=ax)
        ax.get_lines()[0].set(markersize=3, color="#2E86AB")
        ax.get_lines()[1].set(color="#C73E1D")
        ax.set_title(f"Q-Q plot: {col}", fontweight="bold")
    fig.suptitle("Normality check (points on the line = normal)", fontsize=14, fontweight="bold")
    return _save(fig, out_dir, "02_qq_plots.png")


def plot_correlation_heatmaps(biz, sales, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    bcols = BUSINESS_NUMERIC + ["SeniorCitizen", "Churn"]
    sns.heatmap(biz[bcols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[0],
                square=True, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Business data (Pearson r)", fontweight="bold")
    sns.heatmap(sales[SALES_NUMERIC].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                ax=axes[1], square=True, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Sales data (Pearson r)", fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    return _save(fig, out_dir, "03_correlation_heatmaps.png")


def plot_churn_comparisons(biz, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    order = ["Retained", "Churned"]
    for ax, col in zip(axes, ["Tenure", "MonthlyCharges"]):
        sns.boxplot(data=biz, x="Status", y=col, order=order, hue="Status", palette=PALETTE, ax=ax, legend=False)
        sns.stripplot(data=biz, x="Status", y=col, order=order, color="black", size=2, alpha=0.35, ax=ax)
        ax.set_title(f"{col}: retained vs churned", fontweight="bold")
        ax.set_xlabel("")
    return _save(fig, out_dir, "04_churn_boxplots.png")


def plot_churn_by_contract(biz, out_dir):
    rows = []
    for contract, g in biz.groupby("Contract"):
        ci = proportion_ci(int(g["Churn"].sum()), len(g))
        rows.append((contract, ci["estimate"] * 100, ci["lower"] * 100, ci["upper"] * 100, len(g)))
    tbl = pd.DataFrame(rows, columns=["Contract", "rate", "lo", "hi", "n"])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C73E1D", "#F18F01", "#2E86AB"]
    ax.bar(tbl["Contract"], tbl["rate"], color=colors, yerr=[tbl["rate"] - tbl["lo"], tbl["hi"] - tbl["rate"]],
           capsize=8, edgecolor="white")
    for i, r in tbl.iterrows():
        ax.text(i, r["hi"] + 0.8, f"{r['rate']:.1f}%  (n={r['n']})", ha="center", va="bottom", fontsize=9)
    ax.axhline(biz["Churn"].mean() * 100, color="gray", ls="--", label=f"overall {biz['Churn'].mean() * 100:.1f}%")
    ax.set_ylabel("Churn rate (%) with 95% Wilson CI")
    ax.set_title("Churn rate by contract type", fontweight="bold")
    ax.set_ylim(0, tbl["hi"].max() + 8)
    ax.legend()
    return _save(fig, out_dir, "05_churn_rate_by_contract.png")


def plot_tenure_churn_curve(biz, out_dir):
    bins = [0, 6, 12, 24, 36, 48, 60, 72]
    labels = ["1-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    b = biz.assign(TenureBand=pd.cut(biz["Tenure"], bins=bins, labels=labels, include_lowest=True))
    rate = b.groupby("TenureBand", observed=False)["Churn"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(rate.index.astype(str), rate["mean"] * 100, color="#6C4E9B", edgecolor="white")
    for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
        ax.text(i, m * 100 + 1, f"{m * 100:.0f}%\n(n={n})", ha="center", fontsize=9)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by tenure band", fontweight="bold")
    ax.set_ylim(0, rate["mean"].max() * 100 + 12)
    return _save(fig, out_dir, "06_churn_by_tenure_band.png")


def plot_regression(model, sales, out_dir):
    fitted, resid = model.fittedvalues, model.resid
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(sales["Total_Sales"], fitted, color="#2E86AB", alpha=0.7)
    lim = [min(sales["Total_Sales"].min(), fitted.min()), max(sales["Total_Sales"].max(), fitted.max())]
    axes[0].plot(lim, lim, color="#C73E1D", ls="--", label="perfect prediction")
    axes[0].set(xlabel="Actual Total_Sales", ylabel="Predicted", title=f"Actual vs predicted (R^2 = {model.rsquared:.3f})")
    axes[0].legend()
    axes[1].scatter(fitted, resid, color="#3B8B5A", alpha=0.7)
    axes[1].axhline(0, color="black", lw=1)
    sns.regplot(x=fitted, y=resid, lowess=True, scatter=False, color="#C73E1D", ax=axes[1])
    axes[1].set(xlabel="Fitted value", ylabel="Residual", title="Residuals vs fitted (curve = model misses Q x P)")
    for ax in axes:
        ax.title.set_fontweight("bold")
        _thousands(ax.xaxis)
        _thousands(ax.yaxis)
    fig.suptitle("Linear regression: Total_Sales ~ Quantity + Price", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "07_regression_diagnostics.png")


def plot_total_charges_check(biz, out_dir):
    expected = biz["Tenure"] * biz["MonthlyCharges"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(expected, biz["TotalCharges"], s=14, alpha=0.6, color="#2E86AB")
    lim = [0, max(expected.max(), biz["TotalCharges"].max())]
    ax.plot(lim, lim, color="#C73E1D", ls="--", label="TotalCharges = Tenure x MonthlyCharges")
    ax.set(xlabel="Tenure x MonthlyCharges (expected)", ylabel="TotalCharges (recorded)")
    ax.set_title("Data-quality check: recorded vs expected TotalCharges", fontweight="bold")
    ax.legend(loc="upper left")
    return _save(fig, out_dir, "08_totalcharges_quality_check.png")


def plot_confidence_intervals(cis, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    groups = [
        (["Mean monthly charges"], "Monthly charges"),
        (["Churn rate"], "Churn rate"),
        (["Mean order value (t)", "Mean order value (bootstrap)", "Median order value (bootstrap)"], "Order value"),
    ]
    for ax, (keys, title) in zip(axes, groups):
        for i, k in enumerate(keys):
            c = cis[k]
            scale = 100 if k == "Churn rate" else 1
            ax.errorbar(c["estimate"] * scale, i, xerr=[[(c["estimate"] - c["lower"]) * scale],
                                                       [(c["upper"] - c["estimate"]) * scale]],
                        fmt="o", color="#2E86AB", capsize=6, lw=2)
        ax.set_yticks(range(len(keys)), [k.replace("order value ", "") for k in keys])
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.set_title(f"{title} - 95% CI" + (" (%)" if title == "Churn rate" else ""), fontweight="bold")
        if title == "Order value":
            _thousands(ax.xaxis)
    return _save(fig, out_dir, "09_confidence_intervals.png")


def create_visualizations(biz, sales, models, cis, out_dir):
    return [
        plot_distributions(biz, sales, out_dir),
        plot_qq(biz, sales, out_dir),
        plot_correlation_heatmaps(biz, sales, out_dir),
        plot_churn_comparisons(biz, out_dir),
        plot_churn_by_contract(biz, out_dir),
        plot_tenure_churn_curve(biz, out_dir),
        plot_regression(models["sales_linear"], sales, out_dir),
        plot_total_charges_check(biz, out_dir),
        plot_confidence_intervals(cis, out_dir),
    ]


# --------------------------------------------------------------------------- #
# 8. INSIGHTS & REPORTING
# --------------------------------------------------------------------------- #
def fmt_p(p):
    if p < 1e-300:
        return "< 1e-300"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def short_tenure_contract_table(biz, max_tenure=12):
    short = biz[biz["Tenure"] <= max_tenure]
    return pd.crosstab(short["Contract"], short["Status"])


def business_insights(biz, sales, tests, cis, models, quality):
    t = {r.name[:2]: r for r in tests}
    churned = biz[biz["Churn"] == 1]
    short = short_tenure_contract_table(biz)
    mtm_short = short.loc["Month-to-month"] if "Month-to-month" in short.index else None
    lin = models["sales_linear"]
    insights = [
        f"Tenure is the strongest churn signal: churned customers averaged {t['T1'].details['mean_Churned']:.1f} months "
        f"vs {t['T1'].details['mean_Retained']:.1f} for retained (p = {fmt_p(t['T1'].p_value)}, Cohen's d = "
        f"{t['T1'].effect_size:.2f}). Every churned customer left within {int(churned['Tenure'].max())} months.",
        f"Contract type matters (chi-square p = {fmt_p(t['T3'].p_value)}, Cramer's V = {t['T3'].effect_size:.2f}). "
        + (f"Among customers with 12 months or less, {int(mtm_short.get('Churned', 0))} of "
           f"{int(mtm_short.sum())} month-to-month customers churned." if mtm_short is not None else ""),
        f"Churned customers paid more per month ({t['T2'].details['mean_Churned']:.1f} vs "
        f"{t['T2'].details['mean_Retained']:.1f}, p = {fmt_p(t['T2'].p_value)}, Cohen's d = {t['T2'].effect_size:.2f}), a small effect that "
        f"does not survive a Bonferroni correction for 6 tests.",
        f"Payment method (p = {fmt_p(t['T4'].p_value)}) and contract-level pricing (p = {fmt_p(t['T5'].p_value)}) "
        f"show no significant relationship, so neither is a churn lever on this evidence.",
        f"The overall churn rate is {cis['Churn rate']['estimate'] * 100:.1f}% "
        f"(95% CI {cis['Churn rate']['lower'] * 100:.1f}%-{cis['Churn rate']['upper'] * 100:.1f}%).",
        f"Order value is driven by Quantity and Price (R^2 = {lin.rsquared:.3f}); product category alone "
        f"does not significantly change order value (ANOVA p = {fmt_p(t['T6'].p_value)}).",
        f"Data-quality issue: only {quality['pct_within_tolerance']}% of customers have TotalCharges within 10% of "
        f"Tenure x MonthlyCharges (correlation {quality['corr_total_vs_expected']}), so TotalCharges is not "
        f"used to draw conclusions.",
    ]
    recommendations = [
        "Focus retention on the first 12 months: onboarding check-ins at months 1, 3 and 6.",
        "Offer month-to-month customers an incentive to move to a 1- or 2-year contract early in their tenure.",
        "Review pricing for high-charge new customers - they churn slightly more, but test this before acting "
        "because the effect is not robust to multiple-testing correction.",
        "Do not prioritise payment-method changes for churn reduction - no evidence they matter here.",
        "Fix the TotalCharges field at source before using it for revenue or lifetime-value analysis.",
        "Collect marketing-spend data so the effect of marketing on revenue can be tested directly.",
    ]
    return insights, recommendations


def format_results_text(biz, sales, desc_biz, desc_sales, normality, corr, tests, cis, models, quality,
                        spearman=None):
    """Plain-text summary of every statistical test (hypothesis_tests_results.txt)."""
    w = 88
    line = "=" * w
    out = [line, "WEEK 7 - STATISTICAL BUSINESS ANALYSIS: HYPOTHESIS TEST RESULTS".center(w), line,
           f"Business data : business_data.csv ({len(biz)} customers)",
           f"Sales data    : data/sales_data.csv ({len(sales)} orders)",
           f"Significance  : alpha = {ALPHA} (two-sided)", ""]
    for i, r in enumerate(tests, 1):
        out += ["-" * w, r.name, "-" * w,
                f"Test        : {r.test}",
                f"H0          : {r.h0}",
                f"H1          : {r.h1}",
                f"Statistic   : {r.statistic:.4f}",
                f"p-value     : {fmt_p(r.p_value)}",
                f"Effect size : {r.effect_label} = {r.effect_size:.3f}",
                f"Decision    : {'REJECT H0 - significant' if r.reject_h0 else 'FAIL TO REJECT H0 - not significant'}"]
        out += textwrap.wrap("Conclusion  : " + r.conclusion, w, subsequent_indent=" " * 14)
        out.append("")
    thr, survivors = bonferroni(tests)
    out += ["-" * w, "MULTIPLE TESTING", "-" * w,
            f"Bonferroni threshold for {len(tests)} tests: {thr:.4f}",
            "Still significant: " + (", ".join(s[:2] for s in survivors) if survivors else "none"), ""]
    out += ["-" * w, "NORMALITY TESTS (Shapiro-Wilk / D'Agostino-Pearson)", "-" * w]
    for var, row in normality.iterrows():
        out.append(f"{var:<15} W = {row['shapiro_W']:.4f} (p = {fmt_p(row['shapiro_p'])})   "
                   f"K2 = {row['dagostino_K2']:.1f} (p = {fmt_p(row['dagostino_p'])})   "
                   f"normal: {'yes' if row['normal_at_5%'] else 'no'}")
    out += ["", "-" * w, "CORRELATIONS (Pearson, significant at 5% marked *)", "-" * w]
    for _, row in corr.iterrows():
        out.append(f"{row['var_1']:>15} vs {row['var_2']:<15} r = {row['r']:+.3f}  p = {fmt_p(row['p_value'])}"
                   f"  {row['strength']}{' *' if row['significant'] else ''}")
    if spearman is not None:
        out += ["", "-" * w, "CORRELATIONS (Spearman rank check)", "-" * w]
        for _, row in spearman.iterrows():
            out.append(f"{row['var_1']:>15} vs {row['var_2']:<15} rho = {row['r']:+.3f}  p = {fmt_p(row['p_value'])}"
                       f"{' *' if row['significant'] else ''}")
    out += ["", "-" * w, "95% CONFIDENCE INTERVALS", "-" * w]
    for name, c in cis.items():
        scale, unit = (100, "%") if name == "Churn rate" else (1, "")
        out.append(f"{name:<32} {c['estimate'] * scale:>12,.2f}{unit}  [{c['lower'] * scale:,.2f}{unit}, "
                   f"{c['upper'] * scale:,.2f}{unit}]  MOE +/-{c['margin_of_error'] * scale:,.2f}{unit}  ({c['method']})")
    out += ["", "-" * w, "REGRESSION MODELS (OLS)", "-" * w]
    for key, m in models.items():
        s = regression_summary(m)
        out.append(f"{s['formula']}   n = {s['n']}   R^2 = {s['r_squared']:.4f}   adj R^2 = "
                   f"{s['adj_r_squared']:.4f}   F p-value = {fmt_p(s['f_pvalue'])}")
        for term, row in s["coefficients"].iterrows():
            out.append(f"    {term:<15} coef = {row['coef']:>14,.4f}   p = {fmt_p(row['p_value'])}")
    out += ["", "-" * w, "DATA QUALITY: TotalCharges vs Tenure x MonthlyCharges", "-" * w,
            f"Within 10%: {quality['within_tolerance']} of {quality['rows']} rows ({quality['pct_within_tolerance']}%)",
            f"Ratio recorded/expected: median {quality['median_ratio']}, min {quality['min_ratio']}, "
            f"max {quality['max_ratio']}",
            f"Correlation recorded vs expected: {quality['corr_total_vs_expected']}", line]
    return "\n".join(out) + "\n"


def _text_page(pdf, title, paragraphs, fontsize=10):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=16, fontweight="bold", va="top")
    y = 0.91
    for para in paragraphs:
        is_head = para.startswith("## ")
        txt = para[3:] if is_head else para
        wrapped = textwrap.wrap(txt, 98 if not is_head else 80) or [""]
        for ln in wrapped:
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.95
            fig.text(0.07, y, ln, fontsize=12 if is_head else fontsize,
                     fontweight="bold" if is_head else "normal", va="top", family="DejaVu Sans")
            y -= 0.022 if is_head else 0.0175
        y -= 0.008
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf, image_path, caption):
    img = plt.imread(image_path)
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.05, 0.25, 0.9, 0.65])
    ax.imshow(img)
    ax.axis("off")
    for i, ln in enumerate(textwrap.wrap(caption, 95)):
        fig.text(0.07, 0.22 - i * 0.018, ln, fontsize=10, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf_report(path, biz, sales, desc_biz, desc_sales, tests, cis, models, quality, charts,
                     insights, recommendations):
    lin = regression_summary(models["sales_linear"])
    loglog = regression_summary(models["sales_loglog"])
    chg = regression_summary(models["charges_linear"])
    thr, survivors = bonferroni(tests)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _text_page(pdf, "Statistical Business Analysis - Week 7", [
            "## 1. Objective",
            "Use descriptive statistics, hypothesis testing, correlation, confidence intervals and regression "
            "to find what drives customer churn and order value, and turn the results into business actions.",
            "## 2. Data",
            f"business_data.csv - {len(biz)} customers with Tenure, MonthlyCharges, TotalCharges, Contract, "
            f"PaymentMethod, PaperlessBilling, SeniorCitizen and Churn (0/1). Churn rate "
            f"{biz['Churn'].mean() * 100:.1f}% ({int(biz['Churn'].sum())} customers).",
            f"data/sales_data.csv - {len(sales)} orders (Jan-Apr 2024) with Product, Quantity, Price, Region and "
            f"Total_Sales.",
            "Scope note: the datasets contain no marketing-spend column, so the 'marketing spend vs revenue' "
            "example in the task brief cannot be tested. The same methods are applied to the relationships "
            "that do exist in the data.",
            "## 3. Methodology",
            "Descriptive statistics (mean, median, mode, standard deviation, variance, quartiles, skewness, "
            "kurtosis). Normality: Shapiro-Wilk and D'Agostino-Pearson tests plus Q-Q plots. Correlation: "
            "Pearson with p-values, Spearman as a rank-based check. Hypothesis tests at alpha = 0.05: Welch "
            "t-tests (no equal-variance assumption) with Mann-Whitney U as a non-parametric check, chi-square "
            "tests of independence with Cramer's V, one-way ANOVA with Levene's test and Kruskal-Wallis. Because "
            "the main variables are not normally distributed, every parametric test is paired with a "
            "non-parametric one. Confidence intervals: t-intervals for means, Wilson interval for the churn "
            "proportion, percentile bootstrap (10,000 resamples, seed 42) for order value. Regression: OLS "
            "with statsmodels, reporting R^2, coefficients, p-values and residual diagnostics.",
            "## 4. Key results",
        ] + insights + [
            "## 5. Recommendations",
        ] + [f"- {r}" for r in recommendations])

        desc_rows = []
        for name, d in [("Business", desc_biz), ("Sales", desc_sales)]:
            for var, r in d.iterrows():
                desc_rows.append(f"{name} / {var}: mean {r['mean']:,.1f}, median {r['median']:,.1f}, mode "
                                 f"{r['mode']:,.1f}, SD {r['std']:,.1f}, variance {r['variance']:,.1f}, Q1 "
                                 f"{r['q1']:,.1f}, Q3 {r['q3']:,.1f}, skew {r['skewness']:.2f}, kurtosis "
                                 f"{r['kurtosis']:.2f}")
        test_rows = []
        for r in tests:
            test_rows += [f"## {r.name}",
                          f"{r.test}. H0: {r.h0}. H1: {r.h1}.",
                          f"Statistic = {r.statistic:.3f}, p = {fmt_p(r.p_value)}, {r.effect_label} = "
                          f"{r.effect_size:.3f}.",
                          r.conclusion]
        _text_page(pdf, "Results", [
            "## Descriptive statistics"] + desc_rows + [
            "None of the numeric variables is normally distributed (all normality p-values < 0.001): tenure "
            "and charges are close to uniform (negative kurtosis), and order value is right-skewed.",
        ] + test_rows + [
            "## Multiple testing",
            f"Six tests were run, so the Bonferroni threshold is {thr:.4f}. Still significant: "
            + (", ".join(s[:2] for s in survivors) if survivors else "none") + ".",
        ])

        ci_rows = []
        for name, c in cis.items():
            scale, unit = (100, "%") if name == "Churn rate" else (1, "")
            ci_rows.append(f"{name}: {c['estimate'] * scale:,.2f}{unit}, 95% CI {c['lower'] * scale:,.2f}{unit} "
                           f"to {c['upper'] * scale:,.2f}{unit}, margin of error +/-{c['margin_of_error'] * scale:,.2f}"
                           f"{unit} ({c['method']}).")
        _text_page(pdf, "Confidence Intervals & Regression", ["## 95% confidence intervals"] + ci_rows + [
            "## Regression 1: Total_Sales ~ Quantity + Price",
            f"n = {lin['n']}, R^2 = {lin['r_squared']:.3f}, adjusted R^2 = {lin['adj_r_squared']:.3f}, RMSE = "
            f"{lin['rmse']:,.0f}. Each extra unit adds about {lin['coefficients'].loc['Quantity', 'coef']:,.0f} to "
            f"order value and each extra 1 of unit price adds {lin['coefficients'].loc['Price', 'coef']:.2f} "
            f"(both p < 0.001). The residual plot is curved because order value is really Quantity x Price, "
            f"which a linear model can only approximate.",
            f"A log-log model confirms this: log(Total_Sales) ~ log(Quantity) + log(Price) gives R^2 = "
            f"{loglog['r_squared']:.4f} with both coefficients = "
            f"{loglog['coefficients'].loc['log_qty', 'coef']:.2f}, i.e. the exact multiplicative identity.",
            "## Regression 2: TotalCharges ~ Tenure + MonthlyCharges",
            f"R^2 = {chg['r_squared']:.4f}, F-test p = {fmt_p(chg['f_pvalue'])}. Tenure and monthly charges "
            f"explain almost none of TotalCharges, which should not happen if TotalCharges were the running total "
            f"of payments. Only {quality['within_tolerance']} of {quality['rows']} rows "
            f"({quality['pct_within_tolerance']}%) are within 10% of Tenure x MonthlyCharges. This is reported "
            f"as a data-quality problem.",
            "## Limitations",
            "- Observational data: the tests show association, not causation.",
            f"- Only {int(biz['Churn'].sum())} churned customers, so estimates for churners are less precise.",
            "- Very strong patterns (no churn after 11 months; all short-tenure month-to-month customers "
            "churned) suggest the dataset may be simulated; confirm on real customer data before acting.",
            "- No marketing-spend data, so marketing effectiveness could not be tested.",
            "- The sales data covers only about 3 months and 100 orders.",
        ])
        captions = [
            "Figure 1 - Distributions with KDE. Tenure and charges are spread fairly evenly (not bell-shaped); "
            "order value is right-skewed.",
            "Figure 2 - Q-Q plots. S-shaped curves confirm the variables are not normally distributed.",
            "Figure 3 - Correlation heatmaps. Tenure has by far the strongest link to churn (r = -0.51, moderate); in sales data "
            "Quantity and Price each correlate with Total_Sales but not with each other.",
            "Figure 4 - Tenure and monthly charges for retained vs churned customers.",
            "Figure 5 - Churn rate by contract type with 95% Wilson confidence intervals.",
            "Figure 6 - Churn rate by tenure band: all churn happens in the first year.",
            "Figure 7 - Regression diagnostics for Total_Sales ~ Quantity + Price.",
            "Figure 8 - Recorded TotalCharges vs Tenure x MonthlyCharges. Points should sit on the red line; "
            "most do not.",
            "Figure 9 - 95% confidence intervals for key metrics.",
        ]
        for chart, cap in zip(charts, captions):
            _image_page(pdf, chart, cap)
    return path


# --------------------------------------------------------------------------- #
# 9. PIPELINE
# --------------------------------------------------------------------------- #
def run_analysis(business_path=BUSINESS_DATA, sales_path=SALES_DATA, out_dir=BASE_DIR):
    """Run every step and write charts, the results text file and the PDF report."""
    out_dir = Path(out_dir)
    biz = load_business_data(business_path)
    sales = load_sales_data(sales_path)
    log.info("Loaded %d customers and %d orders", len(biz), len(sales))

    desc_biz = descriptive_stats(biz, BUSINESS_NUMERIC)
    desc_sales = descriptive_stats(sales, SALES_NUMERIC)
    normality = pd.concat([normality_tests(biz, BUSINESS_NUMERIC), normality_tests(sales, ["Total_Sales"])])
    corr = pd.concat([correlation_table(biz, BUSINESS_NUMERIC + ["Churn"]),
                      correlation_table(sales, SALES_NUMERIC)], ignore_index=True)
    spearman = pd.concat([correlation_table(biz, BUSINESS_NUMERIC + ["Churn"], "spearman"),
                          correlation_table(sales, SALES_NUMERIC, "spearman")], ignore_index=True)
    tests = run_hypothesis_tests(biz, sales)
    cis = key_confidence_intervals(biz, sales)
    models = run_regressions(biz, sales)
    quality, _ = total_charges_consistency(biz)
    insights, recs = business_insights(biz, sales, tests, cis, models, quality)

    charts = create_visualizations(biz, sales, models, cis, out_dir / "visualizations")
    results_txt = out_dir / "hypothesis_tests_results.txt"
    results_txt.write_text(format_results_text(biz, sales, desc_biz, desc_sales, normality, corr, tests, cis,
                                               models, quality, spearman), encoding="utf-8")
    pdf = build_pdf_report(out_dir / "statistical_report.pdf", biz, sales, desc_biz, desc_sales, tests, cis,
                           models, quality, charts, insights, recs)
    log.info("Wrote %s, %s and %d charts", results_txt.name, pdf.name, len(charts))
    return {"business": biz, "sales": sales, "desc_business": desc_biz, "desc_sales": desc_sales,
            "normality": normality, "pearson": corr, "spearman": spearman, "tests": tests, "cis": cis,
            "models": models, "quality": quality, "insights": insights, "recommendations": recs,
            "charts": charts, "results_txt": results_txt, "pdf": pdf}


def print_summary(res):
    print("\n" + "=" * 70)
    print("STATISTICAL BUSINESS ANALYSIS - SUMMARY".center(70))
    print("=" * 70)
    for r in res["tests"]:
        flag = "SIGNIFICANT" if r.reject_h0 else "not significant"
        print(f"{r.name[:2]}  p = {fmt_p(r.p_value):>10}  {r.effect_label} = {r.effect_size:6.3f}  {flag}")
    print("-" * 70)
    for i, s in enumerate(res["insights"], 1):
        print(textwrap.fill(f"{i}. {s}", 70, subsequent_indent="   "))
    print("=" * 70 + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Week 7 statistical business analysis")
    p.add_argument("--business", default=BUSINESS_DATA, help="business CSV (default: business_data.csv)")
    p.add_argument("--sales", default=SALES_DATA, help="sales CSV (default: data/sales_data.csv)")
    p.add_argument("--out", default=BASE_DIR, help="output folder (default: project folder)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        print_summary(run_analysis(args.business, args.sales, args.out))
    except (FileNotFoundError, DataValidationError) as err:
        log.error(err)
        sys.exit(1)

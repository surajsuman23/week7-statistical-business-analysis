"""Tests for stats_analysis.py. Run from the project folder with:  python -m pytest -v"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stats_analysis as sa  # noqa: E402

BIZ_HEADER = "CustomerID,Tenure,MonthlyCharges,TotalCharges,Contract,PaymentMethod,PaperlessBilling,SeniorCitizen,Churn\n"


@pytest.fixture(scope="module")
def biz():
    return sa.load_business_data()


@pytest.fixture(scope="module")
def sales():
    return sa.load_sales_data()


@pytest.fixture(scope="module")
def tests_run(biz, sales):
    return sa.run_hypothesis_tests(biz, sales)


def write(tmp_path, text, name="data.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------- loading & validation ----------
def test_load_shapes(biz, sales):
    assert len(biz) == 500 and len(sales) == 100
    assert set(biz["Status"]) == {"Retained", "Churned"}


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        sa.load_business_data("nope.csv")


def test_missing_columns_raise(tmp_path):
    with pytest.raises(sa.DataValidationError, match="missing required columns"):
        sa.load_business_data(write(tmp_path, "CustomerID,Tenure\nC1,5\n"))


def test_empty_file_raises(tmp_path):
    with pytest.raises(sa.DataValidationError):
        sa.load_business_data(write(tmp_path, ""))


def test_invalid_churn_value_raises(tmp_path):
    p = write(tmp_path, BIZ_HEADER + "C1,5,50,250,One year,Credit Card,No,0,2\n")
    with pytest.raises(sa.DataValidationError, match="Churn"):
        sa.load_business_data(p)


def test_bad_rows_and_duplicates_dropped(tmp_path):
    p = write(tmp_path, BIZ_HEADER
              + "C1,5,50,250,One year,Credit Card,No,0,0\n"
              + "C1,5,50,250,One year,Credit Card,No,0,0\n"      # duplicate ID
              + "C2,abc,50,250,One year,Credit Card,No,0,1\n")   # non-numeric tenure
    df = sa.load_business_data(p)
    assert list(df["CustomerID"]) == ["C1"]


# ---------- descriptive statistics ----------
def test_descriptive_stats_match_numpy(biz):
    d = sa.descriptive_stats(biz, ["Tenure"]).loc["Tenure"]
    t = biz["Tenure"].to_numpy()
    assert d["mean"] == pytest.approx(t.mean())
    assert d["median"] == pytest.approx(np.median(t))
    assert d["std"] == pytest.approx(t.std(ddof=1))
    assert d["variance"] == pytest.approx(t.var(ddof=1))
    assert d["iqr"] == pytest.approx(d["q3"] - d["q1"])


def test_normality_detects_non_normal_and_normal():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"normal": rng.normal(size=400), "uniform": rng.uniform(size=400)})
    res = sa.normality_tests(df, ["normal", "uniform"])
    assert bool(res.loc["normal", "normal_at_5%"]) is True
    assert bool(res.loc["uniform", "normal_at_5%"]) is False


# ---------- correlation ----------
def test_correlation_matches_scipy(biz):
    tbl = sa.correlation_table(biz, ["Tenure", "Churn"])
    r, p = stats.pearsonr(biz["Tenure"], biz["Churn"])
    assert tbl.loc[0, "r"] == pytest.approx(r)
    assert tbl.loc[0, "p_value"] == pytest.approx(p)
    assert tbl.loc[0, "strength"] == "moderate"


@pytest.mark.parametrize("r,label", [(0.9, "strong"), (-0.5, "moderate"), (0.25, "weak"), (0.05, "negligible")])
def test_correlation_strength_labels(r, label):
    assert sa.correlation_strength(r) == label


# ---------- hypothesis tests ----------
def test_six_tests_with_required_fields(tests_run):
    assert len(tests_run) == 6
    for r in tests_run:
        assert r.h0 and r.h1 and r.conclusion
        assert 0 <= r.p_value <= 1


def test_welch_matches_scipy(biz, tests_run):
    churned = biz.loc[biz["Churn"] == 1, "Tenure"]
    retained = biz.loc[biz["Churn"] == 0, "Tenure"]
    expected = stats.ttest_ind(churned, retained, equal_var=False)
    assert tests_run[0].statistic == pytest.approx(expected.statistic)
    assert tests_run[0].p_value == pytest.approx(expected.pvalue, rel=1e-6)


def test_chi_square_matches_scipy(biz, tests_run):
    chi2, p, *_ = stats.chi2_contingency(pd.crosstab(biz["Contract"], biz["Churn"]))
    assert tests_run[2].statistic == pytest.approx(chi2)
    assert tests_run[2].p_value == pytest.approx(p)
    assert 0 <= tests_run[2].effect_size <= 1


def test_anova_matches_scipy(sales, tests_run):
    f = stats.f_oneway(*[g["Total_Sales"] for _, g in sales.groupby("Product")])
    assert tests_run[5].statistic == pytest.approx(f.statistic)
    assert tests_run[5].p_value == pytest.approx(f.pvalue)


def test_expected_decisions(tests_run):
    decisions = [r.reject_h0 for r in tests_run]
    assert decisions == [True, True, True, False, False, False]


def test_bonferroni(tests_run):
    threshold, survivors = sa.bonferroni(tests_run)
    assert threshold == pytest.approx(0.05 / 6)
    assert [s[:2] for s in survivors] == ["T1", "T3"]


def test_cohens_d_known_value():
    a, b = np.array([2.0, 4, 6]), np.array([1.0, 3, 5])
    assert sa.cohens_d(a, b) == pytest.approx(0.5)


def test_welch_needs_two_values():
    df = pd.DataFrame({"v": [1, 2, 3], "g": ["a", "b", "b"]})
    with pytest.raises(ValueError):
        sa.welch_ttest(df, "v", "g", "a", "b", "x")


# ---------- confidence intervals ----------
def test_mean_ci_contains_mean_and_matches_scipy(biz):
    ci = sa.mean_ci(biz["MonthlyCharges"])
    lo, hi = stats.t.interval(0.95, len(biz) - 1, loc=biz["MonthlyCharges"].mean(), scale=stats.sem(biz["MonthlyCharges"]))
    assert ci["lower"] == pytest.approx(lo) and ci["upper"] == pytest.approx(hi)
    assert ci["lower"] < ci["estimate"] < ci["upper"]


def test_wilson_ci_bounds():
    ci = sa.proportion_ci(53, 500)
    assert 0 < ci["lower"] < 0.106 < ci["upper"] < 1
    zero = sa.proportion_ci(0, 20)
    assert zero["lower"] == pytest.approx(0, abs=1e-12) and zero["upper"] > 0


def test_proportion_ci_rejects_bad_counts():
    with pytest.raises(ValueError):
        sa.proportion_ci(5, 3)


def test_bootstrap_reproducible(sales):
    a = sa.bootstrap_ci(sales["Total_Sales"], n_boot=2000)
    b = sa.bootstrap_ci(sales["Total_Sales"], n_boot=2000)
    assert a["lower"] == b["lower"] and a["upper"] == b["upper"]
    assert a["lower"] < a["estimate"] < a["upper"]


# ---------- regression & data quality ----------
def test_ols_recovers_known_line():
    x = np.arange(20, dtype=float)
    df = pd.DataFrame({"x": x, "y": 3 * x + 2})
    m = sa.fit_ols(df, "y ~ x")
    assert m.params["x"] == pytest.approx(3) and m.params["Intercept"] == pytest.approx(2)
    assert m.rsquared == pytest.approx(1)


def test_loglog_model_is_exact(biz, sales):
    models = sa.run_regressions(biz, sales)
    assert models["sales_loglog"].rsquared == pytest.approx(1, abs=1e-9)
    assert 0.85 < models["sales_linear"].rsquared < 0.9


def test_total_charges_consistency():
    df = pd.DataFrame({"Tenure": [10, 10, 10], "MonthlyCharges": [100, 100, 100], "TotalCharges": [1000, 1050, 300]})
    summary, ratio = sa.total_charges_consistency(df)
    assert summary["within_tolerance"] == 2
    assert ratio.tolist() == pytest.approx([1.0, 1.05, 0.3])


# ---------- end-to-end ----------
def test_full_pipeline_writes_outputs(tmp_path):
    res = sa.run_analysis(out_dir=tmp_path)
    assert len(res["charts"]) == 9 and all(c.exists() for c in res["charts"])
    txt = (tmp_path / "hypothesis_tests_results.txt").read_text(encoding="utf-8")
    assert txt.count("H0          :") == 6
    pdf = tmp_path / "statistical_report.pdf"
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"

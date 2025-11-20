import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant

from valuation_model import feature_engineering, load_data, read_config

warnings.filterwarnings("ignore")


def calculate_vif(df, features):
    """Calculate Variance Inflation Factor for multicollinearity check"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(df[features].values, i) for i in range(len(features))
    ]
    return vif_data.sort_values("VIF", ascending=False)


def test_heteroskedasticity(model, X, y):
    """Test for heteroskedasticity using Breusch-Pagan and White tests"""
    results = {}

    # Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, X)
    results["Breusch-Pagan"] = {
        "Lagrange multiplier statistic": bp_test[0],
        "p-value": bp_test[1],
        "f-value": bp_test[2],
        "f p-value": bp_test[3],
    }

    # White test
    try:
        white_test = het_white(model.resid, X)
        results["White"] = {
            "Lagrange multiplier statistic": white_test[0],
            "p-value": white_test[1],
            "f-value": white_test[2],
            "f p-value": white_test[3],
        }
    except Exception as e:
        results["White"] = {"error": str(e)}

    return results


def test_normality(residuals):
    """Test residuals for normality"""
    results = {}

    # Shapiro-Wilk test (use sample if too large)
    if len(residuals) > 5000:
        sample = np.random.choice(residuals, 5000, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(sample)
        results["Shapiro-Wilk"] = {
            "statistic": shapiro_stat,
            "p-value": shapiro_p,
            "note": "Tested on random sample of 5000 observations",
        }
    else:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        results["Shapiro-Wilk"] = {"statistic": shapiro_stat, "p-value": shapiro_p}

    # Jarque-Bera test
    jb_result = stats.jarque_bera(residuals)
    skew = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    results["Jarque-Bera"] = {
        "statistic": jb_result.statistic,
        "p-value": jb_result.pvalue,
        "skewness": skew,
        "kurtosis": kurtosis,
    }

    # Anderson-Darling test
    ad_result = stats.anderson(residuals, dist="norm")
    results["Anderson-Darling"] = {
        "statistic": ad_result.statistic,
        "critical_values": ad_result.critical_values,
        "significance_levels": ad_result.significance_level,
    }

    return results


def generate_diagnostic_plots(model, X, y, features, output_prefix="diagnostic"):
    """Generate comprehensive diagnostic plots"""
    plt.figure(figsize=(20, 16))

    # 1. Residuals vs Fitted
    ax1 = plt.subplot(3, 3, 1)
    fitted = model.fittedvalues
    residuals = model.resid
    ax1.scatter(fitted, residuals, alpha=0.5, s=10)
    ax1.axhline(y=0, color="r", linestyle="--")
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted Values")

    # 2. Q-Q Plot
    ax2 = plt.subplot(3, 3, 2)
    sm.qqplot(residuals, line="45", ax=ax2)
    ax2.set_title("Normal Q-Q Plot")

    # 3. Scale-Location (sqrt of standardized residuals vs fitted)
    ax3 = plt.subplot(3, 3, 3)
    standardized_resid = (residuals - residuals.mean()) / residuals.std()
    ax3.scatter(fitted, np.sqrt(np.abs(standardized_resid)), alpha=0.5, s=10)
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("√|Standardized Residuals|")
    ax3.set_title("Scale-Location Plot")

    # 4. Residual Histogram
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Residuals")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Histogram of Residuals")

    # 5. Residuals vs Leverage
    ax5 = plt.subplot(3, 3, 5)
    leverage = model.get_influence().hat_matrix_diag
    ax5.scatter(leverage, standardized_resid, alpha=0.5, s=10)
    ax5.axhline(y=0, color="r", linestyle="--")
    ax5.set_xlabel("Leverage")
    ax5.set_ylabel("Standardized Residuals")
    ax5.set_title("Residuals vs Leverage")

    # 6. Cook's Distance
    ax6 = plt.subplot(3, 3, 6)
    cooks_d = model.get_influence().cooks_distance[0]
    ax6.stem(range(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
    ax6.set_xlabel("Observation")
    ax6.set_ylabel("Cook's Distance")
    ax6.set_title("Cook's Distance Plot")

    # 7. Actual vs Predicted
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(y, fitted, alpha=0.5, s=10)
    ax7.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    ax7.set_xlabel("Actual Price")
    ax7.set_ylabel("Predicted Price")
    ax7.set_title("Actual vs Predicted Values")

    # 8. Top 10 Feature Importance (by absolute coefficient)
    ax8 = plt.subplot(3, 3, 8)
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.params[1:]})
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    top_features = coef_df.nlargest(10, "Abs_Coefficient")
    ax8.barh(range(len(top_features)), top_features["Coefficient"])
    ax8.set_yticks(range(len(top_features)))
    ax8.set_yticklabels(top_features["Feature"])
    ax8.set_xlabel("Coefficient Value")
    ax8.set_title("Top 10 Features by Coefficient Magnitude")
    ax8.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # 9. Prediction Error Distribution
    ax9 = plt.subplot(3, 3, 9)
    prediction_error = y - fitted
    ax9.hist(prediction_error, bins=50, edgecolor="black", alpha=0.7)
    ax9.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax9.set_xlabel("Prediction Error (Actual - Predicted)")
    ax9.set_ylabel("Frequency")
    ax9.set_title("Prediction Error Distribution")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    return f"{output_prefix}_plots.png"


def generate_correlation_heatmap(data, output_prefix="correlation"):
    """Generate correlation heatmap for numeric variables"""
    # Get numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    # Limit to reasonable number for visualization
    if len(numeric_cols) > 30:
        # Calculate correlation with target
        corr_with_target = (
            data[numeric_cols].corr()["price"].abs().sort_values(ascending=False)
        )
        top_cols = corr_with_target.head(30).index.tolist()
        corr_matrix = data[top_cols].corr()
    else:
        corr_matrix = data[numeric_cols].corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap (Top 30 Features by Price Correlation)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    return f"{output_prefix}_heatmap.png"


def generate_variable_distributions(data, output_prefix="distributions"):
    """Generate distribution plots for key variables"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # Ensure data is all numeric
    data_numeric = data.copy()
    for col in data_numeric.columns:
        data_numeric[col] = pd.to_numeric(data_numeric[col], errors="coerce")

    # Key variables to plot
    key_vars = ["price", "Year", "Odometer", "EngineSize"]

    # Add some categorical variables if they exist
    if "Is4WD" in data_numeric.columns:
        key_vars.append("Is4WD")
    if "IsNew" in data_numeric.columns:
        key_vars.append("IsNew")
    if "IsDealer" in data_numeric.columns:
        key_vars.append("IsDealer")

    # Add any Region columns
    region_cols = [col for col in data_numeric.columns if col.startswith("Region_")]
    if region_cols:
        key_vars.append(region_cols[0])

    # Add any Transmission columns
    trans_cols = [
        col for col in data_numeric.columns if col.startswith("Transmission_")
    ]
    if trans_cols:
        key_vars.append(trans_cols[0])

    for idx, var in enumerate(key_vars[:9]):
        if var in data_numeric.columns:
            var_data = data_numeric[var].dropna().astype(float)
            axes[idx].hist(var_data, bins=30, edgecolor="black", alpha=0.7)
            axes[idx].set_xlabel(var)
            axes[idx].set_ylabel("Frequency")
            axes[idx].set_title(f"Distribution of {var}")
            axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(key_vars), 9):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_vars.png", dpi=150, bbox_inches="tight")
    plt.close()

    return f"{output_prefix}_vars.png"


def generate_markdown_report(
    model, X, y, features, het_tests, norm_tests, vif_data, config, data_info, plots
):
    """Generate comprehensive markdown report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Car Valuation Model - Statistical Validation Report

**Generated:** {timestamp}
**Make/Model:** {config["make"]} {config["model"]}
**Number of Observations:** {len(y)}
**Number of Features:** {len(features)}

---

## Executive Summary

This report provides a comprehensive statistical validation of the linear regression model
used for car price valuation. The analysis includes tests for model assumptions,
diagnostic plots, and coefficient interpretations.

---

## 1. Model Summary

### 1.1 Overall Model Statistics

"""

    # Model statistics
    report += f"""
| Metric | Value |
|--------|-------|
| R-squared | {model.rsquared:.4f} |
| Adjusted R-squared | {model.rsquared_adj:.4f} |
| F-statistic | {model.fvalue:.2f} |
| Prob (F-statistic) | {model.f_pvalue:.4e} |
| AIC | {model.aic:.2f} |
| BIC | {model.bic:.2f} |
| Root MSE | {np.sqrt(model.mse_resid):.2f} |
| Durbin-Watson | {durbin_watson(model.resid):.4f} |

**Interpretation:**
- **R-squared ({model.rsquared:.4f})**: The model explains {model.rsquared * 100:.2f}% of the variance in car prices.
- **Adjusted R-squared ({model.rsquared_adj:.4f})**: Accounts for the number of predictors in the model.
- **F-statistic p-value ({model.f_pvalue:.4e})**: {"Model is statistically significant (p < 0.05)" if model.f_pvalue < 0.05 else "Model is NOT statistically significant (p >= 0.05)"}.
- **Durbin-Watson ({durbin_watson(model.resid):.4f})**: {"Suggests no autocorrelation (value near 2)" if 1.5 < durbin_watson(model.resid) < 2.5 else "May indicate autocorrelation (value far from 2)"}.

---

## 2. Model Assumptions Testing

### 2.1 Linearity

The linearity assumption is assessed visually through the Residuals vs Fitted plot.
The residuals should be randomly scattered around zero with no clear pattern.

**Visual Check:** See diagnostic plots - Residuals vs Fitted Values

---

### 2.2 Homoskedasticity (Constant Variance)

Tests whether the variance of errors is constant across all levels of the independent variables.

#### Breusch-Pagan Test
"""

    # Heteroskedasticity tests
    if "Breusch-Pagan" in het_tests:
        bp = het_tests["Breusch-Pagan"]
        report += f"""
| Statistic | Value |
|-----------|-------|
| LM Statistic | {bp["Lagrange multiplier statistic"]:.4f} |
| p-value | {bp["p-value"]:.4e} |
| F-statistic | {bp["f-value"]:.4f} |
| F p-value | {bp["f p-value"]:.4e} |

**Interpretation:** {"HETEROSKEDASTICITY DETECTED (p < 0.05) - variance is not constant" if bp["p-value"] < 0.05 else "Homoskedasticity assumption holds (p >= 0.05) - variance is constant"}

"""

    if "White" in het_tests and "error" not in het_tests["White"]:
        white = het_tests["White"]
        report += f"""#### White Test

| Statistic | Value |
|-----------|-------|
| LM Statistic | {white["Lagrange multiplier statistic"]:.4f} |
| p-value | {white["p-value"]:.4e} |

**Interpretation:** {"HETEROSKEDASTICITY DETECTED (p < 0.05)" if white["p-value"] < 0.05 else "Homoskedasticity assumption holds (p >= 0.05)"}

"""

    report += """**Visual Check:** See diagnostic plots - Scale-Location Plot (residuals should be randomly scattered)

---

### 2.3 Normality of Residuals

Tests whether the residuals follow a normal distribution.

"""

    # Normality tests
    if "Shapiro-Wilk" in norm_tests:
        sw = norm_tests["Shapiro-Wilk"]
        report += f"""#### Shapiro-Wilk Test

| Statistic | Value |
|-----------|-------|
| W Statistic | {sw["statistic"]:.4f} |
| p-value | {sw["p-value"]:.4e} |
"""
        if "note" in sw:
            report += f"\n*{sw['note']}*\n"

        report += f"\n**Interpretation:** {'Residuals are NOT normally distributed (p < 0.05)' if sw['p-value'] < 0.05 else 'Residuals are normally distributed (p >= 0.05)'}\n\n"

    if "Jarque-Bera" in norm_tests:
        jb = norm_tests["Jarque-Bera"]
        report += f"""#### Jarque-Bera Test

| Statistic | Value |
|-----------|-------|
| JB Statistic | {jb["statistic"]:.4f} |
| p-value | {jb["p-value"]:.4e} |
| Skewness | {jb["skewness"]:.4f} |
| Kurtosis | {jb["kurtosis"]:.4f} |

**Interpretation:** {"Residuals are NOT normally distributed (p < 0.05)" if jb["p-value"] < 0.05 else "Residuals are normally distributed (p >= 0.05)"}

"""

    report += """**Visual Check:** See diagnostic plots - Normal Q-Q Plot and Histogram of Residuals

---

### 2.4 Multicollinearity

Variance Inflation Factor (VIF) measures how much the variance of a coefficient is inflated due to collinearity.

**Rule of Thumb:** VIF > 10 indicates problematic multicollinearity; VIF > 5 may be concerning.

#### Top 15 Features by VIF

"""

    # VIF table
    report += "| Feature | VIF |\n|---------|-----|\n"
    for idx, row in vif_data.head(15).iterrows():
        vif_value = row["VIF"]
        warning = " ⚠️" if vif_value > 10 else " ⚡" if vif_value > 5 else ""
        report += f"| {row['Feature']} | {vif_value:.2f}{warning} |\n"

    high_vif_count = len(vif_data[vif_data["VIF"] > 10])
    moderate_vif_count = len(vif_data[(vif_data["VIF"] > 5) & (vif_data["VIF"] <= 10)])

    report += f"""
**Summary:**
- {high_vif_count} features with VIF > 10 (severe multicollinearity)
- {moderate_vif_count} features with VIF between 5-10 (moderate multicollinearity)

"""

    report += """---

### 2.5 Independence of Errors

**Durbin-Watson Statistic:** Values range from 0 to 4, with 2 indicating no autocorrelation.

"""

    dw = durbin_watson(model.resid)
    report += f"""| Statistic | Value |
|-----------|-------|
| Durbin-Watson | {dw:.4f} |

**Interpretation:** """

    if 1.5 < dw < 2.5:
        report += "No significant autocorrelation detected.\n"
    elif dw < 1.5:
        report += "Positive autocorrelation may be present.\n"
    else:
        report += "Negative autocorrelation may be present.\n"

    report += """
---

## 3. Model Coefficients

### 3.1 Top 20 Significant Coefficients (by absolute t-statistic)

"""

    # Coefficients table
    coef_summary = pd.DataFrame(
        {
            "Feature": ["Intercept"] + features,
            "Coefficient": model.params.values,
            "Std Error": model.bse.values,
            "t-statistic": model.tvalues.values,
            "p-value": model.pvalues.values,
            "CI Lower": model.conf_int()[0].values,
            "CI Upper": model.conf_int()[1].values,
        }
    )

    coef_summary["Abs_t"] = coef_summary["t-statistic"].abs()
    coef_summary = coef_summary.sort_values("Abs_t", ascending=False)

    report += "| Feature | Coefficient | Std Error | t-stat | p-value | 95% CI |\n"
    report += "|---------|-------------|-----------|--------|---------|--------|\n"

    for idx, row in coef_summary.head(20).iterrows():
        sig = (
            "***"
            if row["p-value"] < 0.001
            else "**"
            if row["p-value"] < 0.01
            else "*"
            if row["p-value"] < 0.05
            else ""
        )
        report += (
            f"| {row['Feature']} | {row['Coefficient']:.2f} | {row['Std Error']:.2f} | "
        )
        report += f"{row['t-statistic']:.2f} | {row['p-value']:.4f} {sig} | "
        report += f"[{row['CI Lower']:.2f}, {row['CI Upper']:.2f}] |\n"

    report += """
**Significance codes:** *** p < 0.001, ** p < 0.01, * p < 0.05

**Interpretation:** Each coefficient represents the expected change in price (in dollars) for a one-unit change
in that feature, holding all other features constant.

---

### 3.2 Most Impactful Features (Top 15 by Coefficient Magnitude)

"""

    top_impact = coef_summary[coef_summary["Feature"] != "Intercept"].nlargest(
        15, "Abs_t"
    )

    report += "| Rank | Feature | Impact on Price | p-value |\n"
    report += "|------|---------|----------------|----------|\n"

    for rank, (idx, row) in enumerate(top_impact.iterrows(), 1):
        impact = (
            f"+${row['Coefficient']:.2f}"
            if row["Coefficient"] > 0
            else f"-${abs(row['Coefficient']):.2f}"
        )
        sig = (
            "***"
            if row["p-value"] < 0.001
            else "**"
            if row["p-value"] < 0.01
            else "*"
            if row["p-value"] < 0.05
            else ""
        )
        report += (
            f"| {rank} | {row['Feature']} | {impact} | {row['p-value']:.4f} {sig} |\n"
        )

    report += """
---

## 4. Model Performance Metrics

### 4.1 Prediction Accuracy

"""

    # Calculate prediction metrics
    y_pred = model.fittedvalues
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    report += f"""| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | ${mae:.2f} |
| Root Mean Squared Error (RMSE) | ${rmse:.2f} |
| Mean Absolute Percentage Error (MAPE) | {mape:.2f}% |

**Interpretation:**
- On average, predictions are off by ${mae:.2f} (MAE)
- The model's predictions deviate by approximately {mape:.2f}% from actual prices (MAPE)

---

## 5. Diagnostic Visualizations

### 5.1 Available Plots

"""

    if "diagnostic_plot" in plots:
        report += f"""1. **Diagnostic Plots** - `{plots["diagnostic_plot"]}`
   - Residuals vs Fitted Values
   - Normal Q-Q Plot
   - Scale-Location Plot
   - Histogram of Residuals
   - Residuals vs Leverage
   - Cook's Distance
   - Actual vs Predicted
   - Top 10 Features by Coefficient
   - Prediction Error Distribution

"""

    if "correlation_plot" in plots:
        report += f"""2. **Correlation Heatmap** - `{plots["correlation_plot"]}`
   - Shows relationships between variables

"""

    if "distribution_plot" in plots:
        report += f"""3. **Variable Distributions** - `{plots["distribution_plot"]}`
   - Shows distribution of key variables

"""

    report += """---

## 6. Conclusions and Recommendations

### 6.1 Model Validity

"""

    # Generate conclusions based on tests
    issues = []
    strengths = []

    if model.f_pvalue < 0.05:
        strengths.append("Model is statistically significant overall")
    else:
        issues.append("Model is NOT statistically significant (p >= 0.05)")

    if model.rsquared > 0.7:
        strengths.append(f"Strong explanatory power (R² = {model.rsquared:.4f})")
    elif model.rsquared > 0.5:
        strengths.append(f"Moderate explanatory power (R² = {model.rsquared:.4f})")
    else:
        issues.append(f"Weak explanatory power (R² = {model.rsquared:.4f})")

    if "Breusch-Pagan" in het_tests and het_tests["Breusch-Pagan"]["p-value"] < 0.05:
        issues.append(
            "Heteroskedasticity detected - consider using robust standard errors or transforming the target variable"
        )

    if "Shapiro-Wilk" in norm_tests and norm_tests["Shapiro-Wilk"]["p-value"] < 0.05:
        issues.append(
            "Residuals are not normally distributed - may affect inference but not necessarily predictions"
        )

    if high_vif_count > 0:
        issues.append(
            f"{high_vif_count} features with severe multicollinearity (VIF > 10) - consider removing or combining correlated features"
        )

    report += "**Strengths:**\n"
    for strength in strengths:
        report += f"- ✓ {strength}\n"

    report += "\n**Potential Issues:**\n"
    if issues:
        for issue in issues:
            report += f"- ⚠️ {issue}\n"
    else:
        report += "- No major issues detected\n"

    report += """
### 6.2 Recommendations

1. **If heteroskedasticity is present:** Consider using weighted least squares (WLS) or transforming the target variable (e.g., log transformation)

2. **If multicollinearity is severe:** Consider:
   - Removing highly correlated features
   - Using regularization techniques (Ridge or Lasso regression)
   - Combining correlated features through PCA or domain knowledge

3. **If normality assumption is violated:**
   - This primarily affects confidence intervals and hypothesis tests
   - Predictions may still be reliable
   - Consider robust regression methods if extreme outliers are present

4. **Model Improvements:**
   - Investigate outliers and influential points (see Cook's distance plot)
   - Consider non-linear transformations of variables
   - Explore interaction terms between important features
   - Cross-validate the model on held-out data

---

## 7. Data Summary

"""

    report += f"""| Information | Value |
|-------------|-------|
| Total Observations | {data_info["n_observations"]} |
| Number of Features | {data_info["n_features"]} |
| Target Variable | price |
| Mean Price | ${data_info["mean_price"]:.2f} |
| Median Price | ${data_info["median_price"]:.2f} |
| Std Dev Price | ${data_info["std_price"]:.2f} |
| Min Price | ${data_info["min_price"]:.2f} |
| Max Price | ${data_info["max_price"]:.2f} |

---

*Report generated automatically by model_validation_report.py*
"""

    return report


def main():
    print("Loading data and configuration...")
    config = read_config()
    make = config["make"]
    model_name = config["model"]

    # Load and prepare data
    data = load_data(f"data/{make}_{model_name}_data.pkl")

    # Filter out auction items if BidCount exists
    try:
        original_size = len(data)
        data = data[data["BidCount"].isnull()].reset_index(drop=True)
        print(f"Filtered out {original_size - len(data)} auction items")
    except KeyError:
        pass

    print("Applying feature engineering...")
    data = feature_engineering(data, config)

    # Remove ListingId if present
    if "ListingId" in data.columns:
        data = data.drop("ListingId", axis=1)

    # Separate target and features
    y = data["price"]
    X = data.drop("price", axis=1)

    # Ensure all columns are numeric
    print("Converting all columns to numeric...")
    print(f"Dtypes before conversion:\n{X.dtypes.value_counts()}")

    # Convert all to numeric, forcing float type
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fill any NaN values with 0
    X = X.fillna(0)

    # Convert to float explicitly
    X = X.astype(float)

    # Ensure y is numeric
    y = pd.to_numeric(y, errors="coerce")
    y = y.astype(float)

    # Remove any rows where y is NaN
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]

    print(f"Dtypes after conversion:\n{X.dtypes.value_counts()}")

    # Store data info
    data_info = {
        "n_observations": len(y),
        "n_features": len(X.columns),
        "mean_price": y.mean(),
        "median_price": y.median(),
        "std_price": y.std(),
        "min_price": y.min(),
        "max_price": y.max(),
    }

    print(f"Data shape: {X.shape}")
    print(f"Target variable: {y.name}, n={len(y)}")
    print(
        f"All columns numeric: {X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]}"
    )

    # Fit OLS regression with statsmodels
    print("\nFitting OLS regression model...")
    X_with_const = add_constant(X)
    ols_model = sm.OLS(y, X_with_const).fit()

    print(f"R-squared: {ols_model.rsquared:.4f}")
    print(f"Adjusted R-squared: {ols_model.rsquared_adj:.4f}")

    # Run statistical tests
    print("\nRunning heteroskedasticity tests...")
    het_tests = test_heteroskedasticity(ols_model, X_with_const, y)

    print("Running normality tests...")
    norm_tests = test_normality(ols_model.resid)

    print("Calculating VIF for multicollinearity...")
    vif_data = calculate_vif(X_with_const, X_with_const.columns.tolist())

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plots = {}
    plots["diagnostic_plot"] = generate_diagnostic_plots(
        ols_model,
        X_with_const,
        y,
        X.columns.tolist(),
        output_prefix=f"reports/{make}_{model_name}_diagnostic",
    )

    print("Generating correlation heatmap...")
    plots["correlation_plot"] = generate_correlation_heatmap(
        data, output_prefix=f"reports/{make}_{model_name}_correlation"
    )

    print("Generating variable distributions...")
    plots["distribution_plot"] = generate_variable_distributions(
        data, output_prefix=f"reports/{make}_{model_name}_distributions"
    )

    # Generate report
    print("\nGenerating markdown report...")
    report = generate_markdown_report(
        ols_model,
        X_with_const,
        y,
        X.columns.tolist(),
        het_tests,
        norm_tests,
        vif_data,
        config,
        data_info,
        plots,
    )

    # Save report
    report_filename = f"reports/{make}_{model_name}_validation_report.md"
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"\n✓ Report saved to: {report_filename}")
    print(f"✓ Diagnostic plots saved to: {plots['diagnostic_plot']}")
    print(f"✓ Correlation heatmap saved to: {plots['correlation_plot']}")
    print(f"✓ Distribution plots saved to: {plots['distribution_plot']}")

    # Also print summary to console
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(ols_model.summary())


if __name__ == "__main__":
    main()

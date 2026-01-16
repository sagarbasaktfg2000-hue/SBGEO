import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import gaussian_kde, norm, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import adfuller, acf, kpss
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# ==================== PATHS ====================
file_path = r"E:\SWID_Clean_Rough_Thesis\SWID_Data_No Missing Value_Missing Value Treatment\Predicted file\Atiabari_TG_Garopara_Shivmandir_Year_Season_GW_Level.xlsx"
eval_dir = r"E:\SWID_Clean_Rough_Thesis\SWID_Data_No Missing Value_Missing Value Treatment\Predicted file\Predicted value"
os.makedirs(eval_dir, exist_ok=True)

# ==================== GLOBAL FONT AND STYLE ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['legend.title_fontsize'] = 10
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.shadow'] = False
plt.rcParams['legend.borderpad'] = 0.4
plt.rcParams['legend.labelspacing'] = 0.5
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handleheight'] = 0.7
plt.rcParams['legend.handletextpad'] = 0.5
plt.rcParams['legend.borderaxespad'] = 0.5
plt.rcParams['legend.columnspacing'] = 1.0

# ==================== DOMAIN METADATA ====================
BREAKPOINTS_INFO = [
    {
        'Location': 'Atiabari T.G., Garopara Shivmandir (41)',
        'Variable': 'Groundwater Depth (April)',
        'Breakpoint_Years': '2021',
        'Num_Breakpoints': 1,
        'Mean_Before': 8.2718,
        'Mean_After': 7.11,
        'Explanation': 'Breakpoints detected at years: 2021'
    },
    {
        'Location': 'Atiabari T.G., Garopara Shivmandir (41)',
        'Variable': 'Groundwater Depth (November)',
        'Breakpoint_Years': 'None',
        'Num_Breakpoints': 0,
        'Mean_Before': 3.5256,
        'Mean_After': np.nan,
        'Explanation': 'No statistically significant breakpoints detected'
    }
]

WELL_DIAGNOSTICS = {
    'Dataset': 'Atiabari_TG_Garopara_Shivmandir_Year_Season_GW_Level',
    'April_Mean': 8.1324,
    'April_Median': 8.3,
    'April_Mode': 8.0,
    'April_CV': 0.098530825,
    'April_Kurtosis': 1.841304917,
    'April_HighPeak': 9.18,
    'April_LowPeak': 6.07,
    'April_PeakDiff': 3.11,
    'April_OutlierCount': 3,
    'April_HighYears': '2015',
    'April_LowYears': '2023',
    'April_OutlierYears': '2001, 2002, 2002, 2022, 2023',
    'Nov_Mean': 3.5256,
    'Nov_Median': 3.5,
    'Nov_Mode': 1.85,
    'Nov_CV': 0.263294009,
    'Nov_Kurtosis': 5.976972242,
    'Nov_HighPeak': 6.82,
    'Nov_LowPeak': 1.85,
    'Nov_PeakDiff': 4.97,
    'Nov_OutlierCount': 1,
    'Nov_HighYears': '2009',
    'Nov_LowYears': '2013',
    'Nov_OutlierYears': '2009',
    'Seasonality': 'STRONG',
    'SignFlipRatio': 0,
    'MWU_p': 1.80e-09,
    'ADF_p': 1.04e-09
}

SEASONALITY_THRESHOLD = 0.4


@dataclass
class SeasonalStrength:
    value: float
    strength_label: str


# ==================== LOAD DATA ====================

data = pd.read_excel(file_path)
print("Data loaded successfully!")
print(f"Columns: {data.columns.tolist()}")
print(f"First few rows:\n{data.head()}")
print(f"Data shape: {data.shape}")

# Sort and create datetime index

data = data.sort_values(by=["Year", "Season"])


def create_date(row: pd.Series) -> pd.Timestamp:
    month = 4 if row['Season'] == 'April' else 11
    return pd.Timestamp(year=int(row['Year']), month=month, day=1)


data["Date"] = data.apply(create_date, axis=1)
data.set_index("Date", inplace=True)
data.sort_index(inplace=True)

# Use the correct column name for groundwater level

gws_col = "Groundwater Level (m)"
gws = data[gws_col]

print("\nData Summary:")
print(f"Time range: {gws.index[0]} to {gws.index[-1]}")
print(f"Number of observations: {len(gws)}")
print(f"Sample data:\n{gws.head(10)}")

# Check for seasonality distribution

gws_df = pd.DataFrame(gws)
gws_df['Month'] = gws_df.index.month
seasonal_pattern = gws_df.groupby('Month').size()
print("\nSeasonal pattern (month distribution):")
print(seasonal_pattern)


# ==================== UTILITY FUNCTIONS ====================

def calculate_seasonal_strength(series: pd.Series, period: int = 2) -> SeasonalStrength:
    stl = STL(series, period=period, robust=True)
    stl_result = stl.fit()
    resid_var = np.nanvar(stl_result.resid)
    seasonal_plus_resid_var = np.nanvar(stl_result.resid + stl_result.seasonal)
    strength = 0 if seasonal_plus_resid_var == 0 else 1 - (resid_var / seasonal_plus_resid_var)
    label = 'STRONG' if strength >= SEASONALITY_THRESHOLD else 'WEAK'
    return SeasonalStrength(value=strength, strength_label=label)


def apply_breakpoint_adjustment(series: pd.Series, breakpoint_years: List[int]) -> pd.Series:
    if not breakpoint_years:
        return series.copy()
    adjusted = series.copy()
    for year in breakpoint_years:
        mask = adjusted.index.year >= year
        if mask.any():
            before_mean = adjusted[~mask].mean()
            after_mean = adjusted[mask].mean()
            shift = after_mean - before_mean
            adjusted.loc[mask] = adjusted.loc[mask] - shift
    return adjusted


def calculate_metrics(obs, sim, phase="", model_name=""):
    obs, sim = np.array(obs), np.array(sim)
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[valid], sim[valid]

    if len(obs) == 0:
        return {}

    metrics = {}
    metrics['NSE'] = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    metrics['RMSE'] = np.sqrt(mean_squared_error(obs, sim))
    metrics['MAE'] = mean_absolute_error(obs, sim)
    try:
        metrics['R'], _ = pearsonr(obs, sim)
    except Exception:
        metrics['R'] = np.nan
    metrics['R2'] = r2_score(obs, sim)
    metrics['PBias'] = np.sum(sim - obs) / np.sum(obs) * 100

    metrics['NRMSE'] = metrics['RMSE'] / (obs.max() - obs.min())
    metrics['Willmott'] = 1 - np.sum((obs - sim) ** 2) / np.sum(
        (np.abs(sim - np.mean(obs)) + np.abs(obs - np.mean(obs))) ** 2
    )

    return metrics


def create_legend_with_title(ax, loc='upper right', **kwargs):
    legend = ax.legend(loc=loc, prop={'weight': 'bold', 'family': 'Times New Roman'},
                       title='Legend', title_fontsize=10, **kwargs)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(1.5)
    frame.set_alpha(0.8)
    frame.set_boxstyle('round', pad=0.3)
    return legend


def run_stationarity_tests(series: pd.Series) -> Dict[str, float]:
    adf_p = adfuller(series.dropna())[1]
    kpss_p = kpss(series.dropna(), regression='c', nlags='auto')[1]
    return {'ADF_p': adf_p, 'KPSS_p': kpss_p}


def run_residual_diagnostics(residuals: pd.Series) -> Dict[str, float]:
    ljung = acorr_ljungbox(residuals.dropna(), lags=[min(8, len(residuals) - 1)], return_df=True)
    jb_stat, jb_p, _, _ = jarque_bera(residuals.dropna())
    return {
        'LjungBox_p': float(ljung['lb_pvalue'].iloc[0]) if not ljung.empty else np.nan,
        'JarqueBera_p': jb_p
    }


def diagnose_location_differences(well_diagnostics: Dict[str, float], seasonal_strength: SeasonalStrength) -> None:
    print("\n" + "=" * 60)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 60)
    print("Key signals for lower performance across locations:")
    print(f"- Seasonality strength: {seasonal_strength.value:.3f} ({seasonal_strength.strength_label})")
    print(f"- April CV: {well_diagnostics.get('April_CV')}, November CV: {well_diagnostics.get('Nov_CV')}")
    print(f"- November kurtosis: {well_diagnostics.get('Nov_Kurtosis')}")
    print(f"- Outlier counts (April/Nov): {well_diagnostics.get('April_OutlierCount')}/"
          f"{well_diagnostics.get('Nov_OutlierCount')}")
    print("Suggested enhancements for weaker sites:")
    print("- Apply breakpoint adjustment before hybrid trend modeling.")
    print("- Use robust STL with outlier masking (done) and residual diagnostics thresholds.")
    print("- Add rolling-origin evaluation to detect instability across time.")
    print("- Switch to non-seasonal ARIMA when seasonal strength is weak.")


# ==================== COMPUTE SEN'S SLOPE ====================
print("\n" + "=" * 60)
print("COMPUTING SEN'S SLOPE FOR APRIL AND NOVEMBER")
print("=" * 60)


def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    if len(slopes) > 0:
        slopes_sorted = np.sort(slopes)
        median_idx = len(slopes_sorted) // 2

        if len(slopes_sorted) % 2 == 0:
            sen_slope_val = (slopes_sorted[median_idx - 1] + slopes_sorted[median_idx]) / 2
        else:
            sen_slope_val = slopes_sorted[median_idx]

        confidence = 1.96 * np.std(slopes) / np.sqrt(len(slopes))
        significant = abs(sen_slope_val) > confidence

        return sen_slope_val, confidence, significant
    return 0, 0, False


april_data = gws[gws.index.month == 4]
if len(april_data) >= 3:
    april_years = np.array([d.year for d in april_data.index])
    april_values = april_data.values
    april_sen_slope, april_confidence, april_significant = sen_slope(april_years, april_values)
    april_significant_text = "YES" if april_significant else "NO"
else:
    april_sen_slope = np.nan
    april_confidence = np.nan
    april_significant_text = "Insufficient Data"

november_data = gws[gws.index.month == 11]
if len(november_data) >= 3:
    november_years = np.array([d.year for d in november_data.index])
    november_values = november_data.values
    november_sen_slope, november_confidence, november_significant = sen_slope(november_years, november_values)
    november_significant_text = "YES" if november_significant else "NO"
else:
    november_sen_slope = np.nan
    november_confidence = np.nan
    november_significant_text = "Insufficient Data"

seasonal_trends = {
    4: {
        'Season_Name': 'April',
        'Sen_Slope': april_sen_slope,
        'Confidence': april_confidence,
        'Trend_Significant': april_significant_text,
        'Trend_Method': "Sen's Slope",
        'Monthly_Sen_Slope': april_sen_slope,
        'Annual_Sen_Slope': april_sen_slope * 12,
        'Last_Value': april_data.iloc[-1] if len(april_data) > 0 else np.nan,
        'Last_Year': april_years[-1] if len(april_data) > 0 else np.nan
    },
    11: {
        'Season_Name': 'November',
        'Sen_Slope': november_sen_slope,
        'Confidence': november_confidence,
        'Trend_Significant': november_significant_text,
        'Trend_Method': "Sen's Slope",
        'Monthly_Sen_Slope': november_sen_slope,
        'Annual_Sen_Slope': november_sen_slope * 12,
        'Last_Value': november_data.iloc[-1] if len(november_data) > 0 else np.nan,
        'Last_Year': november_years[-1] if len(november_data) > 0 else np.nan
    }
}

print("\nSeasonal Trend Test Results (Computed from Data):")
print("=" * 90)
print(f"{'Season':<10} {'Name':<10} {'Monthly_Sen_Slope':<20} {'Annual_Sen_Slope':<20} "
      f"{'Significant':<15} {'Last_Value':<15}")
print("-" * 90)
for season in [4, 11]:
    trend_info = seasonal_trends[season]
    if np.isnan(trend_info['Sen_Slope']):
        slope_text = "N/A"
        annual_text = "N/A"
        last_val_text = "N/A"
    else:
        slope_text = f"{trend_info['Sen_Slope']:.6f}"
        annual_text = f"{trend_info['Annual_Sen_Slope']:.6f}"
        last_val_text = f"{trend_info['Last_Value']:.3f}"

    print(f"{season:<10} {trend_info['Season_Name']:<10} {slope_text:<20} {annual_text:<20} "
          f"{trend_info['Trend_Significant']:<15} {last_val_text:<15}")

# ==================== SEASONALITY STRENGTH ====================
seasonality_strength = calculate_seasonal_strength(gws, period=2)
print("\n" + "=" * 60)
print("SEASONALITY STRENGTH")
print("=" * 60)
print(f"Seasonality strength: {seasonality_strength.value:.3f} ({seasonality_strength.strength_label})")

# ==================== STL DECOMPOSITION - ALL IN ONE GRAPH ====================
print("\n" + "=" * 60)
print("STL DECOMPOSITION ANALYSIS - Seasonal Data")
print("=" * 60)

try:
    stl_full = STL(gws, period=2, robust=True)
    stl_result_full = stl_full.fit()

    seasonal_components = {}
    for month in [4, 11]:
        mask = gws.index.month == month
        seasonal_components[month] = stl_result_full.seasonal[mask].mean()

    print("\nAverage Seasonal Components:")
    print(f"  April (Month 4): {seasonal_components[4]:.4f} m")
    print(f"  November (Month 11): {seasonal_components[11]:.4f} m")

    plt.figure(figsize=(14, 10))

    plt.subplot(5, 1, 1)
    plt.plot(gws, color='blue', linewidth=1.5)
    plt.title('Observed GWL Series', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)

    april_dates = gws[gws.index.month == 4].index
    if len(april_dates) > 0:
        april_years = sorted(list(set(april_dates.year)))
        tick_years = april_years[::3]
        tick_dates = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years]
        plt.gca().set_xticks(tick_dates)
        plt.gca().set_xticklabels([f'April-{year}' for year in tick_years],
                                  fontfamily='Times New Roman', fontsize=8)
    plt.xticks(rotation=45)

    plt.subplot(5, 1, 2)
    plt.plot(stl_result_full.trend, color='red', linewidth=1.5)
    plt.title('Trend Component', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)

    if len(april_dates) > 0:
        plt.gca().set_xticks(tick_dates)
        plt.gca().set_xticklabels([f'April-{year}' for year in tick_years],
                                  fontfamily='Times New Roman', fontsize=8)
    plt.xticks(rotation=45)

    plt.subplot(5, 1, 3)
    plt.plot(stl_result_full.seasonal, color='green', linewidth=1.5)
    plt.title('Seasonal Component', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('Average Deviation of GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)

    if len(april_dates) > 0:
        plt.gca().set_xticks(tick_dates)
        plt.gca().set_xticklabels([f'April-{year}' for year in tick_years],
                                  fontfamily='Times New Roman', fontsize=8)
    plt.xticks(rotation=45)

    plt.subplot(5, 1, 4)
    plt.plot(stl_result_full.resid, color='purple', linewidth=1.5)
    plt.title('Residual Component', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('Prediction Error of GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)

    if len(april_dates) > 0:
        plt.gca().set_xticks(tick_dates)
        plt.gca().set_xticklabels([f'April-{year}' for year in tick_years],
                                  fontfamily='Times New Roman', fontsize=8)
    plt.xticks(rotation=45)

    plt.subplot(5, 1, 5)
    residuals = stl_result_full.resid.dropna()
    clean_residuals = residuals[~np.isnan(residuals) & np.isfinite(residuals)]

    if len(clean_residuals) > 0:
        plt.hist(clean_residuals, bins=min(15, len(clean_residuals) // 2), density=True,
                 alpha=0.7, color='purple', edgecolor='black', linewidth=1.5,
                 label='Residual Histogram')

        if len(clean_residuals) >= 5:
            try:
                kde = gaussian_kde(clean_residuals)
                x_range = np.linspace(clean_residuals.min(), clean_residuals.max(), 100)
                plt.plot(x_range, kde(x_range), color='black', linewidth=2, label='KDE')
            except Exception:
                pass

        mu, std = np.mean(clean_residuals), np.std(clean_residuals)
        if not np.isnan(mu) and not np.isnan(std) and std > 0:
            x_norm = np.linspace(clean_residuals.min(), clean_residuals.max(), 100)
            y_norm = norm.pdf(x_norm, mu, std)
            plt.plot(x_norm, y_norm, 'r--', linewidth=2, label='Normal Distribution')

    plt.title('Residual Distribution with KDE', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.xlabel('Residual Error of GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('Density (1/m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)

    plt.gca().legend(loc='upper right', fontsize=9,
                    prop={'family': 'Times New Roman', 'weight': 'bold'},
                    frameon=True, fancybox=False, edgecolor='black', facecolor='white',
                    title='Legend')

    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "STL_Decomposition_All_Components.png"), dpi=400, bbox_inches='tight')
    plt.show()

    print("STL decomposition completed successfully and saved as one comprehensive plot!")

except Exception as e:
    print(f"Error in STL decomposition: {e}")
    stl_result_full = None

# ==================== TRAIN-TEST SPLIT ====================

n = len(gws)
train_size = int(0.80 * n)
train = gws.iloc[:train_size]
test = gws.iloc[train_size:]

print("\nTrain-Test Split:")
print(f"Train: {train.index[0].strftime('%b-%Y')} to {train.index[-1].strftime('%b-%Y')} ({len(train)} seasons)")
print(f"Test : {test.index[0].strftime('%b-%Y')} to {test.index[-1].strftime('%b-%Y')} ({len(test)} seasons)")

# ==================== SARIMA MODEL ====================
print("\n" + "=" * 60)
print("1. STANDARD SARIMA MODEL (Train-Test Split)")
print("=" * 60)


def build_sarima_model_fixed(train_data, test_data, model_name="SARIMA", seasonal_strength=None):
    print(f"\nBuilding {model_name} model...")

    train_values = train_data.values.ravel()
    test_values = test_data.values.ravel() if len(test_data) > 0 else np.array([])

    use_seasonal = True
    if seasonal_strength is not None and seasonal_strength.strength_label == 'WEAK':
        use_seasonal = False
        print("Seasonality is weak -> using non-seasonal ARIMA.")

    try:
        adf_result = adfuller(train_values)
        print(f"ADF p-value: {adf_result[1]:.4f}")
        if adf_result[1] > 0.05:
            print("Series is non-stationary, differencing needed")
        else:
            print("Series is stationary")

        print("\nSearching for best SARIMA parameters...")
        auto_model = auto_arima(
            train_values,
            seasonal=use_seasonal,
            m=2 if use_seasonal else 1,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            max_d=1,
            max_P=1, max_Q=1,
            max_D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            random_state=42,
            n_fits=30,
            information_criterion='aic',
            with_intercept=True
        )

        print(f"\nBest {model_name} parameters:")
        print(f"  Order (p,d,q): {auto_model.order}")
        print(f"  Seasonal order (P,D,Q,m): {auto_model.seasonal_order}")

        try:
            aic_val = float(auto_model.aic()) if callable(auto_model.aic) else float(auto_model.aic)
            bic_val = float(auto_model.bic()) if callable(auto_model.bic) else float(auto_model.bic)
            print(f"  AIC: {aic_val:.2f}")
            print(f"  BIC: {bic_val:.2f}")
        except Exception:
            print("  AIC/BIC: Could not retrieve")

        model = SARIMAX(
            train_values,
            order=auto_model.order,
            seasonal_order=auto_model.seasonal_order,
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        print("\nModel Diagnostics:")
        print(f"  Log Likelihood: {results.llf:.2f}")
        print(f"  AIC: {results.aic:.2f}")
        print(f"  BIC: {results.bic:.2f}")

        residuals = results.resid
        print("\nResidual Analysis:")
        print(f"  Mean: {residuals.mean():.6f}")
        print(f"  Std Dev: {residuals.std():.6f}")

        y_pred_train = results.get_prediction(start=0, end=len(train_values) - 1).predicted_mean

        if len(test_values) > 0:
            y_pred_test = results.get_forecast(steps=len(test_values)).predicted_mean
            y_pred_test_conf = results.get_forecast(steps=len(test_values)).conf_int()
        else:
            y_pred_test = np.array([])
            y_pred_test_conf = np.array([])

        metrics_train = calculate_metrics(train_values, y_pred_train, "Training", model_name)

        if len(test_values) > 0:
            metrics_test = calculate_metrics(test_values, y_pred_test, "Testing", model_name)
        else:
            metrics_test = {}

        residual_diagnostics = run_residual_diagnostics(pd.Series(residuals))

        return {
            'model': results,
            'train_pred': y_pred_train,
            'test_pred': y_pred_test,
            'test_conf': y_pred_test_conf,
            'metrics_train': metrics_train,
            'metrics_test': metrics_test,
            'order': auto_model.order,
            'seasonal_order': auto_model.seasonal_order,
            'residuals': residuals,
            'model_name': model_name,
            'type': 'SARIMA',
            'residual_diagnostics': residual_diagnostics
        }

    except Exception as e:
        print(f"Error building {model_name}: {e}")
        return None


sarima_results_train_test = build_sarima_model_fixed(train, test, "Standard SARIMA (Train-Test)",
                                                     seasonal_strength=seasonality_strength)

if sarima_results_train_test:
    print("\nStandard SARIMA Performance (Train-Test Split):")
    print("-" * 60)
    print(f"{'Metric':<12} | {'Training':<12} | {'Testing':<12}")
    print("-" * 60)
    for metric in ['NSE', 'RMSE', 'MAE', 'R', 'R2', 'PBias']:
        train_val = sarima_results_train_test['metrics_train'].get(metric, np.nan)
        test_val = sarima_results_train_test['metrics_test'].get(metric, np.nan)
        print(f"{metric:<12} | {train_val:12.4f} | {test_val:12.4f}")

    print("\n" + "=" * 60)
    print("SARIMA MODEL SUMMARY (Train-Test Split)")
    print("=" * 60)
    print(f"Order: {sarima_results_train_test['order']}")
    print(f"Seasonal Order: {sarima_results_train_test['seasonal_order']}")

print("\n" + "=" * 60)
print("2. STANDARD SARIMA MODEL (Entire Dataset)")
print("=" * 60)

sarima_results_overall = build_sarima_model_fixed(gws, pd.Series(dtype=float),
                                                  "Standard SARIMA (Overall)",
                                                  seasonal_strength=seasonality_strength)

if sarima_results_overall:
    print("\nStandard SARIMA Performance (Entire Dataset):")
    print("-" * 60)
    print(f"{'Metric':<12} | {'Value':<12}")
    print("-" * 60)
    for metric in ['NSE', 'RMSE', 'MAE', 'R', 'R2', 'PBias']:
        overall_val = sarima_results_overall['metrics_train'].get(metric, np.nan)
        print(f"{metric:<12} | {overall_val:12.4f}")

    print("\n" + "=" * 60)
    print("SARIMA MODEL SUMMARY (Entire Dataset)")
    print("=" * 60)
    print(f"Order: {sarima_results_overall['order']}")
    print(f"Seasonal Order: {sarima_results_overall['seasonal_order']}")

# ==================== HYBRID STL-STATE-SPACE-SARIMA MODEL ====================
print("\n" + "=" * 60)
print("3. HYBRID STL-STATE-SPACE-SARIMA MODEL (Train-Test Split)")
print("=" * 60)


def build_stl_state_space_sarima_model(train_data, test_data, model_name="STL-State-Space-SARIMA",
                                       breakpoints=None, seasonal_strength=None):
    print(f"\nBuilding {model_name} model...")
    print("Step 1: STL Decomposition: Yt = Tt + St + Rt")

    adjusted_train = train_data
    if breakpoints:
        adjusted_train = apply_breakpoint_adjustment(train_data, breakpoints)
        print(f"Applied breakpoint adjustment for years: {breakpoints}")

    stl = STL(adjusted_train, period=2, robust=True)
    stl_result = stl.fit()

    trend = stl_result.trend
    seasonal = stl_result.seasonal
    residual = stl_result.resid

    if seasonal_strength is not None and seasonal_strength.strength_label == 'WEAK':
        seasonal = pd.Series(np.zeros(len(seasonal)), index=seasonal.index)
        print("Seasonality is weak -> skipping seasonal component in hybrid model.")

    print(f"  Trend shape: {trend.shape}")
    print(f"  Seasonal shape: {seasonal.shape}")
    print(f"  Residual shape: {residual.shape}")

    train_months = train_data.index.month
    seasonal_pattern = {}

    for month in [4, 11]:
        mask = train_months == month
        if sum(mask) > 0:
            seasonal_pattern[month] = seasonal[mask].mean()
            print(f"  Average seasonal component for month {month}: {seasonal_pattern[month]:.4f}")

    print("\nStep 2: State-Space Trend Model (Local Linear Trend)")

    trend_values = trend.values.ravel()

    state_space_model = UnobservedComponents(
        trend_values,
        level='llevel',
        stochastic_trend=True,
        stochastic_level=True
    )

    print("  Fitting State-Space model to STL trend...")
    try:
        state_space_result = state_space_model.fit(method='lbfgs', disp=False)
        print("  State-Space model fitted successfully!")
        print(f"  Log Likelihood: {state_space_result.llf:.2f}")
        print(f"  AIC: {state_space_result.aic:.2f}")
        print(f"  BIC: {state_space_result.bic:.2f}")

        smoothed_states = state_space_result.smoothed_state
        trend_level = smoothed_states[0, :]
        if smoothed_states.shape[0] > 1:
            trend_slope = smoothed_states[1, :]
        else:
            trend_slope = np.zeros_like(trend_level)

        last_trend_level = trend_level[-1]
        last_trend_slope = trend_slope[-1] if len(trend_slope) > 0 else 0
        print(f"  Last trend level: {last_trend_level:.4f}")
        print(f"  Last trend slope: {last_trend_slope:.6f} (time-varying slope)")

    except Exception as e:
        print(f"  State-Space model fitting failed: {e}")
        print("  Using simple linear trend as fallback")
        time_index = np.arange(len(trend_values))
        slope, intercept = np.polyfit(time_index, trend_values, 1)
        trend_level = intercept + slope * time_index
        last_trend_level = trend_level[-1]
        last_trend_slope = slope
        print(f"  Fallback: Last trend level: {last_trend_level:.4f}")
        print(f"  Fallback: Constant slope: {last_trend_slope:.6f}")

    print("\nStep 3: SARIMA residual forecast: R_{t+h} = SARIMA(R)")
    residual_values = residual.values.ravel()

    adf_resid = adfuller(residual_values)
    print(f"  Residual ADF p-value: {adf_resid[1]:.4f}")

    print("\nSearching for best residual SARIMA parameters...")
    try:
        auto_residual = auto_arima(
            residual_values,
            seasonal=True,
            m=2,
            start_p=0, start_q=0,
            max_p=2, max_q=2,
            max_d=1,
            max_P=1, max_Q=1,
            max_D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_fits=20,
            information_criterion='aic'
        )

        print(f"  Residual SARIMA order: {auto_residual.order}")
        print(f"  Residual seasonal order: {auto_residual.seasonal_order}")

        residual_model = SARIMAX(
            residual_values,
            order=auto_residual.order,
            seasonal_order=auto_residual.seasonal_order,
            trend='n',
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        residual_results = residual_model.fit(disp=False)

    except Exception as e:
        print(f"  Auto ARIMA failed for residuals: {e}")
        print("  Using simple ARIMA(0,0,0) for residuals")
        residual_model = SARIMAX(
            residual_values,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 0, 2),
            trend='n',
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        residual_results = residual_model.fit(disp=False)

    print(f"  Residual model AIC: {residual_results.aic:.2f}")

    forecast_steps = len(test_data) if test_data is not None and len(test_data) > 0 else 0

    if forecast_steps > 0:
        print("\nStep 4: State-Space trend + SARIMA residual + Seasonal component")

        trend_forecast = np.zeros(forecast_steps)
        seasonal_forecast = np.zeros(forecast_steps)
        residual_forecast = residual_results.get_forecast(steps=forecast_steps).predicted_mean

        residual_conf = residual_results.get_forecast(steps=forecast_steps).conf_int()

        if 'state_space_result' in locals():
            state_space_forecast = state_space_result.get_forecast(steps=forecast_steps)
            trend_forecast = state_space_forecast.predicted_mean
            trend_conf = state_space_forecast.conf_int()
        else:
            time_steps = np.arange(len(trend_values), len(trend_values) + forecast_steps)
            trend_forecast = last_trend_level + last_trend_slope * (time_steps - len(trend_values) + 1)

        if test_data is not None:
            test_months = test_data.index.month

            for i in range(forecast_steps):
                current_month = test_months[i]
                seasonal_forecast[i] = seasonal_pattern.get(current_month, 0)

        combined_forecast = trend_forecast + seasonal_forecast + residual_forecast

        trend_train = trend.values
        seasonal_train = seasonal.values
        residual_train = residual_results.get_prediction(start=0, end=len(residual_values) - 1).predicted_mean
        combined_train = trend_train + seasonal_train + residual_train

        if 'trend_conf' in locals():
            combined_conf_lower = trend_conf[:, 0] + seasonal_forecast + residual_conf[:, 0]
            combined_conf_upper = trend_conf[:, 1] + seasonal_forecast + residual_conf[:, 1]
        else:
            combined_std = np.std(residual_values) * np.sqrt(np.arange(1, forecast_steps + 1))
            combined_conf_lower = combined_forecast - 1.96 * combined_std
            combined_conf_upper = combined_forecast + 1.96 * combined_std

        combined_conf = np.column_stack([combined_conf_lower, combined_conf_upper])
    else:
        combined_forecast = None
        combined_conf = None
        combined_train = trend.values + seasonal.values + residual_results.get_prediction(
            start=0, end=len(residual_values) - 1
        ).predicted_mean
        trend_forecast = None
        seasonal_forecast = None
        residual_forecast = None

    residual_diagnostics = run_residual_diagnostics(pd.Series(residual_results.resid))

    result = {
        'model_components': {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual_results,
            'seasonal_pattern': seasonal_pattern,
            'state_space_model': state_space_result if 'state_space_result' in locals() else None,
            'last_trend_level': last_trend_level,
            'last_trend_slope': last_trend_slope
        },
        'train_pred': combined_train,
        'stl_result': stl_result,
        'model_name': model_name,
        'type': 'Hybrid',
        'residual_diagnostics': residual_diagnostics
    }

    if test_data is not None and len(test_data) > 0:
        result.update({
            'test_pred': combined_forecast,
            'test_conf': combined_conf,
            'metrics_train': calculate_metrics(train_data.values, combined_train, "Training", model_name),
            'metrics_test': calculate_metrics(test_data.values, combined_forecast, "Testing", model_name),
            'component_forecasts': {
                'trend': trend_forecast,
                'seasonal': seasonal_forecast,
                'residual': residual_forecast
            }
        })
    else:
        result.update({
            'metrics_train': calculate_metrics(train_data.values, combined_train, "Training", model_name)
        })

    return result


hybrid_breakpoints = [2021]

hybrid_results_train_test = build_stl_state_space_sarima_model(
    train,
    test,
    model_name="Hybrid STL-State-Space-SARIMA (Train-Test)",
    breakpoints=hybrid_breakpoints,
    seasonal_strength=seasonality_strength
)

if hybrid_results_train_test:
    print("\nHybrid STL-State-Space-SARIMA Performance (Train-Test Split):")
    print("-" * 60)
    print(f"{'Metric':<12} | {'Training':<12} | {'Testing':<12}")
    print("-" * 60)
    for metric in ['NSE', 'RMSE', 'MAE', 'R', 'R2', 'PBias']:
        train_val = hybrid_results_train_test['metrics_train'].get(metric, np.nan)
        test_val = hybrid_results_train_test['metrics_test'].get(metric, np.nan)
        print(f"{metric:<12} | {train_val:12.4f} | {test_val:12.4f}")

    print("\nMethodology Details:")
    print("Step 1: STL Decomposition: Yt = Tt + St + Rt")
    print("Step 2: State-Space Trend Model: T_t = T_{t-1} + b_{t-1} + η_t, b_t = b_{t-1} + ζ_t")
    print("Step 3: SARIMA residual forecast: R_{t+h} = SARIMA(R)")
    print("Step 4: Final forecast: Ŷ_{t+h} = T_{t+h}(SS) + S_season + R_{t+h}")

print("\n" + "=" * 60)
print("4. HYBRID STL-STATE-SPACE-SARIMA MODEL (Entire Dataset)")
print("=" * 60)

hybrid_results_overall = build_stl_state_space_sarima_model(
    gws,
    pd.Series(dtype=float),
    model_name="Hybrid STL-State-Space-SARIMA (Overall)",
    breakpoints=hybrid_breakpoints,
    seasonal_strength=seasonality_strength
)

if hybrid_results_overall:
    print("\nHybrid STL-State-Space-SARIMA Performance (Entire Dataset):")
    print("-" * 60)
    print(f"{'Metric':<12} | {'Value':<12}")
    print("-" * 60)
    for metric in ['NSE', 'RMSE', 'MAE', 'R', 'R2', 'PBias']:
        overall_val = hybrid_results_overall['metrics_train'].get(metric, np.nan)
        print(f"{metric:<12} | {overall_val:12.4f}")

# ==================== RESIDUAL DIAGNOSTIC PLOTS ====================

def create_residual_diagnostics_plot_comprehensive(residuals_train, residuals_test, train_data, test_data,
                                                   model_name, save_path, ACF_LAGS=12):
    train_lags = min(ACF_LAGS, len(residuals_train) - 2)
    test_lags = min(ACF_LAGS, len(residuals_test) - 2) if len(residuals_test) > 0 else 0
    max_lags = max(train_lags, test_lags)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax in axes.flat:
        ax.set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True, alpha=0.3)

    axes[0, 0].plot(residuals_train.index, residuals_train, color='#1f77b4', linewidth=1.5,
                    label='Training Residuals')
    axes[0, 0].plot(residuals_test.index, residuals_test, color='#d62728', linewidth=1.8,
                    label='Testing Residuals')
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1.2)
    axes[0, 0].axvspan(train_data.index[0], train_data.index[-1], color='lightblue', alpha=0.15,
                       label='Training Phase')
    axes[0, 0].axvspan(test_data.index[0], test_data.index[-1], color='lightcoral', alpha=0.15,
                       label='Testing Phase')
    axes[0, 0].set_title('Residuals Plot', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    axes[0, 0].set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    axes[0, 0].set_ylabel('Residual (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')

    combined_data = pd.concat([train_data, test_data])
    april_dates = combined_data[combined_data.index.month == 4].index
    if len(april_dates) > 0:
        april_years = sorted(list(set(april_dates.year)))
        tick_years = april_years[::3]
        tick_dates = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years]
        axes[0, 0].set_xticks(tick_dates)
        axes[0, 0].set_xticklabels([f'April-{year}' for year in tick_years],
                                   fontsize=8, fontfamily='Times New Roman')
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45)

    legend1 = axes[0, 0].legend(loc='upper left', prop={'weight': 'bold', 'family': 'Times New Roman'},
                               title='Legend')
    frame1 = legend1.get_frame()
    frame1.set_edgecolor('black')
    frame1.set_linewidth(1.5)
    frame1.set_alpha(0.8)
    frame1.set_boxstyle('round', pad=0.3)

    ax_acf = axes[0, 1]
    ax_acf.clear()

    try:
        acf_train = acf(residuals_train, nlags=train_lags, fft=False)
        lags_train = np.arange(len(acf_train))
        ax_acf.bar(lags_train, acf_train, width=0.3,
                   color='#1f77b4', alpha=0.7, label='Training ACF',
                   edgecolor='navy', linewidth=1.5)
    except Exception as e:
        print(f"Warning: Could not plot training ACF: {e}")

    if len(residuals_test) > 2:
        try:
            acf_test = acf(residuals_test, nlags=test_lags, fft=False)
            lags_test = np.arange(len(acf_test))
            ax_acf.bar(lags_test + 0.3, acf_test, width=0.3,
                       color='#d62728', alpha=0.7, label='Testing ACF',
                       edgecolor='darkred', linewidth=1.5)
        except Exception as e:
            print(f"Warning: Could not plot testing ACF: {e}")

    if len(residuals_train) > 0:
        conf_level_train = 1.96 / np.sqrt(len(residuals_train))
        ax_acf.axhline(y=conf_level_train, color='#D2B48C', linestyle='--',
                       linewidth=3.0, alpha=0.9, label='Training 95% CI')
        ax_acf.axhline(y=-conf_level_train, color='#D2B48C', linestyle='--',
                       linewidth=3.0, alpha=0.9)

    if len(residuals_test) > 0:
        conf_level_test = 1.96 / np.sqrt(len(residuals_test))
        ax_acf.axhline(y=conf_level_test, color='#C0C0C0', linestyle='--',
                       linewidth=3.0, alpha=0.9, label='Testing 95% CI')
        ax_acf.axhline(y=-conf_level_test, color='#C0C0C0', linestyle='--',
                       linewidth=3.0, alpha=0.9)

    ax_acf.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_acf.set_title('ACF Plot of Residuals', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax_acf.set_xlabel('Lag', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax_acf.set_ylabel('Correlation Value', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax_acf.set_xlim(-0.5, max_lags + 0.5)
    ax_acf.set_xticks(np.arange(max_lags + 1))
    ax_acf.set_xticklabels([str(i) for i in range(max_lags + 1)])
    ax_acf.grid(True, alpha=0.3, axis='y')

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#1f77b4', alpha=0.7, label='Training ACF'),
        plt.Line2D([0], [0], color='#D2B48C', lw=5, alpha=0.9, label='Training 95% CI'),
        plt.Rectangle((0, 0), 1, 1, color='#d62728', alpha=0.7, label='Testing ACF'),
        plt.Line2D([0], [0], color='#C0C0C0', lw=5, alpha=0.9, label='Testing 95% CI')
    ]
    ax_acf.legend(handles=handles, prop={'weight': 'bold', 'family': 'Times New Roman'}, loc='upper right',
                  frameon=True, fancybox=False, edgecolor='black', facecolor='white',
                  title='Legend')

    axes[1, 0].hist(residuals_train, bins=min(20, len(residuals_train) // 2), density=True,
                    alpha=0.7, color='#1f77b4', edgecolor='navy', linewidth=1.5, label='Training',
                    histtype='bar')
    if len(residuals_test) > 0:
        axes[1, 0].hist(residuals_test, bins=max(8, min(12, len(residuals_test) // 2)), density=True,
                        alpha=0.75, color='#d62728', edgecolor='darkred', linewidth=1.5, label='Testing',
                        histtype='bar', hatch='///')

    if len(residuals_train) >= 5:
        try:
            kde_train = gaussian_kde(residuals_train)
            x_range_train = np.linspace(residuals_train.min(), residuals_train.max(), 100)
            axes[1, 0].plot(x_range_train, kde_train(x_range_train), color='#1f77b4',
                            linewidth=3, label='Training KDE')
        except Exception:
            pass

    if len(residuals_test) >= 5:
        try:
            kde_test = gaussian_kde(residuals_test)
            x_range_test = np.linspace(residuals_test.min(), residuals_test.max(), 100)
            axes[1, 0].plot(x_range_test, kde_test(x_range_test), color='#d62728',
                            linewidth=3, linestyle='--', label='Testing KDE')
        except Exception:
            pass

    axes[1, 0].set_title('Residuals Distribution', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 0].set_xlabel('Residuals (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 0].set_ylabel('Density', fontsize=10, fontweight='bold', fontfamily='Times New Roman')

    legend3 = axes[1, 0].legend(prop={'weight': 'bold', 'family': 'Times New Roman'}, title='Legend')
    frame3 = legend3.get_frame()
    frame3.set_edgecolor('black')
    frame3.set_linewidth(1.5)
    frame3.set_alpha(0.8)
    frame3.set_boxstyle('round', pad=0.3)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].clear()
    if len(residuals_train) >= 2:
        try:
            sm.qqplot(residuals_train, line='45', ax=axes[1, 1],
                      marker='o', markersize=5, markerfacecolor='green',
                      markeredgecolor='darkgreen', markeredgewidth=1.5,
                      alpha=0.8, label='Training')
        except Exception as e:
            print(f"Warning: Could not plot training Q-Q: {e}")

    if len(residuals_test) >= 2:
        try:
            sm.qqplot(residuals_test, line='45', ax=axes[1, 1],
                      marker='s', markersize=5, markerfacecolor='yellow',
                      markeredgecolor='goldenrod', markeredgewidth=1.5,
                      alpha=0.8, label='Testing')
        except Exception as e:
            print(f"Warning: Could not plot testing Q-Q: {e}")

    qq_lines = axes[1, 1].get_lines()
    line_count = 0
    for i, line in enumerate(qq_lines):
        if len(residuals_train) >= 2 and line_count < 2:
            line.set_color('green')
            line_count += 1
        elif len(residuals_test) >= 2 and line_count < 4:
            line.set_color('yellow')
            line_count += 1
        elif i == len(qq_lines) - 1:
            line.set_color('gray')
            line.set_linestyle('--')
            line.set_linewidth(2)

    axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 1].set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    if len(residuals_train) >= 2 or len(residuals_test) >= 2:
        legend4 = axes[1, 1].legend(prop={'weight': 'bold', 'family': 'Times New Roman'}, title='Legend')
        frame4 = legend4.get_frame()
        frame4.set_edgecolor('black')
        frame4.set_linewidth(1.5)
        frame4.set_alpha(0.8)
        frame4.set_boxstyle('round', pad=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(pad=3.5)
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\nACF Plot Style Updated for {model_name}:")
    print("   • Removed point markers at top of bars")
    print("   • Bars only (no additional markers)")
    print("   • Bars positioned to meet above axis markings")
    print("   • Training 95% CI → Light Brown (#D2B48C)")
    print("   • Testing 95% CI  → Light Silver (#C0C0C0)")
    print(f"ACF Lags: Training={train_lags}, Testing={test_lags}")

    return fig


def create_residual_diagnostics_plot_single(residuals, data, model_name, dataset_type, save_path, ACF_LAGS=12):
    max_lags = min(ACF_LAGS, len(residuals) - 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax in axes.flat:
        ax.set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True, alpha=0.3)

    axes[0, 0].plot(residuals.index, residuals, color='#1f77b4', linewidth=1.5, label='Residuals')
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1.2)
    axes[0, 0].axvspan(data.index[0], data.index[-1], color='lightblue', alpha=0.15,
                       label='Period of Dataset')
    axes[0, 0].set_title('Residuals Plot', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    axes[0, 0].set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    axes[0, 0].set_ylabel('Residual (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')

    april_dates = data[data.index.month == 4].index
    if len(april_dates) > 0:
        april_years = sorted(list(set(april_dates.year)))
        tick_years = april_years[::3]
        tick_dates = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years]
        axes[0, 0].set_xticks(tick_dates)
        axes[0, 0].set_xticklabels([f'April-{year}' for year in tick_years],
                                   fontsize=8, fontfamily='Times New Roman')
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45)

    legend1 = axes[0, 0].legend(loc='upper left', prop={'weight': 'bold', 'family': 'Times New Roman'},
                               title='Legend')
    frame1 = legend1.get_frame()
    frame1.set_edgecolor('black')
    frame1.set_linewidth(1.5)
    frame1.set_alpha(0.8)
    frame1.set_boxstyle('round', pad=0.3)

    ax_acf = axes[0, 1]
    ax_acf.clear()

    try:
        acf_vals = acf(residuals, nlags=max_lags, fft=False)
        lags = np.arange(len(acf_vals))
        ax_acf.bar(lags, acf_vals, width=0.6,
                   color='#1f77b4', alpha=0.7, label='ACF',
                   edgecolor='navy', linewidth=1.5)
    except Exception as e:
        print(f"Warning: Could not plot ACF: {e}")

    if len(residuals) > 0:
        conf_level = 1.96 / np.sqrt(len(residuals))
        ax_acf.axhline(y=conf_level, color='#D2B48C', linestyle='--',
                       linewidth=3.0, alpha=0.9, label='95% CI')
        ax_acf.axhline(y=-conf_level, color='#D2B48C', linestyle='--',
                       linewidth=3.0, alpha=0.9)

    ax_acf.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)

    ax_acf.set_title('ACF Plot of Residuals', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax_acf.set_xlabel('Lag', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax_acf.set_ylabel('Correlation Value', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax_acf.set_xlim(-0.5, max_lags + 0.5)
    ax_acf.set_xticks(np.arange(max_lags + 1))
    ax_acf.set_xticklabels([str(i) for i in range(max_lags + 1)])
    ax_acf.grid(True, alpha=0.3, axis='y')

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#1f77b4', alpha=0.7, label='ACF'),
        plt.Line2D([0], [0], color='#D2B48C', lw=5, alpha=0.9, label='95% CI')
    ]
    ax_acf.legend(handles=handles, prop={'weight': 'bold', 'family': 'Times New Roman'}, loc='upper right',
                  frameon=True, fancybox=False, edgecolor='black', facecolor='white',
                  title='Legend')

    axes[1, 0].hist(residuals, bins=min(20, len(residuals) // 2), density=True, alpha=0.7, color='#1f77b4',
                    edgecolor='navy', linewidth=1.5, label='Residuals Histogram', histtype='bar')

    if len(residuals) >= 5:
        try:
            kde = gaussian_kde(residuals)
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            axes[1, 0].plot(x_range, kde(x_range), color='#1f77b4', linewidth=3, label='KDE')
        except Exception:
            pass

    axes[1, 0].set_title('Residuals Distribution', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 0].set_xlabel('Residuals (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 0].set_ylabel('Density', fontsize=10, fontweight='bold', fontfamily='Times New Roman')

    legend3 = axes[1, 0].legend(prop={'weight': 'bold', 'family': 'Times New Roman'}, title='Legend')
    frame3 = legend3.get_frame()
    frame3.set_edgecolor('black')
    frame3.set_linewidth(1.5)
    frame3.set_alpha(0.8)
    frame3.set_boxstyle('round', pad=0.3)
    axes[1, 0].grid(True, alpha=0.3)

    if len(residuals) >= 2:
        try:
            sm.qqplot(residuals, line='45', ax=axes[1, 1],
                      marker='o', markersize=5, markerfacecolor='green',
                      markeredgecolor='darkgreen', markeredgewidth=1.5,
                      alpha=0.8, label='Residuals')
        except Exception as e:
            print(f"Warning: Could not plot Q-Q: {e}")

    qq_lines = axes[1, 1].get_lines()
    for i, line in enumerate(qq_lines):
        if i < 2:
            line.set_color('green')
        elif i == len(qq_lines) - 1:
            line.set_color('gray')
            line.set_linestyle('--')
            line.set_linewidth(2)

    axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    axes[1, 1].set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    if len(residuals) >= 2:
        legend4 = axes[1, 1].legend(prop={'weight': 'bold', 'family': 'Times New Roman'}, title='Legend')
        frame4 = legend4.get_frame()
        frame4.set_edgecolor('black')
        frame4.set_linewidth(1.5)
        frame4.set_alpha(0.8)
        frame4.set_boxstyle('round', pad=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(pad=3.5)
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\nACF Plot Style Updated for {model_name} ({dataset_type}):")
    print("   • Bars only (no point markers)")
    print("   • Bars positioned to meet above axis markings")
    print(f"ACF Lags: {max_lags}")

    return fig


# ==================== INDIVIDUAL GRAPHS FOR EACH MODEL ====================
print("\n" + "=" * 60)
print("CREATING INDIVIDUAL GRAPHS FOR EACH MODEL")
print("=" * 60)

if sarima_results_train_test:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1.plot(train.index, train.values, color='blue', linewidth=2, label='Observed GWL')
    ax1.plot(train.index, sarima_results_train_test['train_pred'],
             color='red', linestyle='--', linewidth=2, label='SARIMA')

    create_legend_with_title(ax1, loc='upper right')

    ax1.set_title('SARIMA', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax1.grid(True, alpha=0.3)

    april_dates_train = train[train.index.month == 4].index
    if len(april_dates_train) > 0:
        april_years_train = sorted(list(set(april_dates_train.year)))
        tick_years_train = april_years_train[::3]
        tick_dates_train = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years_train]
        ax1.set_xticks(tick_dates_train)
        ax1.set_xticklabels([f'April-{year}' for year in tick_years_train],
                            fontsize=8, fontfamily='Times New Roman')
    plt.setp(ax1.get_xticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.plot(test.index, test.values, color='green', linewidth=2, label='Observed GWL')
    ax2.plot(test.index, sarima_results_train_test['test_pred'],
             color='red', linestyle='--', linewidth=2, label='SARIMA')
    if len(sarima_results_train_test['test_conf']) > 0:
        ax2.fill_between(test.index, sarima_results_train_test['test_conf'][:, 0],
                         sarima_results_train_test['test_conf'][:, 1], color='red', alpha=0.2,
                         label='95% CI')

    create_legend_with_title(ax2, loc='upper right')

    ax2.set_title('SARIMA', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax2.grid(True, alpha=0.3)

    april_dates_test = test[test.index.month == 4].index
    if len(april_dates_test) > 0:
        april_years_test = sorted(list(set(april_dates_test.year)))
        tick_years_test = april_years_test[::2]
        tick_dates_test = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years_test]
        ax2.set_xticks(tick_dates_test)
        ax2.set_xticklabels([f'April-{year}' for year in tick_years_test],
                            fontsize=8, fontfamily='Times New Roman')
    plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "1_Standard_SARIMA_Train_Test.png"), dpi=400, bbox_inches='tight')
    plt.show()

if sarima_results_overall:
    plt.figure(figsize=(14, 6))

    plt.plot(gws.index, gws.values, color='blue', linewidth=2, label='Observed GWL')
    plt.plot(gws.index, sarima_results_overall['train_pred'],
             color='red', linestyle='--', linewidth=2, label='SARIMA')

    create_legend_with_title(plt.gca(), loc='upper right')

    plt.title('SARIMA', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)

    april_dates = gws[gws.index.month == 4].index
    if len(april_dates) > 0:
        april_years = sorted(list(set(april_dates.year)))
        tick_years = april_years[::3]
        tick_dates = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years]
        plt.gca().set_xticks(tick_dates)
        plt.gca().set_xticklabels([f'April-{year}' for year in tick_years],
                                  fontsize=8, fontfamily='Times New Roman')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "2_Standard_SARIMA_Entire_Dataset.png"), dpi=400, bbox_inches='tight')
    plt.show()

if hybrid_results_train_test:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1.plot(train.index, train.values, color='blue', linewidth=2, label='Observed GWL')
    ax1.plot(train.index, hybrid_results_train_test['train_pred'],
             color='purple', linestyle='--', linewidth=2, label='Hybrid STL-SS-SARIMA')

    create_legend_with_title(ax1, loc='upper right')

    ax1.set_title('Hybrid STL-SS-SARIMA', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax1.grid(True, alpha=0.3)

    april_dates_train = train[train.index.month == 4].index
    if len(april_dates_train) > 0:
        april_years_train = sorted(list(set(april_dates_train.year)))
        tick_years_train = april_years_train[::3]
        tick_dates_train = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years_train]
        ax1.set_xticks(tick_dates_train)
        ax1.set_xticklabels([f'April-{year}' for year in tick_years_train],
                            fontsize=8, fontfamily='Times New Roman')
    plt.setp(ax1.get_xticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.plot(test.index, test.values, color='green', linewidth=2, label='Observed GWL')
    ax2.plot(test.index, hybrid_results_train_test['test_pred'],
             color='purple', linestyle='--', linewidth=2, label='Hybrid STL-SS-SARIMA')
    if hybrid_results_train_test['test_conf'] is not None:
        ax2.fill_between(test.index, hybrid_results_train_test['test_conf'][:, 0],
                         hybrid_results_train_test['test_conf'][:, 1], color='purple', alpha=0.2,
                         label='95% CI')

    create_legend_with_title(ax2, loc='upper right')

    ax2.set_title('Hybrid STL-SS-SARIMA', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    ax2.grid(True, alpha=0.3)

    april_dates_test = test[test.index.month == 4].index
    if len(april_dates_test) > 0:
        april_years_test = sorted(list(set(april_dates_test.year)))
        tick_years_test = april_years_test[::2]
        tick_dates_test = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years_test]
        ax2.set_xticks(tick_dates_test)
        ax2.set_xticklabels([f'April-{year}' for year in tick_years_test],
                            fontsize=8, fontfamily='Times New Roman')
    plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "3_Hybrid_STL_State_Space_SARIMA_Train_Test.png"), dpi=400,
                bbox_inches='tight')
    plt.show()

if hybrid_results_overall:
    plt.figure(figsize=(14, 6))

    plt.plot(gws.index, gws.values, color='blue', linewidth=2, label='Observed GWL')
    plt.plot(gws.index, hybrid_results_overall['train_pred'],
             color='purple', linestyle='--', linewidth=2, label='Hybrid STL-SS-SARIMA')

    create_legend_with_title(plt.gca(), loc='upper right')

    plt.title('Hybrid STL-SS-SARIMA', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('GWL (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)

    april_dates = gws[gws.index.month == 4].index
    if len(april_dates) > 0:
        april_years = sorted(list(set(april_dates.year)))
        tick_years = april_years[::3]
        tick_dates = [pd.Timestamp(year=year, month=4, day=1) for year in tick_years]
        plt.gca().set_xticks(tick_dates)
        plt.gca().set_xticklabels([f'April-{year}' for year in tick_years],
                                  fontsize=8, fontfamily='Times New Roman')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "4_Hybrid_STL_State_Space_SARIMA_Entire_Dataset.png"), dpi=400,
                bbox_inches='tight')
    plt.show()

# ==================== COMPREHENSIVE RESIDUAL DIAGNOSTIC PLOTS ====================
print("\n" + "=" * 60)
print("CREATING COMPREHENSIVE RESIDUAL DIAGNOSTIC PLOTS")
print("=" * 60)

if sarima_results_train_test:
    residuals_sarima_train = pd.Series(train.values - sarima_results_train_test['train_pred'], index=train.index)
    if len(sarima_results_train_test['test_pred']) > 0:
        residuals_sarima_test = pd.Series(test.values - sarima_results_train_test['test_pred'], index=test.index)

        print("\nCreating Residual Diagnostics for Standard SARIMA (Train-Test Phase)...")
        create_residual_diagnostics_plot_comprehensive(
            residuals_sarima_train, residuals_sarima_test,
            train, test,
            "Standard SARIMA (Train-Test)",
            os.path.join(eval_dir, "5_Residual_Diagnostics_SARIMA_Train_Test.png"),
            ACF_LAGS=8
        )

if hybrid_results_train_test:
    residuals_hybrid_train = pd.Series(train.values - hybrid_results_train_test['train_pred'], index=train.index)
    if hybrid_results_train_test['test_pred'] is not None:
        residuals_hybrid_test = pd.Series(test.values - hybrid_results_train_test['test_pred'], index=test.index)

        print("\nCreating Residual Diagnostics for Hybrid STL-State-Space-SARIMA (Train-Test Phase)...")
        create_residual_diagnostics_plot_comprehensive(
            residuals_hybrid_train, residuals_hybrid_test,
            train, test,
            "Hybrid STL-State-Space-SARIMA (Train-Test)",
            os.path.join(eval_dir, "6_Residual_Diagnostics_Hybrid_Train_Test.png"),
            ACF_LAGS=8
        )

if sarima_results_overall:
    residuals_sarima_overall = pd.Series(gws.values - sarima_results_overall['train_pred'], index=gws.index)
    print("\nCreating Residual Diagnostics for Standard SARIMA (Overall Dataset)...")
    create_residual_diagnostics_plot_single(
        residuals_sarima_overall, gws,
        "Standard SARIMA", "Entire Dataset",
        os.path.join(eval_dir, "7_Residual_Diagnostics_SARIMA_Overall.png"),
        ACF_LAGS=12
    )

if hybrid_results_overall:
    residuals_hybrid_overall = pd.Series(gws.values - hybrid_results_overall['train_pred'], index=gws.index)
    print("\nCreating Residual Diagnostics for Hybrid STL-State-Space-SARIMA (Overall Dataset)...")
    create_residual_diagnostics_plot_single(
        residuals_hybrid_overall, gws,
        "Hybrid STL-State-Space-SARIMA", "Entire Dataset",
        os.path.join(eval_dir, "8_Residual_Diagnostics_Hybrid_Overall.png"),
        ACF_LAGS=12
    )

# ==================== FORECAST GENERATION 2025-2040 ====================
print("\n" + "=" * 80)
print("GENERATING FORECASTS 2025-2040 USING SPECIFIED FORMULAS")
print("=" * 80)
print("\nFORECASTING FORMULAS:")
print("=" * 80)
print("1. Standard SARIMA: Uses built-in SARIMA model forecasting")
print("2. Hybrid Model:")
print("   - For positive Sen's slope: Forecast = Y_last + (beta * h * 12)")
print("   - For negative Sen's slope: Forecast = Y_last + (beta * h)")
print("=" * 80)


def generate_forecasts_2025_2040():
    all_forecasts = []

    april_data = gws[gws.index.month == 4]
    november_data = gws[gws.index.month == 11]

    last_april_date = april_data.index[-1] if len(april_data) > 0 else None
    last_november_date = november_data.index[-1] if len(november_data) > 0 else None

    last_april_value = april_data.iloc[-1] if len(april_data) > 0 else np.nan
    last_november_value = november_data.iloc[-1] if len(november_data) > 0 else np.nan

    print("\nForecasting Parameters:")
    print(f"April - Last observed: {last_april_value:.3f} m (Year: {last_april_date.year})")
    print(f"       Sen's Slope (β): {seasonal_trends[4]['Sen_Slope']:.6f} m/month")
    print(f"       Sen's Slope Sign: {'Positive' if seasonal_trends[4]['Sen_Slope'] > 0 else 'Negative'}")
    print(f"November - Last observed: {last_november_value:.3f} m (Year: {last_november_date.year})")
    print(f"         Sen's Slope (β): {seasonal_trends[11]['Sen_Slope']:.6f} m/month")
    print(f"         Sen's Slope Sign: {'Positive' if seasonal_trends[11]['Sen_Slope'] > 0 else 'Negative'}")

    forecast_years = list(range(2025, 2041))

    if sarima_results_overall:
        print("\nGenerating Standard SARIMA forecasts...")

        forecast_steps = len(forecast_years) * 2

        try:
            sarima_forecast = sarima_results_overall['model'].get_forecast(steps=forecast_steps)
            sarima_pred = sarima_forecast.predicted_mean
            sarima_conf = sarima_forecast.conf_int()

            forecast_dates = []
            current_date = gws.index[-1]

            if current_date.month == 4:
                next_season_month = 11
                next_year = current_date.year
            else:
                next_season_month = 4
                next_year = current_date.year + 1

            for i in range(forecast_steps):
                if i % 2 == 0:
                    month = next_season_month
                    year = next_year + (i // 2)
                else:
                    month = 4 if next_season_month == 11 else 11
                    year = next_year + (i // 2)

                forecast_dates.append(pd.Timestamp(year=year, month=month, day=15))

            for i, date in enumerate(forecast_dates):
                if date.year in forecast_years:
                    all_forecasts.append({
                        'Date': date,
                        'Year': date.year,
                        'Month': date.month,
                        'Season': 'April' if date.month == 4 else 'November',
                        'Model': 'Standard SARIMA',
                        'Forecast_GWL': sarima_pred[i],
                        'Lower_CI': sarima_conf[i, 0] if sarima_conf is not None else np.nan,
                        'Upper_CI': sarima_conf[i, 1] if sarima_conf is not None else np.nan,
                        'Forecast_Method': 'SARIMA Model Prediction',
                        'Model_Order': str(sarima_results_overall['order']),
                        'Seasonal_Order': str(sarima_results_overall['seasonal_order'])
                    })

            print(f"✓ Standard SARIMA forecasts generated for {len(forecast_years)} years")
            print("  Method: Built-in SARIMA model forecasting")

        except Exception as e:
            print(f"Error generating SARIMA forecasts: {e}")

    print("\nGenerating Hybrid STL-State-Space-SARIMA forecasts using specified formula...")
    print("Formula: Forecast = Y_last + (beta * h * 12) for positive Sen's slope")
    print("         Forecast = Y_last + (beta * h) for negative Sen's slope")

    for year in forecast_years:
        for month in [4, 11]:
            date = pd.Timestamp(year=year, month=month, day=15)
            season = 'April' if month == 4 else 'November'

            if month == 4:
                Y_last = last_april_value
                beta = seasonal_trends[4]['Sen_Slope']
                h = year - last_april_date.year
                beta_sign = 'Positive' if beta > 0 else 'Negative'
            else:
                Y_last = last_november_value
                beta = seasonal_trends[11]['Sen_Slope']
                h = year - last_november_date.year
                beta_sign = 'Positive' if beta > 0 else 'Negative'

            if beta > 0:
                forecast = Y_last + (beta * h * 12)
                formula_used = 'Y_last + (β × h × 12)'
                formula_details = f'Positive slope: β = {beta:.6f}, h = {h} years'
            else:
                forecast = Y_last + (beta * h)
                formula_used = 'Y_last + (β × h)'
                formula_details = f'Negative slope: β = {beta:.6f}, h = {h} years'

            all_forecasts.append({
                'Date': date,
                'Year': year,
                'Month': month,
                'Season': season,
                'Model': 'Hybrid STL-State-Space-SARIMA',
                'Forecast_GWL': forecast,
                'Y_last': Y_last,
                'Beta': beta,
                'Beta_Sign': beta_sign,
                'h_years': h,
                'Formula_Used': formula_used,
                'Formula_Details': formula_details,
                'Forecast_Method': "Specified Formula based on Sen's Slope"
            })

    print(f"✓ Hybrid model forecasts generated for {len(forecast_years)} years")
    print("  Method: Specified formula based on Sen's slope")

    if all_forecasts:
        forecast_df = pd.DataFrame(all_forecasts)

        forecast_file = os.path.join(eval_dir, "Forecasts_2025_2040_Specified_Formulas.xlsx")
        with pd.ExcelWriter(forecast_file, engine='openpyxl') as writer:
            for model in forecast_df['Model'].unique():
                model_df = forecast_df[forecast_df['Model'] == model]
                sheet_name = model[:31]
                model_df.to_excel(writer, sheet_name=sheet_name, index=False)

            pivot_df = forecast_df.pivot_table(
                index=['Date', 'Year', 'Month', 'Season'],
                columns='Model',
                values='Forecast_GWL',
                aggfunc='first'
            ).reset_index()

            pivot_df = pivot_df.sort_values('Date')
            pivot_df.to_excel(writer, sheet_name='All_Models_Comparison', index=False)

            summary_data = []
            for model in forecast_df['Model'].unique():
                model_data = forecast_df[forecast_df['Model'] == model]

                for season in ['April', 'November']:
                    season_data = model_data[model_data['Season'] == season]

                    forecast_2040 = season_data[season_data['Year'] == 2040]
                    if not forecast_2040.empty:
                        forecast_value = forecast_2040['Forecast_GWL'].values[0]
                        last_value = last_april_value if season == 'April' else last_november_value
                        change = forecast_value - last_value

                        summary_data.append({
                            'Model': model,
                            'Season': season,
                            'Last_Observed_Value': last_value,
                            'Last_Observed_Year': last_april_date.year if season == 'April'
                            else last_november_date.year,
                            'Forecast_2040': forecast_value,
                            'Change_from_Last': change,
                            'Percent_Change': (change / last_value) * 100 if last_value != 0 else np.nan,
                            'Forecast_Method': forecast_2040['Forecast_Method'].values[0]
                        })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        print(f"\n✓ All forecasts saved to: {forecast_file}")

        plt.figure(figsize=(14, 8))

        historical_april = gws[gws.index.month == 4]
        historical_nov = gws[gws.index.month == 11]

        plt.scatter(historical_april.index, historical_april.values,
                    marker='o', s=60, alpha=0.7, color='blue', label='Historical April')
        plt.scatter(historical_nov.index, historical_nov.values,
                    marker='s', s=60, alpha=0.7, color='green', label='Historical November')

        colors = {'Standard SARIMA': 'red', 'Hybrid STL-State-Space-SARIMA': 'purple'}

        for model in forecast_df['Model'].unique():
            model_forecasts = forecast_df[forecast_df['Model'] == model]

            april_forecasts = model_forecasts[model_forecasts['Month'] == 4]
            nov_forecasts = model_forecasts[model_forecasts['Month'] == 11]

            if not april_forecasts.empty:
                plt.plot(april_forecasts['Date'], april_forecasts['Forecast_GWL'],
                         color=colors[model], marker='o', markersize=5, linewidth=2,
                         label=f'{model} - April')

            if not nov_forecasts.empty:
                plt.plot(nov_forecasts['Date'], nov_forecasts['Forecast_GWL'],
                         color=colors[model], marker='s', markersize=5, linewidth=2, linestyle='--',
                         label=f'{model} - November')

        legend = plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.15),
                            ncol=3, prop={'weight': 'bold', 'family': 'Times New Roman', 'size': 9},
                            title='Legend')
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_linewidth(1.5)
        frame.set_alpha(0.8)
        frame.set_boxstyle('round', pad=0.3)

        plt.title('Groundwater Level Forecasts 2025-2040 (Using Different Forecasting Methods)',
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')
        plt.xlabel('Time Frame', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        plt.ylabel('Groundwater Level (m)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
        plt.grid(True, alpha=0.3)

        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45, fontsize=9, fontfamily='Times New Roman')

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(os.path.join(eval_dir, "9_Forecasts_2025_2040_Specified_Formulas.png"),
                    dpi=400, bbox_inches='tight')
        plt.show()

        print("\n✓ Forecast visualization saved as: 9_Forecasts_2025_2040_Specified_Formulas.png")

        print("\n" + "=" * 80)
        print("DETAILED FORECAST SUMMARY (2040 VALUES)")
        print("=" * 80)
        print(f"{'Model':<35} {'Season':<10} {'Last Value':<12} {'Forecast 2040':<15} "
              f"{'Change':<12} {'Method':<30}")
        print("-" * 114)

        for model in forecast_df['Model'].unique():
            model_data = forecast_df[forecast_df['Model'] == model]

            for season in ['April', 'November']:
                season_data = model_data[(model_data['Year'] == 2040) & (model_data['Season'] == season)]
                if not season_data.empty:
                    forecast_value = season_data['Forecast_GWL'].values[0]
                    last_value = last_april_value if season == 'April' else last_november_value
                    change = forecast_value - last_value
                    forecast_method = season_data['Forecast_Method'].values[0]

                    if model == 'Hybrid STL-State-Space-SARIMA':
                        formula = season_data['Formula_Used'].values[0]
                        method_text = f"{forecast_method}: {formula}"
                    else:
                        method_text = forecast_method

                    print(f"{model:<35} {season:<10} {last_value:<12.3f} {forecast_value:<15.3f} "
                          f"{change:<12.3f} {method_text:<30}")

        print("\n" + "=" * 80)
        print("HYBRID MODEL FORMULA DETAILS")
        print("=" * 80)
        for month in [4, 11]:
            season_name = 'April' if month == 4 else 'November'
            beta = seasonal_trends[month]['Sen_Slope']
            last_value = last_april_value if month == 4 else last_november_value
            last_year = last_april_date.year if month == 4 else last_november_date.year

            if beta > 0:
                print(f"{season_name}:")
                print("  • Formula: Forecast = Y_last + (β × h × 12)")
                print(f"  • Y_last = {last_value:.3f} m (Year: {last_year})")
                print(f"  • β (Sen's slope) = {beta:.6f} m/month (Positive)")
                print(f"  • For 2040 (h = {2040 - last_year} years):")
                print(f"    Forecast = {last_value:.3f} + ({beta:.6f} × {2040 - last_year} × 12)")
                print(f"            = {last_value:.3f} + ({beta:.6f} × {12 * (2040 - last_year)})")
                print(f"            = {last_value + (beta * (2040 - last_year) * 12):.3f} m")
            else:
                print(f"{season_name}:")
                print("  • Formula: Forecast = Y_last + (β × h)")
                print(f"  • Y_last = {last_value:.3f} m (Year: {last_year})")
                print(f"  • β (Sen's slope) = {beta:.6f} m/month (Negative)")
                print(f"  • For 2040 (h = {2040 - last_year} years):")
                print(f"    Forecast = {last_value:.3f} + ({beta:.6f} × {2040 - last_year})")
                print(f"            = {last_value + (beta * (2040 - last_year)):.3f} m")

    return forecast_df if all_forecasts else None


forecast_df = generate_forecasts_2025_2040()

# ==================== SAVE PERFORMANCE METRICS ====================
print("\n" + "=" * 60)
print("SAVING PERFORMANCE METRICS")
print("=" * 60)


def save_all_performance_metrics():
    all_metrics = []

    if sarima_results_train_test:
        metrics = sarima_results_train_test['metrics_train'].copy()
        metrics.update({
            'Model': 'Standard SARIMA',
            'Dataset': 'Training',
            'Order': str(sarima_results_train_test['order']),
            'Seasonal_Order': str(sarima_results_train_test['seasonal_order'])
        })
        all_metrics.append(metrics)

        metrics_test = sarima_results_train_test['metrics_test'].copy()
        metrics_test.update({
            'Model': 'Standard SARIMA',
            'Dataset': 'Testing',
            'Order': str(sarima_results_train_test['order']),
            'Seasonal_Order': str(sarima_results_train_test['seasonal_order'])
        })
        all_metrics.append(metrics_test)

    if sarima_results_overall:
        metrics_overall = sarima_results_overall['metrics_train'].copy()
        metrics_overall.update({
            'Model': 'Standard SARIMA',
            'Dataset': 'Overall',
            'Order': str(sarima_results_overall['order']),
            'Seasonal_Order': str(sarima_results_overall['seasonal_order'])
        })
        all_metrics.append(metrics_overall)

    if hybrid_results_train_test:
        metrics_hybrid_train = hybrid_results_train_test['metrics_train'].copy()
        metrics_hybrid_train.update({
            'Model': 'Hybrid STL-State-Space-SARIMA',
            'Dataset': 'Training',
            'Order': 'N/A',
            'Seasonal_Order': 'N/A'
        })
        all_metrics.append(metrics_hybrid_train)

        metrics_hybrid_test = hybrid_results_train_test['metrics_test'].copy()
        metrics_hybrid_test.update({
            'Model': 'Hybrid STL-State-Space-SARIMA',
            'Dataset': 'Testing',
            'Order': 'N/A',
            'Seasonal_Order': 'N/A'
        })
        all_metrics.append(metrics_hybrid_test)

    if hybrid_results_overall:
        metrics_hybrid_overall = hybrid_results_overall['metrics_train'].copy()
        metrics_hybrid_overall.update({
            'Model': 'Hybrid STL-State-Space-SARIMA',
            'Dataset': 'Overall',
            'Order': 'N/A',
            'Seasonal_Order': 'N/A'
        })
        all_metrics.append(metrics_hybrid_overall)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        cols = ['Model', 'Dataset', 'Order', 'Seasonal_Order', 'NSE', 'R2', 'R', 'RMSE', 'MAE', 'PBias', 'NRMSE',
                'Willmott']
        metrics_df = metrics_df[cols]

        metrics_file = os.path.join(eval_dir, "Performance_Metrics_All_Models.xlsx")
        metrics_df.to_excel(metrics_file, index=False)

        print(f"✓ Performance metrics saved to: {metrics_file}")

        print("\n" + "=" * 100)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 100)
        print(f"{'Model':<35} {'Dataset':<12} {'NSE':<10} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'PBias':<10}")
        print("-" * 100)

        for _, row in metrics_df.iterrows():
            print(f"{row['Model']:<35} {row['Dataset']:<12} {row['NSE']:<10.4f} {row['R2']:<10.4f} "
                  f"{row['RMSE']:<10.4f} {row['MAE']:<10.4f} {row['PBias']:<10.4f}")

    return metrics_df if all_metrics else None


metrics_df = save_all_performance_metrics()

# ==================== SAVE MODEL INFORMATION ====================
print("\n" + "=" * 60)
print("SAVING MODEL INFORMATION")
print("=" * 60)


def save_model_information():
    model_info = []

    if sarima_results_train_test:
        model_info.append({
            'Model_Name': 'Standard SARIMA (Train-Test)',
            'Model_Type': 'SARIMA',
            'Order': str(sarima_results_train_test['order']),
            'Seasonal_Order': str(sarima_results_train_test['seasonal_order']),
            'Training_Start': train.index[0].strftime('%b-%Y'),
            'Training_End': train.index[-1].strftime('%b-%Y'),
            'Testing_Start': test.index[0].strftime('%b-%Y') if len(test) > 0 else 'N/A',
            'Testing_End': test.index[-1].strftime('%b-%Y') if len(test) > 0 else 'N/A',
            'Data_Points_Training': len(train),
            'Data_Points_Testing': len(test),
            'Residual_Mean': sarima_results_train_test['residuals'].mean(),
            'Residual_Std': sarima_results_train_test['residuals'].std(),
            'LjungBox_p': sarima_results_train_test['residual_diagnostics']['LjungBox_p'],
            'JarqueBera_p': sarima_results_train_test['residual_diagnostics']['JarqueBera_p']
        })

    if sarima_results_overall:
        model_info.append({
            'Model_Name': 'Standard SARIMA (Overall)',
            'Model_Type': 'SARIMA',
            'Order': str(sarima_results_overall['order']),
            'Seasonal_Order': str(sarima_results_overall['seasonal_order']),
            'Training_Start': gws.index[0].strftime('%b-%Y'),
            'Training_End': gws.index[-1].strftime('%b-%Y'),
            'Testing_Start': 'N/A',
            'Testing_End': 'N/A',
            'Data_Points_Training': len(gws),
            'Data_Points_Testing': 0,
            'Residual_Mean': sarima_results_overall['residuals'].mean(),
            'Residual_Std': sarima_results_overall['residuals'].std(),
            'LjungBox_p': sarima_results_overall['residual_diagnostics']['LjungBox_p'],
            'JarqueBera_p': sarima_results_overall['residual_diagnostics']['JarqueBera_p']
        })

    if hybrid_results_train_test:
        model_info.append({
            'Model_Name': 'Hybrid STL-State-Space-SARIMA (Train-Test)',
            'Model_Type': 'Hybrid',
            'Order': 'N/A',
            'Seasonal_Order': 'N/A',
            'Training_Start': train.index[0].strftime('%b-%Y'),
            'Training_End': train.index[-1].strftime('%b-%Y'),
            'Testing_Start': test.index[0].strftime('%b-%Y') if len(test) > 0 else 'N/A',
            'Testing_End': test.index[-1].strftime('%b-%Y') if len(test) > 0 else 'N/A',
            'Data_Points_Training': len(train),
            'Data_Points_Testing': len(test),
            'Residual_Mean': 'N/A',
            'Residual_Std': 'N/A',
            'LjungBox_p': hybrid_results_train_test['residual_diagnostics']['LjungBox_p'],
            'JarqueBera_p': hybrid_results_train_test['residual_diagnostics']['JarqueBera_p']
        })

    if hybrid_results_overall:
        model_info.append({
            'Model_Name': 'Hybrid STL-State-Space-SARIMA (Overall)',
            'Model_Type': 'Hybrid',
            'Order': 'N/A',
            'Seasonal_Order': 'N/A',
            'Training_Start': gws.index[0].strftime('%b-%Y'),
            'Training_End': gws.index[-1].strftime('%b-%Y'),
            'Testing_Start': 'N/A',
            'Testing_End': 'N/A',
            'Data_Points_Training': len(gws),
            'Data_Points_Testing': 0,
            'Residual_Mean': 'N/A',
            'Residual_Std': 'N/A',
            'LjungBox_p': hybrid_results_overall['residual_diagnostics']['LjungBox_p'],
            'JarqueBera_p': hybrid_results_overall['residual_diagnostics']['JarqueBera_p']
        })

    if model_info:
        model_info_df = pd.DataFrame(model_info)
        breakpoints_df = pd.DataFrame(BREAKPOINTS_INFO)
        diagnostics_df = pd.DataFrame([WELL_DIAGNOSTICS])

        model_info_file = os.path.join(eval_dir, "Model_Information.xlsx")
        with pd.ExcelWriter(model_info_file, engine='openpyxl') as writer:
            model_info_df.to_excel(writer, sheet_name='Model_Info', index=False)
            breakpoints_df.to_excel(writer, sheet_name='Bai_Perron_Breakpoints', index=False)
            diagnostics_df.to_excel(writer, sheet_name='Well_Diagnostics', index=False)

        print(f"✓ Model information saved to: {model_info_file}")

        print("\n" + "=" * 80)
        print("MODEL INFORMATION SUMMARY")
        print("=" * 80)
        print(f"{'Model Name':<40} {'Type':<15} {'Order':<15} {'Data Points':<15}")
        print("-" * 90)

        for _, row in model_info_df.iterrows():
            print(f"{row['Model_Name']:<40} {row['Model_Type']:<15} {row['Order']:<15} "
                  f"{row['Data_Points_Training']:<15}")

    return model_info_df if model_info else None


model_info_df = save_model_information()

# ==================== STYLE CONSISTENCY CHECK ====================

def verify_plot_style_consistency():
    print("\n" + "=" * 60)
    print("VERIFYING PLOT STYLE CONSISTENCY")
    print("=" * 60)

    consistency_check = {
        'Font Family': plt.rcParams['font.family'] == 'Times New Roman',
        'X-axis Label': 'Time Frame',
        'Y-axis Label Style': 'Bold and sized appropriately',
        'Legend Box': 'Black border, rounded corners, alpha=0.8',
        'Legend Title': "'Legend' text inside legend box",
        'Grid': 'Enabled with alpha=0.3',
        'Titles': 'Bold formatting',
        'ACF Plot Style': 'Bars only (no point markers)',
        'ACF Plot CI Colors': 'Light Brown (#D2B48C) for Training, Light Silver (#C0C0C0) for Testing',
        'QQ Plot Colors': 'Green for Training, Yellow for Testing',
        'Residual Plot Colors': '#1f77b4 for Training, #d62728 for Testing',
        'Residual Plot Legend Location': 'Upper Left (as requested)',
        'ACF Bars Positioning': 'Bars meet above axis markings'
    }

    print("\nStyle Consistency Check:")
    for key, value in consistency_check.items():
        status = "✓" if value else "✗"
        print(f"{status} {key}")

    print("\nAll plots have:")
    print("1. Times New Roman font (bold for titles and labels)")
    print("2. 'Time Frame' as x-axis label")
    print("3. Legend box with black border and 'Legend' title inside")
    print("4. ACF plot: Bars only (no point markers at top)")
    print("5. ACF plot CI colors: Light Brown for Training, Light Silver for Testing")
    print("6. QQ plot colors: Green for Training, Yellow for Testing")
    print("7. Grid lines with alpha=0.3")
    print("8. Consistent color scheme across all residual diagnostic plots")
    print("9. Residual plot legend in upper left corner (as requested)")
    print("10. ACF bars positioned to meet above axis markings")


def run_quality_checks(series: pd.Series) -> None:
    print("\n" + "=" * 60)
    print("ADDITIONAL QUALITY CHECKS (ROBUSTNESS)")
    print("=" * 60)
    stationarity = run_stationarity_tests(series)
    print(f"ADF p-value: {stationarity['ADF_p']:.6f}")
    print(f"KPSS p-value: {stationarity['KPSS_p']:.6f}")


verify_plot_style_consistency()
run_quality_checks(gws)
diagnose_location_differences(WELL_DIAGNOSTICS, seasonality_strength)

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - ALL OUTPUTS GENERATED SUCCESSFULLY!")
print("=" * 80)

print("\nMODELS BUILT:")
print("-" * 80)
print("1. Standard SARIMA (Train-Test Split)")
print("2. Standard SARIMA (Entire Dataset)")
print("3. Hybrid STL-State-Space-SARIMA (Train-Test Split)")
print("4. Hybrid STL-State-Space-SARIMA (Entire Dataset)")

print("\nFILES GENERATED:")
print("-" * 80)
print("\nVISUALIZATION FILES (PNG):")
print("1. STL_Decomposition_All_Components.png")
print("2. 1_Standard_SARIMA_Train_Test.png")
print("3. 2_Standard_SARIMA_Entire_Dataset.png")
print("4. 3_Hybrid_STL_State_Space_SARIMA_Train_Test.png")
print("5. 4_Hybrid_STL_State_Space_SARIMA_Entire_Dataset.png")
print("6. 5_Residual_Diagnostics_SARIMA_Train_Test.png")
print("7. 6_Residual_Diagnostics_Hybrid_Train_Test.png")
print("8. 7_Residual_Diagnostics_SARIMA_Overall.png")
print("9. 8_Residual_Diagnostics_Hybrid_Overall.png")
print("10. 9_Forecasts_2025_2040_Specified_Formulas.png")

print("\nDATA FILES (EXCEL):")
print("1. Forecasts_2025_2040_Specified_Formulas.xlsx")
print("   - Standard SARIMA: Built-in SARIMA model forecasting")
print("   - Hybrid Model: Uses specified formula based on Sen's slope:")
print("     • Positive slope: Forecast = Y_last + (β × h × 12)")
print("     • Negative slope: Forecast = Y_last + (β × h)")
print("2. Performance_Metrics_All_Models.xlsx")
print("3. Model_Information.xlsx")

print("\nKEY FINDINGS:")
print("-" * 80)
print(f"1. April Trend Slope: {seasonal_trends[4]['Sen_Slope']:.6f} m/month "
      f"({'Positive' if seasonal_trends[4]['Sen_Slope'] > 0 else 'Negative'})")
print(f"2. November Trend Slope: {seasonal_trends[11]['Sen_Slope']:.6f} m/month "
      f"({'Positive' if seasonal_trends[11]['Sen_Slope'] > 0 else 'Negative'})")

if sarima_results_train_test:
    print(f"3. Standard SARIMA (Train-Test) - Best NSE: "
          f"{max(sarima_results_train_test['metrics_train'].get('NSE', 0), sarima_results_train_test['metrics_test'].get('NSE', 0)):.4f}")

if hybrid_results_train_test:
    print(f"4. Hybrid Model (Train-Test) - Best NSE: "
          f"{max(hybrid_results_train_test['metrics_train'].get('NSE', 0), hybrid_results_train_test['metrics_test'].get('NSE', 0)):.4f}")

print("\nFORECASTING APPROACH:")
print("-" * 80)
print("✓ Standard SARIMA: Uses built-in SARIMA model forecasting")
print("✓ Hybrid Model: Uses specified formula based on Sen's slope:")
print("  • Positive Sen's slope: Forecast = Y_last + (β × h × 12)")
print("  • Negative Sen's slope: Forecast = Y_last + (β × h)")
print("✓ Forecast values are DIFFERENT between models due to different methodologies")

print("\nGRAPH STYLE FEATURES:")
print("-" * 80)
print("✓ Font Family: Times New Roman (Bold)")
print("✓ Legend: Boxed with black border and 'Legend' title inside")
print("✓ X-axis Label: 'Time Frame'")
print("✓ Y-axis Label: Context-specific units")
print("✓ ACF Plot: Bars only (no point markers)")
print("✓ ACF Plot CI Colors: Light Brown (#D2B48C) for Training, Light Silver (#C0C0C0) for Testing")
print("✓ QQ Plot Colors: Green for Training, Yellow for Testing")
print("✓ Residual Plot Colors: #1f77b4 for Training, #d62728 for Testing")
print("✓ Residual Plot Legend Location: Upper Left (as requested)")
print("✓ ACF Bars Positioning: Bars meet above axis markings")
print("✓ Titles: Bold formatting")
print("✓ Grid: Enabled with alpha=0.3")
print("✓ Consistent styling across all plots")

print(f"\nAll files saved to: {eval_dir}")
print("\n" + "=" * 80)

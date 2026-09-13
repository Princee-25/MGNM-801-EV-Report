"""
================================================================================
MGNM801 - BUSINESS ANALYTICS USING PYTHON
Descriptive and Predictive Analysis of a Current Business Problem Using
Live/External Online Data

BUSINESS PROBLEM:
An Indian EV two-wheeler manufacturer is planning its market expansion
strategy and needs data-driven guidance on (a) which states to prioritize
for new dealership/service-center investment, (b) how demand differs across
vehicle categories, (c) seasonal timing for production/inventory planning,
and (d) a forecast of near-term state-level sales to size that investment.

DATA SOURCE:
  Source name : Electric Vehicle Sales by State in India (Kaggle)
  URL         : https://www.kaggle.com/datasets/mafzal19/electric-vehicle-sales-by-state-in-india
  Underlying  : Vahan Dashboard, Ministry of Road Transport & Highways,
                Government of India (vahan.parivahan.gov.in)
  Collected on: [FILL IN YOUR DOWNLOAD DATE]
  Records     : 96,845 rows (raw) -> 95,985 after removing incomplete 2024 data
  Variables   : Year, Month_Name, Date, State, Vehicle_Class, Vehicle_Category,
                Vehicle_Type, EV_Sales_Quantity  (8 columns)
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.bbox'] = 'tight'
CHART_DIR = "charts"
import os
os.makedirs(CHART_DIR, exist_ok=True)

RANDOM_STATE = 42

# ==============================================================================
# SECTION 2: DATA COLLECTION  (Rubric: Collect External Online Data - 4 Marks)
# ==============================================================================
DATA_PATH = "EV_Dataset.csv"          # downloaded CSV from Kaggle (see source note above)
df_raw = pd.read_csv(DATA_PATH)

print("=" * 70)
print("SECTION 2: DATA COLLECTION")
print("=" * 70)
print(f"Raw records collected : {df_raw.shape[0]}")
print(f"Variables collected   : {df_raw.shape[1]} -> {list(df_raw.columns)}")
print(df_raw.head())

# ==============================================================================
# SECTION 3: DATA PREPARATION  (Rubric: Data Preparation - 4 Marks)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DATA PREPARATION")
print("=" * 70)

df = df_raw.copy()

print("\n--- .info() ---")
df.info()

print("\n--- .describe() ---")
print(df.describe())

print("\n--- Missing values per column ---")
print(df.isnull().sum())

print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

# 3a. Fix Date dtype.
# NOTE: the raw 'Date' string is formatted MM-DD-YYYY with the day fixed at 01
# (verified by cross-checking against the Month_Name column). Parsing it as
# DD-MM-YYYY silently forces every row into January - a real data-quality trap
# worth watching for.
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%Y')
df['Month_Num'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# 3b. Drop incomplete 2024 data (only January 2024 present -> would distort
# year-over-year and annual comparisons).
before_rows = df.shape[0]
df = df[df['Year'] <= 2023].copy()
print(f"\nDropped {before_rows - df.shape[0]} rows of incomplete 2024 data.")
print(f"Cleaned dataset shape: {df.shape}")

# 3c. Focused subset for the business problem: 2-Wheeler category
df_2w = df[df['Vehicle_Category'] == '2-Wheelers'].copy()
print(f"2-Wheeler subset shape: {df_2w.shape}")

# 3d. Feature engineering - State x Year aggregation with YoY growth
state_year = (df_2w.groupby(['State', 'Year'], as_index=False)['EV_Sales_Quantity']
              .sum()
              .sort_values(['State', 'Year']))
state_year['YoY_Growth_%'] = state_year.groupby('State')['EV_Sales_Quantity'].pct_change() * 100

# 3e. Feature engineering - State x Year x Month aggregation for modeling,
# plus a time index and a 1-month lag feature (previous month's sales for
# the same state), which is a strong predictor for short-term forecasting.
state_month = (df_2w.groupby(['State', 'Year', 'Month_Num'], as_index=False)['EV_Sales_Quantity']
               .sum()
               .sort_values(['State', 'Year', 'Month_Num']))
state_month['Time_Index'] = (state_month['Year'] - 2014) * 12 + state_month['Month_Num']
state_month['Quarter'] = ((state_month['Month_Num'] - 1) // 3) + 1
state_month['Lag_1_Sales'] = state_month.groupby('State')['EV_Sales_Quantity'].shift(1)
state_month = state_month.dropna(subset=['Lag_1_Sales']).reset_index(drop=True)

print("\nSample of engineered state_month table:")
print(state_month.head())

# ==============================================================================
# SECTION 4: DESCRIPTIVE ANALYTICS  (Rubric: Descriptive Analytics - 4 Marks)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: DESCRIPTIVE ANALYTICS")
print("=" * 70)

# 4a. National yearly trend (2-Wheelers)
national_yearly = df_2w.groupby('Year')['EV_Sales_Quantity'].sum()
print("\nNational yearly 2-Wheeler EV sales:")
print(national_yearly)

corr_year_sales = national_yearly.reset_index()['Year'].corr(
    national_yearly.reset_index()['EV_Sales_Quantity'])
print(f"\nCorrelation between Year and national 2W sales: {corr_year_sales:.2f}")

# 4b. Category share of the overall EV market (2018-2023, once volumes are meaningful)
cat_totals = df[df['Year'] >= 2018].groupby('Vehicle_Category')['EV_Sales_Quantity'].sum().sort_values(ascending=False)
cat_share = (cat_totals / cat_totals.sum() * 100).round(1)
print("\nVehicle category totals (2018-2023):")
print(cat_totals)
print("\nVehicle category share (%):")
print(cat_share)

# 4c. Top states by 2023 volume, with CAGR 2018-2023
pivot_state_year = df_2w.groupby(['State', 'Year'])['EV_Sales_Quantity'].sum().unstack(fill_value=0)

def calc_cagr(row, start_year=2018, end_year=2023):
    start = row.get(start_year, 0)
    end = row.get(end_year, 0)
    n_years = end_year - start_year
    if start <= 0:
        return np.nan
    return ((end / start) ** (1 / n_years) - 1) * 100

pivot_state_year['CAGR_2018_2023_%'] = pivot_state_year.apply(calc_cagr, axis=1)
pivot_state_year['Total_2023'] = pivot_state_year.get(2023, 0)
top10_states = pivot_state_year.sort_values('Total_2023', ascending=False).head(10)[['CAGR_2018_2023_%', 'Total_2023']]
print("\nTop 10 states by 2023 2-Wheeler EV sales (with CAGR 2018-2023):")
print(top10_states.round(1))

# 4d. Monthly seasonality
monthly_totals = df_2w.groupby('Month_Num')['EV_Sales_Quantity'].sum()
monthly_share = (monthly_totals / monthly_totals.sum() * 100).round(2)
print("\nMonthly seasonality - share of total 2W sales by calendar month (%):")
print(monthly_share)

# 4e. Summary statistics on state-month sales (2018-2023) - the core operating metric
summary_stats = state_month[state_month['Year'] >= 2018]['EV_Sales_Quantity'].describe()
print("\nSummary statistics - state-month 2W sales quantity (2018-2023):")
print(summary_stats)

# ==============================================================================
# SECTION 5: DATA VISUALIZATION  (Rubric: Data Visualization - 4 Marks, min 5 charts)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DATA VISUALIZATION (charts saved to ./charts/)")
print("=" * 70)

charts_saved = 0
charts_failed = []

def finish_chart(filename):
    """Save the current figure, try to display it inline (Jupyter/Colab),
    then close it. Wrapped by each chart block in a try/except so one
    failing chart never stops the rest of the script from running."""
    global charts_saved
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/{filename}')
    try:
        plt.show()   # shows inline if running in Jupyter/Colab; harmless no-op otherwise
    except Exception:
        pass
    plt.close()
    charts_saved += 1
    print(f"  Saved {filename}")

# --- Chart 1: LINE chart - National yearly 2-Wheeler EV sales trend ---
try:
    plt.figure(figsize=(8, 5))
    plt.plot(national_yearly.index, national_yearly.values, marker='o', linewidth=2, color='#1f77b4')
    plt.title('National 2-Wheeler EV Sales Trend, India (2014-2023)')
    plt.xlabel('Year')
    plt.ylabel('EV Sales Quantity (units)')
    plt.grid(alpha=0.3)
    finish_chart('chart1_national_yearly_trend.png')
except Exception as e:
    print(f"  [FAILED] Chart 1 (line trend): {e}")
    charts_failed.append(('chart1', str(e)))

# --- Chart 2: BAR chart - Top 10 states by 2023 sales volume ---
try:
    plt.figure(figsize=(9, 5.5))
    top10_sorted = top10_states.sort_values('Total_2023')
    plt.barh(top10_sorted.index, top10_sorted['Total_2023'], color='#2ca02c')
    plt.title('Top 10 States by 2-Wheeler EV Sales Volume, 2023')
    plt.xlabel('EV Sales Quantity (units)')
    finish_chart('chart2_top10_states_2023.png')
except Exception as e:
    print(f"  [FAILED] Chart 2 (bar - top states): {e}")
    charts_failed.append(('chart2', str(e)))

# --- Chart 3: HISTOGRAM - Distribution of state-month sales quantities ---
try:
    plt.figure(figsize=(8, 5))
    nonzero_vals = state_month.loc[state_month['EV_Sales_Quantity'] > 0, 'EV_Sales_Quantity']
    plt.hist(np.log10(nonzero_vals + 1), bins=30, color='#ff7f0e', edgecolor='white')
    plt.title('Distribution of State-Month 2W EV Sales (log10 scale, non-zero months)')
    plt.xlabel('log10(EV Sales Quantity + 1)')
    plt.ylabel('Frequency (number of state-months)')
    finish_chart('chart3_sales_distribution_histogram.png')
except Exception as e:
    print(f"  [FAILED] Chart 3 (histogram): {e}")
    charts_failed.append(('chart3', str(e)))

# --- Chart 4: SCATTER - Growth (CAGR) vs Scale (2023 volume) by state ---
try:
    plt.figure(figsize=(8, 6))
    plot_data = pivot_state_year.dropna(subset=['CAGR_2018_2023_%'])
    plt.scatter(plot_data['CAGR_2018_2023_%'], plot_data['Total_2023'], color='#9467bd', alpha=0.7)
    label_states = plot_data.sort_values('Total_2023', ascending=False).head(5)
    for state, row in label_states.iterrows():
        plt.annotate(state, (row['CAGR_2018_2023_%'], row['Total_2023']),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.title('State-Level Growth (CAGR) vs. Market Scale (2023 Volume)')
    plt.xlabel('CAGR 2018-2023 (%)')
    plt.ylabel('2023 EV Sales Volume (units)')
    plt.grid(alpha=0.3)
    finish_chart('chart4_growth_vs_scale_scatter.png')
except Exception as e:
    print(f"  [FAILED] Chart 4 (scatter): {e}")
    charts_failed.append(('chart4', str(e)))

# --- Chart 5: BOX PLOT - Seasonal spread of state sales by quarter ---
try:
    plt.figure(figsize=(8, 5))
    q_data = [state_month.loc[(state_month['Quarter'] == q) & (state_month['Year'] >= 2021), 'EV_Sales_Quantity']
              for q in [1, 2, 3, 4]]
    q_labels = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
    # matplotlib renamed the 'labels' kwarg to 'tick_labels' in v3.9+; try the
    # modern name first and fall back to the older one so this runs on any version.
    try:
        plt.boxplot(q_data, tick_labels=q_labels, showfliers=False)
    except TypeError:
        plt.boxplot(q_data, labels=q_labels, showfliers=False)
    plt.title('Spread of State-Level Monthly 2W EV Sales by Quarter (2021-2023)')
    plt.ylabel('EV Sales Quantity (units)')
    finish_chart('chart5_quarterly_boxplot.png')
except Exception as e:
    print(f"  [FAILED] Chart 5 (box plot): {e}")
    charts_failed.append(('chart5', str(e)))

# --- Chart 6 (bonus): BAR - Vehicle category market share ---
try:
    plt.figure(figsize=(7, 5))
    plt.bar(cat_share.index, cat_share.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.title('EV Market Share by Vehicle Category, India (2018-2023)')
    plt.ylabel('Share of Total EV Sales (%)')
    plt.xticks(rotation=20)
    finish_chart('chart6_category_share.png')
except Exception as e:
    print(f"  [FAILED] Chart 6 (category share bar): {e}")
    charts_failed.append(('chart6', str(e)))

print(f"\n{charts_saved} of 6 charts saved successfully to ./{CHART_DIR}/")
if charts_failed:
    print("The following charts failed - see error messages above:")
    for name, err in charts_failed:
        print(f"  - {name}: {err}")

# ==============================================================================
# SECTION 6: PREDICTIVE ANALYTICS  (Rubric: Predictive Analytics - 5 Marks)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: PREDICTIVE ANALYTICS")
print("=" * 70)

# Target variable  : EV_Sales_Quantity (monthly, by state)
# Predictors       : State (encoded), Year, Month_Num, Quarter, Time_Index, Lag_1_Sales
# Train/test split : TIME-BASED (train on 2018-2022, test on 2023) rather than random,
#                     since this mirrors how the manufacturer would actually use the
#                     model - forecasting a future period from past data.

model_data = state_month[state_month['Year'] >= 2018].copy()
model_data = pd.get_dummies(model_data, columns=['State'], drop_first=True)

feature_cols = [c for c in model_data.columns
                 if c not in ['EV_Sales_Quantity', 'Year']] + ['Year']
# (Year kept as both a raw feature and implicitly via Time_Index)

train_data = model_data[model_data['Year'] <= 2022]
test_data = model_data[model_data['Year'] == 2023]

X_train = train_data[feature_cols]
y_train = train_data['EV_Sales_Quantity']
X_test = test_data[feature_cols]
y_test = test_data['EV_Sales_Quantity']

print(f"Training rows: {X_train.shape[0]} (2018-2022) | Test rows: {X_test.shape[0]} (2023)")

# --- Model 1: Linear Regression (baseline) ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# --- Model 2: Random Forest Regressor (main model - captures non-linear state effects) ---
rf_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\nModel Performance on 2023 Test Set:")
print(f"{'Model':<20}{'R2':>8}{'MAE':>12}{'RMSE':>12}")
print(f"{'Linear Regression':<20}{lr_r2:>8.3f}{lr_mae:>12.1f}{lr_rmse:>12.1f}")
print(f"{'Random Forest':<20}{rf_r2:>8.3f}{rf_mae:>12.1f}{rf_rmse:>12.1f}")

# Feature importance from Random Forest (top predictors)
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 predictors (Random Forest feature importance):")
print(importances.head(10))

# --- Chart 7: Actual vs Predicted (Random Forest) - supports model evaluation narrative ---
try:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, rf_pred, alpha=0.5, color='#17becf')
    max_val = max(y_test.max(), rf_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction')
    plt.title('Random Forest: Actual vs. Predicted 2W EV Sales (2023 Test Set)')
    plt.xlabel('Actual Sales Quantity')
    plt.ylabel('Predicted Sales Quantity')
    plt.legend()
    finish_chart('chart7_actual_vs_predicted.png')
except Exception as e:
    print(f"  [FAILED] Chart 7 (actual vs predicted): {e}")
    charts_failed.append(('chart7', str(e)))

# --- Secondary model: STRUCTURAL DRIVERS (excludes Lag_1_Sales) ---
# Purpose: Lag_1_Sales dominates the forecasting model above (as expected -
# recent momentum is the strongest short-term signal). To answer the
# business question "which states and which months actually drive demand"
# (rather than just "what happened last month"), we fit a second Random
# Forest WITHOUT the lag feature. This surfaces State and seasonal effects
# that the lag feature was masking, and is more useful for the manufacturer's
# longer-range expansion planning (where a full sales history may not yet exist).
driver_features = [c for c in feature_cols if c != 'Lag_1_Sales']
X_train_d = train_data[driver_features]
X_test_d = test_data[driver_features]

rf_driver_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
rf_driver_model.fit(X_train_d, y_train)
rf_driver_pred = rf_driver_model.predict(X_test_d)

rf_driver_r2 = r2_score(y_test, rf_driver_pred)
rf_driver_mae = mean_absolute_error(y_test, rf_driver_pred)
rf_driver_rmse = np.sqrt(mean_squared_error(y_test, rf_driver_pred))

print("\nSecondary model - Structural Drivers (Random Forest, no lag feature):")
print(f"R2={rf_driver_r2:.3f}  MAE={rf_driver_mae:.1f}  RMSE={rf_driver_rmse:.1f}")

driver_importances = pd.Series(rf_driver_model.feature_importances_, index=driver_features).sort_values(ascending=False)
print("\nTop 10 structural predictors (state/seasonal drivers, no lag):")
print(driver_importances.head(10))

# ==============================================================================
# SECTION 7: BUSINESS INSIGHTS & RECOMMENDATIONS (printed summary)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: FINDINGS")
print("=" * 70)
print("""
1. National 2W EV sales grew from ~1,500 units/yr (2014-17) to 856,836 units
   in 2023, with a strong Year-Sales correlation (r ~ 0.77) - adoption is a
   sustained trend, not a one-off spike.
2. 2-Wheelers hold the largest share (52.1%) of India's EV market, narrowly
   ahead of 3-Wheelers (43.3%); 4-Wheelers remain a small niche (4.2%).
3. Maharashtra and Karnataka are the largest existing 2W EV markets, but
   Kerala (228% CAGR) and Gujarat (205% CAGR) are growing fastest off a
   smaller base - a "big vs. fast" trade-off for expansion planning.
4. October-December (festive season) accounts for ~32.6% of annual 2W sales,
   versus a June low of 5.8% - demand is strongly seasonal.
5. The forecasting model (with lag) reaches R2=0.867 on unseen 2023 data,
   driven overwhelmingly by prior-month sales - near-term demand is highly
   persistent. The structural-drivers model (without lag) isolates State and
   seasonal timing as the next most useful signals for longer-range planning
   where a sales history does not yet exist (e.g. a brand-new state market).
""")

print("=" * 70)
print("SECTION 7 (cont.): BUSINESS RECOMMENDATIONS")
print("=" * 70)
print("""
1. Prioritize Maharashtra and Karnataka for immediate dealership expansion -
   they combine the largest existing volume with strong, proven CAGR
   (121.7% and 149.1% respectively), minimizing market-entry risk.
2. Open a secondary watch-list for Kerala and Gujarat - their CAGR (228%
   and 205%) suggests they could overtake mid-tier states within 2-3 years;
   early positioning there is lower-cost than catching up later.
3. Align production and inventory build-up with the Q3 ramp-up (Jul-Sep)
   so stock is in place before the Oct-Dec festive-season demand spike,
   rather than reacting to it in November.
4. Treat 4-Wheeler EVs as a lower near-term priority for this manufacturer's
   expansion budget - at 4.2% market share they represent a smaller and
   slower-maturing opportunity than 2W/3W segments.
5. Use the forecasting model's monthly output as a rolling input to
   state-level inventory planning, but validate it against the structural-
   drivers model in any state with less than 12 months of sales history,
   since the lag feature is uninformative there.
""")

print("=" * 70)
print("SECTION 7 (cont.): CONCLUSION")
print("=" * 70)
print("""
India's 2-wheeler EV market has moved from a negligible niche to a
sustained, high-growth segment, concentrated in a handful of states and
strongly seasonal in timing. A manufacturer that sequences its expansion
around proven high-growth states, times production to the festive-season
demand curve, and uses both short-term (lag-based) and structural
(state/seasonal) forecasting models together will be better positioned
to allocate capital efficiently than one relying on national averages
or intuition alone.
""")

print("=" * 70)
print("SECTION 7 (cont.): REFERENCES")
print("=" * 70)
print("""
1. Kaggle. "Electric Vehicle Sales by State in India" dataset.
   https://www.kaggle.com/datasets/mafzal19/electric-vehicle-sales-by-state-in-india
   [FILL IN: your exact download date]
2. Vahan Dashboard, Ministry of Road Transport & Highways, Government of
   India. https://vahan.parivahan.gov.in/vahan4dashboard/ (underlying
   registration data source for most Indian EV sales datasets).
3. Python Software Foundation. Python Language Reference.
4. McKinney, W. pandas: powerful data structures for data analysis
   (pandas documentation, pandas.pydata.org).
5. Hunter, J.D. Matplotlib: A 2D Graphics Environment (matplotlib.org).
6. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python
   (scikit-learn.org).
""")

print("=" * 70)
print("SCRIPT COMPLETE")
print("=" * 70)

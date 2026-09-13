# MGNM-801-EV-Report

# EV Two-Wheeler Market Expansion Analytics — India

Business Analytics project (MGNM801) analyzing India's EV market to guide 
a two-wheeler manufacturer's state-level expansion strategy, using descriptive 
statistics, Matplotlib visualizations, and two predictive models.

## Business Problem
An Indian EV two-wheeler manufacturer needs data-driven guidance on which 
states to prioritize for dealership expansion, how demand varies by vehicle 
category, and how to time production around seasonal demand.

## Data Source
- **Dataset:** Electric Vehicle Sales by State in India ([Kaggle](https://www.kaggle.com/datasets/mafzal19/electric-vehicle-sales-by-state-in-india))
- **Underlying source:** Vahan Dashboard, Ministry of Road Transport & Highways, Government of India
- **Coverage:** 2014–2023, 34 states, 96,845 raw records

## Contents
| File | Description |
|---|---|
| `ev_business_analytics_project.py` | Full pipeline: cleaning, feature engineering, descriptive stats, 6 charts, 2 predictive models |
| `EV_Dataset.csv` | Raw data |
| `charts/` | Generated chart images |
| `MGNM801_EV_Business_Analytics_Report.docx` | Full written report |

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn
python ev_business_analytics_project.py
```

## Key Results
- National 2W EV sales grew from ~1,500 units/yr (2014–17) to 856,836 units (2023)
- Random Forest forecasting model: **R² = 0.867** on 2023 test data
- Top markets by volume: Maharashtra, Karnataka | Fastest-growing: Kerala, Gujarat

## Author
Prince Bhatia — 12400933

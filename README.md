# Project Title: Time Series Forecasting & Trend Analysis

Cleans a time-series dataset, analyses trend and seasonality, trains forecasting models, evaluates them on a time-based split, and forecasts the future with a 95% confidence band.

Pipeline:-
Data: any CSV with a date column and a numeric target (or built-in synthetic sales data)
Preprocessing: duplicates, missing values, outliers, lag/rolling/calendar features
Trend analysis: seasonal decomposition (trend, seasonality, residuals)
Models: Naive, Seasonal naive, Linear Regression, Random Forest, SARIMA
Evaluation: chronological split; MAE, RMSE, R²
Visualisation: actual vs predicted, metrics, feature importance, future forecast

Usage:
-> pip install -r requirements.txt
-> python predanalysis.py                                  
-> python predanalysis.py --csv data.csv --date-col Date --target-col Sales --freq D --horizon 60

Tech stack: Python, pandas, NumPy, scikit-learn, statsmodels, Matplotlib

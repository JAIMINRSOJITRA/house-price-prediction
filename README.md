# House Price Prediction

Predicts house prices from listing features using an XGBoost regression model, served through a Streamlit app.

## Features

- Trained on the classic Housing dataset (area, bedrooms, bathrooms, amenities, furnishing status)
- 18 engineered features: raw listing fields plus derived ones (total rooms, luxury flag, amenity score, area buckets, log-area, etc.)
- Interactive Streamlit UI — enter listing details and get an instant price estimate
- Model trained and compared across Linear Regression, Random Forest, and XGBoost (XGBoost pipeline is what's deployed)

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-316192?style=flat)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

## Project Structure

```
housepriceprediction.ipynb   # EDA, feature engineering, model training/comparison
app.py                       # Streamlit app — loads xgboost_pipeline.pkl and serves predictions
Housing.csv                  # Training dataset
xgboost_pipeline.pkl         # Trained StandardScaler + XGBRegressor pipeline
```

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

The feature engineering in `app.py` is built to exactly match the training pipeline in the notebook (same 18 features, same order, same formulas) — verified by inspecting the trained pipeline's expected feature schema directly.

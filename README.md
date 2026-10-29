# Explainable AutoML

An interactive AutoML application built with Streamlit that not only trains and compares machine learning models, but also **explains** each step — from preprocessing recommendations to model results — in plain language using the Google Gemini API.

## Features

- **Data upload:** CSV and Excel files
- **Data analysis:** basic statistics, missing values, duplicates, and a detailed profiling report (ydata-profiling)
- **Preprocessing recommendations:** automatic suggestions for scaling, encoding, missing value handling, and duplicate rows
- **Preprocessing operations** (fitted on the training set only to prevent data leakage):
  - Encoding: Label Encoding (binary) / One-Hot Encoding (multi-class)
  - Scaling: StandardScaler
  - Imputation: median (numerical) / mode (categorical)
- **Model training:**
  - _Manual Selection_ — train a single model of your choice
  - _AutoML Selection_ — automatically detects the task type, trains all suitable models, and selects the best one
- **Explainability:** Gemini explains the preprocessing recommendations and the model results for users without a strong ML background
- **Model explainability (SHAP):** feature importance (bar), effect direction (beeswarm) and single-prediction (waterfall) plots for any trained model, plus a plain-language Gemini interpretation of the SHAP results

## Supported Models

| Classification         | Regression             |
| ---------------------- | ---------------------- |
| Random Forest          | Random Forest          |
| Support Vector Machine | Support Vector Machine |
| Decision Tree          | Decision Tree          |
| Logistic Regression    | Linear Regression      |
| K-Nearest Neighbors    | Gradient Boosting      |

Evaluation metrics: **Accuracy, Precision, Recall, F1 Score** (weighted) for classification; **MAE, RMSE, R²** for regression.

## Installation

```bash
git clone https://github.com/petekis/explainable-automl.git
cd explainable-automl
pip install -r requirements.txt
```

Create a `.env` file in the project folder and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
streamlit run explainable_automl.py
```

1. Upload a CSV or Excel file
2. Review the data analysis and preprocessing recommendations
3. Select the target column and split the data
4. Apply preprocessing operations
5. Train models manually or with AutoML
6. Ask Gemini to explain the results
7. Compute SHAP values and ask Gemini to interpret them

## Tech Stack

Python · Streamlit · pandas · scikit-learn · SHAP · ydata-profiling · Google Gemini API

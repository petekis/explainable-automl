import pandas as pd
from config import HIGH_MISSING_RATIO

def missing_value_advice(series, percentage):
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if percentage > HIGH_MISSING_RATIO:
        if is_numeric:
            return ("More than half of the values are missing. Imputation would mostly create "
                    "artificial values; consider dropping this column.")
        return ("More than half of the values are missing. Instead of imputation, encode this column "
                "so that missing values become their own category, or drop it.")
    if is_numeric:
        return "Median imputation is recommended."
    return "Mode (most frequent value) imputation is recommended."

def build_recommendations(df):
    rec_dict = {}
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numerical_columns) != 0:
        rec_dict["scaling"] = {}
        rec_dict["scaling"]["numeric_col"] = numerical_columns.tolist()
        rec_dict["scaling"]["recommendation"] = "Numerical columns is detected. Scaling is recommended"
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) != 0:
        rec_dict["encoding"] = []
        for column in categorical_columns:
            unique_count = df[column].nunique()
            if unique_count <= 2:
                rec_dict["encoding"].append({
                    "column": column,
                    "unique_values": unique_count,
                    "method": "Ordinal Encoding",
                    "recommendation": "Binary categorical column. Ordinal Encoding is recommended."
                })
            else:
                rec_dict["encoding"].append({
                    "column": column,
                    "unique_values": unique_count,
                    "method": "One-Hot Encoding",
                    "recommendation": "Categorical column with more than two categories. One-Hot Encoding is recommended."
                })
    missing_ratio = df.isnull().sum() * 100 / df.shape[0]
    missing_ratio = missing_ratio[missing_ratio > 0]
    if not missing_ratio.empty:
        rec_dict["missing_val"] = []
        for column, percentage in missing_ratio.items():
            rec_dict["missing_val"].append({
                "column": column,
                "missing_percentage": round(percentage, 1),
                "recommendation": missing_value_advice(df[column], percentage)
            })
    duplicated_mask = df.duplicated(keep=False)
    if duplicated_mask.any():
        groups = int(df[duplicated_mask].drop_duplicates().shape[0])
        extra_copies = int(df.duplicated().sum())
        rec_dict["duplicated"] = {
            "groups": groups,
            "extra_copies": extra_copies,
            "recommendation": (
                f"{groups} distinct rows appear more than once ({extra_copies} extra copies). "
                "If each row should be a unique record (e.g., the data has an ID column), these are likely "
                "real duplicates and should be dropped. If identical rows can belong to different records "
                "(e.g., there is no ID column), review them before dropping."
            )
        }
    return rec_dict

def check_training_readiness(X_train, X_test):
    problems = []
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_columns:
        problems.append(f"Encode these categorical columns: {', '.join(categorical_columns)}")

    missing = X_train.isnull().sum() + X_test.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        details = ", ".join(f"{column} ({count} missing)" for column, count in missing.items())
        problems.append(f"Impute missing values in: {details}")

    return problems
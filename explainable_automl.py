import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit import dataframe, session_state
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from dotenv import load_dotenv
from google import genai
import copy 
import shap

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key) if api_key else None

st.title("Explainable AutoML")
if client is None:
    st.warning("GEMINI API key is not set in the .env file. AI explanations are disabled.")
DEFAULTS = {
    "file_type": None,
    "raw_df": None,
    "processed_df": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "target_column": None,
    "task_type": None,
    "results_df": None,
    "selected_model": None,
    "selected_choice": None,
    "trained_models": {},
    "preprocess_explanation": None,
    "results_explanation": None,
    "shap_values": None,
    "shap_model": None,
    "shap_explanation": None
}

KEEP_ON_RESET = ["file_type", "raw_df", "processed_df"]

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)

def reset_state():
    for key, value in DEFAULTS.items():
        if key not in KEEP_ON_RESET:
            st.session_state[key] = copy.deepcopy(value)
    for widget_key in ["target", "encoding_column", "imputation_columns","drop_column"]:
        st.session_state.pop(widget_key, None)

RESULT_KEYS = ["results_df", "selected_model", "task_type", "selected_choice",
               "trained_models", "results_explanation", "shap_values", "shap_model",
               "shap_explanation"]

def clear_results():
    for key in RESULT_KEYS:
        st.session_state[key] = copy.deepcopy(DEFAULTS[key])

def analize_data(df):
    with st.spinner("Analizing data..."):
        profile = ProfileReport(df, title="Data Analysis")
        components.html(profile.to_html(), height=1000, scrolling=True)

HIGH_MISSING_RATIO = 50
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


def basic_analysis(df):
    with st.expander("Basic Analysis"):
        st.write(
            "Shape of dataframe: ", df.shape,
            "Describe: ", df.describe(),
            "Data types: ", df.dtypes.astype(str),
            "Are there any null variables?", df.isnull().sum(),
            "Missing ratio: ", df.isnull().sum() * 100 / df.shape[0],
            "Numerical Columns: ", df.select_dtypes(include=["int64", "float64"]).columns,
            "Categorical Columns: ", df.select_dtypes(include=["object", "category"]),
            "Number of duplicated rows (extra copies): ", df.duplicated().sum()
        )

def show_recommendations(df):
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

def display_recommendations(rec_dict, df):
    if "scaling" in rec_dict:
        st.write("**Scaling:**", rec_dict["scaling"]["recommendation"])
        st.write("Etkilenen sütunlar:", rec_dict["scaling"]["numeric_col"])

    if "encoding" in rec_dict:
        st.write("**Encoding:**")
        for item in rec_dict["encoding"]:
            st.write(f"- `{item['column']}` ({item['unique_values']} unique) → {item['method']}")

    if "missing_val" in rec_dict:
        st.write("**Missing Values:**")
        for item in rec_dict["missing_val"]:
            st.write(f"- `{item['column']}` ({item['missing_percentage']}% missing): {item['recommendation']}")

    if "duplicated" in rec_dict:
        st.warning(rec_dict["duplicated"]["recommendation"])
        with st.expander("Show duplicated rows"):
            duplicated_rows = (df[df.duplicated(keep=False)]
                               .groupby(list(df.columns), dropna=False)
                               .size()
                               .reset_index(name="count")
                               .sort_values("count", ascending=False))
            st.dataframe(duplicated_rows)

def apply_recommendations():
    X_train = st.session_state.X_train.copy()
    X_test = st.session_state.X_test.copy()
    choice = st.selectbox("Please select the operation: ", ("Scaling", "Encoding", "Imputation", "Drop Column"), key="operation_choice")

    if choice == "Encoding":
        categorical_columns =X_train.select_dtypes(include=["object", "category"]).columns
        if len(categorical_columns) == 0:
            st.info("There are no categorical columns left to encode.")
        else:
            column = st.selectbox("Please select the column for encoding: ", categorical_columns, key="encoding_column")
            if st.button("Run Encoding"):
                if X_train[column].nunique() <= 2:
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X_train[[column]] = oe.fit_transform(X_train[[column]])
                    X_test[[column]] = oe.transform(X_test[[column]])
                else:
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    ohe.fit(X_train[[column]])
                    train_arr = ohe.transform(X_train[[column]])
                    test_arr = ohe.transform(X_test[[column]])
                    cols = ohe.get_feature_names_out([column])
                    train_enc_df = pd.DataFrame(train_arr, columns=cols, index=X_train.index)
                    test_enc_df = pd.DataFrame(test_arr, columns=cols, index=X_test.index)
                    X_train = pd.concat([X_train.drop(columns=[column]), train_enc_df], axis=1)
                    X_test = pd.concat([X_test.drop(columns=[column]), test_enc_df], axis=1)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                clear_results()
                st.success("Encoding applied.")
                st.rerun()
    elif choice == "Scaling":
        numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns
        if len(numerical_columns) == 0:
            st.warning("There is no numerical columns to scale.")
        else:
            if st.button("Run Scaling"):
                scaler = StandardScaler()
                X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
                X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                clear_results()
                st.success("Scaling applied.")
                st.rerun()
    elif choice == "Imputation":
        column = st.selectbox("Please select the column for imputation: ",  X_train.columns,
                              key="imputation_columns")
        st.write(f"Missing values in `{column}`: train = {X_train[column].isnull().sum()}, "
                 f"test = {X_test[column].isnull().sum()}")
        missing_percentage = X_train[column].isnull().mean() * 100
        if missing_percentage > HIGH_MISSING_RATIO:
            st.warning(f"{missing_percentage:.1f}% of `{column}` is missing in the training set. "
                       + missing_value_advice(X_train[column], missing_percentage))
            
        if st.button("Run Imputation"):
            if X_train[column].dtype in ["int64", "float64"]:
                fill_value = X_train[column].median()
            else:
                fill_value = X_train[column].mode()[0]
            X_train[column] = X_train[column].fillna(fill_value)
            X_test[column] = X_test[column].fillna(fill_value)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            clear_results()
            st.success("Imputation applied.")
            st.rerun()
    elif choice == "Drop Column":
        if X_train.shape[1] <= 1:
            st.info("At least one feature column must remain, so no more columns can be dropped.")
        else:
            column = st.selectbox("Please select the column to drop: ", X_train.columns, key="drop_column")
            if st.button("Run Drop Column"):
                X_train = X_train.drop(columns=[column])
                X_test = X_test.drop(columns=[column])
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                clear_results()
                st.success(f"`{column}` dropped.")
                st.rerun()


def get_tasktype(model):
    classification_models = {"Random Forest Classifier", "Support Vector Machine Classifier",
        "Decision Tree Classifier", "Logistic Regressor", "KNN"}

    return "classification" if model in classification_models else "regression"

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

def model_selection():
    problems = check_training_readiness(st.session_state.X_train, st.session_state.X_test)
    if problems:
        st.warning("The data is not ready for model training yet:\n\n" +
                   "\n".join(f"- {problem}" for problem in problems))
        return
    y_train = st.session_state.y_train
    choice = st.radio("Model Selection Preference", ["Manual Selection", "AutoML Selection"])
    if choice == "Manual Selection":
        model = st.selectbox("Please select your model", (
        "Random Forest Regressor", "Random Forest Classifier", "Support Vector Machine Regressor","Support Vector Machine Classifier", "Decision Tree Classifier","Decision Tree Regressor",
        "Logistic Regressor","Gradient Boosting Regressor", "KNN", "Linear Regressor"), key="model_selection")
        if st.button("Apply Model"):
            clear_results()

            task_type = get_tasktype(model)
            result = model_process(model, task_type)
            results_df = pd.DataFrame([result])

            st.session_state.results_df = results_df
            st.session_state.task_type = task_type
            st.session_state.selected_model = result
            st.session_state.selected_choice = choice

    else:
        unique_ratio = y_train.nunique() / len(y_train)
        recommended_task = "classification" if unique_ratio <= 0.30 else "regression"
        model_list = {
            "classification": ["Random Forest Classifier", "Support Vector Machine Classifier", "Decision Tree Classifier",
                                      "Logistic Regressor", "KNN"],
            "regression": ["Random Forest Regressor", "Support Vector Machine Regressor",
                            "Decision Tree Regressor", "Linear Regressor", "Gradient Boosting Regressor"]
        }
        with st.expander("Task Type Recommendation"):
            st.write(f"Recommended task type: {recommended_task}")

            task_type = st.radio(
                "Task Type",
                ["classification", "regression"],
                index=0 if recommended_task == "classification" else 1
            )

            model_list = model_list[task_type]

        if st.button("Run AutoML"):
            clear_results()
            results = []

            for model in model_list:
                result = model_process(
                    model,
                    task_type
                )
                results.append(result)

            results_df = pd.DataFrame(results)

            st.session_state.results_df = results_df
            st.session_state.task_type = task_type

            if task_type == "classification":
                selected_model = results_df.loc[
                    results_df["Accuracy Score"].idxmax()
                ]

            else:
                selected_model = results_df.loc[
                    results_df["RMSE"].idxmin()
                ]
            st.session_state.selected_model = selected_model
            st.session_state.selected_choice = choice

def display_results():
    if st.session_state.results_df is None:
        return
    st.subheader("Results")
    st.dataframe(st.session_state.results_df)
    if st.session_state.selected_choice == "AutoML Selection":
        st.success(f"Selected Model: {st.session_state.selected_model['Model']}")

def model_process(model, task_type):
    if model == "Random Forest Regressor":
        model_object = RandomForestRegressor(random_state=42)

    if model == "Random Forest Classifier":
        model_object = RandomForestClassifier(random_state=42)

    if model == "Support Vector Machine Regressor":
        model_object = SVR()

    if model == "Support Vector Machine Classifier":
        model_object = SVC()

    if model == "Decision Tree Regressor":
        model_object = DecisionTreeRegressor(random_state=42)

    if model == "Decision Tree Classifier":
        model_object = DecisionTreeClassifier(random_state=42)

    if model == "Logistic Regressor":
        model_object = LogisticRegression()

    if model == "Gradient Boosting Regressor":
        model_object = GradientBoostingRegressor(random_state=42)

    if model == "Linear Regressor":
        model_object = LinearRegression()

    if model == "KNN":
        model_object = KNeighborsClassifier()

    X_train = st.session_state.X_train.copy()
    X_test = st.session_state.X_test.copy()
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    feature_names = X_train.columns.tolist()

    scaler = None
    if isinstance(model_object, (LinearRegression, SVR, SVC, KNeighborsClassifier, LogisticRegression)):
        scaler = StandardScaler().set_output(transform="pandas")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model_object.fit(X_train, y_train)
    prediction = model_object.predict(X_test)

    st.session_state.trained_models[model] = {
        "model": model_object,
        "scaler": scaler,
        "feature_names": feature_names
    }
    return show_result(model, task_type, y_test, prediction)

TREE_MODELS = (RandomForestClassifier, RandomForestRegressor, DecisionTreeClassifier,
               DecisionTreeRegressor, GradientBoostingRegressor)
LINEAR_MODELS = (LinearRegression, LogisticRegression)
SHAP_BACKGROUND_SIZE = 100
SHAP_SAMPLE_SIZE = 200
SHAP_SLOW_SAMPLE_SIZE = 50

def compute_shap(model_name):
    if model_name not in st.session_state.trained_models:
        st.error(f"Model '{model_name}' is not trained yet.")
        return None

    model_info = st.session_state.trained_models[model_name]
    model_object = model_info["model"]
    scaler = model_info["scaler"]
    feature_names = model_info["feature_names"]

    X_train = st.session_state.X_train[feature_names]
    X_test = st.session_state.X_test[feature_names]
    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    background = shap.sample(X_train, SHAP_BACKGROUND_SIZE, random_state=42)
    sample = shap.sample(X_test, SHAP_SAMPLE_SIZE, random_state=42)

    if isinstance(model_object, TREE_MODELS):
        explainer = shap.TreeExplainer(model_object)
    elif isinstance(model_object, LINEAR_MODELS):
        explainer = shap.LinearExplainer(model_object, background)
    else:
        background = shap.sample(background, SHAP_SLOW_SAMPLE_SIZE, random_state=42)
        sample = shap.sample(sample, SHAP_SLOW_SAMPLE_SIZE, random_state=42)
        predict = lambda data: model_object.predict(pd.DataFrame(data, columns=feature_names))
        explainer = shap.Explainer(predict, background)

    shap_values = explainer(sample)
    st.session_state.shap_values = shap_values
    st.session_state.shap_model = model_name
    st.session_state.shap_explanation = None
    return shap_values

def show_shap_plot(plot_function, explanation):
    plot_function(explanation, show=False)
    st.pyplot(plt.gcf())
    plt.close()

def shap_summary(shap_values, top_n=10):
    values = shap_values.values
    data = np.asarray(shap_values.data, dtype=float)
    importance = np.abs(values).mean(axis=0)

    effects = []
    for i in range(values.shape[1]):
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(data[:, i], values[:, i])[0, 1]
        if corr > 0.3:
            effects.append("higher values increase the prediction")
        elif corr < -0.3:
            effects.append("higher values decrease the prediction")
        else:
            effects.append("mixed / unclear")

    summary = pd.DataFrame({
        "Feature": shap_values.feature_names,
        "Mean |SHAP|": importance.round(4),
        "Effect": effects
    })
    return summary.sort_values("Mean |SHAP|", ascending=False).head(top_n)

def display_shap():
    if not st.session_state.trained_models:
        return
    st.subheader("Model Explainability (SHAP)")

    model_names = list(st.session_state.trained_models)
    default_model = st.session_state.selected_model["Model"]
    model_name = st.selectbox("Model to explain", model_names, index=model_names.index(default_model))
    model_object = st.session_state.trained_models[model_name]["model"]

    if not isinstance(model_object, TREE_MODELS + LINEAR_MODELS):
        st.caption("This model has no built-in SHAP support, so a slower model-agnostic method is used "
                   f"on {SHAP_SLOW_SAMPLE_SIZE} test rows, and it explains the predicted value directly.")

    if st.button("Compute SHAP values"):
        with st.spinner("Computing SHAP values..."):
            compute_shap(model_name)
            

    if st.session_state.shap_values is None or st.session_state.shap_model != model_name:
        return

    shap_values = st.session_state.shap_values
    class_label = None
    if shap_values.values.ndim == 3:
        classes = list(model_object.classes_)
        class_label = st.selectbox(
            "Class to explain", classes, index=1 if len(classes) == 2 else 0,
            on_change=lambda: st.session_state.update(shap_explanation=None))
        shap_values = shap_values[:, :, classes.index(class_label)]

    st.markdown("**Feature importance**")
    show_shap_plot(shap.plots.bar, shap_values)
    st.caption("Average absolute SHAP value: the longer the bar, the more the feature affects predictions.")

    st.markdown("**Direction of the effect**")
    show_shap_plot(shap.plots.beeswarm, shap_values)
    st.caption("Each dot is one row. Red means a high feature value, blue a low one. "
               "Dots to the right push the prediction up, dots to the left push it down.")

    st.markdown("**Single prediction**")
    row = st.number_input("Row to explain", min_value=0, max_value=len(shap_values) - 1, value=0)
    show_shap_plot(shap.plots.waterfall, shap_values[row])
    st.caption("Starts from the average prediction E[f(x)] and shows how each feature "
               "moved this row's prediction to f(x).")

    summary = shap_summary(shap_values)
    if st.button("Explain SHAP with Gemini", disabled=client is None):
        explanation = gemini_shap_explanation(
            summary, model_name, st.session_state.task_type, class_label)
        if explanation:
            st.session_state.shap_explanation = explanation

    if st.session_state.shap_explanation:
        st.subheader("AI SHAP Explanation")
        st.write(st.session_state.shap_explanation)
        with st.expander("Data sent to Gemini"):
            st.dataframe(summary, hide_index=True)

def show_result(model, task_type, y_test, prediction):
    if task_type == "regression":
        model_name = model
        mae = mean_absolute_error(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, prediction)
        result_dict = {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        }
        return result_dict

    if task_type == "classification":
        model_name = model
        class_report = classification_report(y_test, prediction)
        acc_score = accuracy_score(y_test, prediction)
        conf_matrix = confusion_matrix(y_test, prediction)
        result_dict = {
            "Model": model_name,
            "Accuracy Score": acc_score,
        }
        return result_dict

def ask_gemini(prompt):
    if client is None:
        st.error("Gemini is not configured. Add GEMINI_API_KEY to your .env file.")
        return None

    try:
        response = client.models.generate_content(
            model = "gemini-3.6-flash",
            contents = prompt)
        return response.text
    except Exception as error:
        status = getattr(error, "code", None)
        if status == 429:
            st.error("Gemini quota exceeded. Please wait a while and try again.")
        elif status in (401, 403):
            st.error("Gemini rejected the API key. Please check GEMINI_API_KEY in your .env file.")
        elif status == 503:
            st.error("Gemini is busy right now (high demand). Please try again in a few minutes.")
        else:
            st.error(f"Gemini could not generate an explanation: {error}")
        return None

def gemini_explanation(results_df, task_type, selected_model, selected_choice):
    if selected_choice == "Manual Selection":
        selection_instruction = f"""
            The user manually selected the following model:

            {selected_model["Model"]}

            Explain the performance of this selected model.
            Do not describe it as the automatically best model.
            Explain its metrics and what the results mean.
            """

    else:
        selection_instruction = f"""
            The system evaluated multiple models automatically.

            The best performing model was:

            {selected_model["Model"]}

            Explain why this model was selected compared with
            the other models.
            """

    prompt = f"""
        You are an AI assistant inside an Explainable AutoML application.

        Task type: {task_type}

        Model comparison results:
        {results_df.to_string(index=False)}

        {selection_instruction}

        Explain the results in simple language for a user
        who may not have a strong machine learning background.

        Also explain what the evaluation metrics mean.
        """
    return ask_gemini(prompt)

def gemini_preprocess_explanation(rec_dict):
    prompt = f"""
            You are an AI assistant inside an Explainable AutoML application.

            Here are the preprocessing recommendations generated by the system:

            {rec_dict}

            Explain these recommendations to the user in simple language.

            For each recommendation:
            - Explain why this preprocessing operation may be needed.
            - Explain which columns are affected.
            - Explain the difference between encoding methods when relevant.
            - Do not recommend an operation if it is not present in the provided recommendations.
            """
    return ask_gemini(prompt)

def gemini_shap_explanation(summary_df, model_name, task_type, class_label):
    class_line = f"Explained class: {class_label}" if class_label is not None else ""
    prompt = f"""
            You are an AI assistant inside an Explainable AutoML application.

            Task type: {task_type}
            Model: {model_name}
            {class_line}

            SHAP feature summary (mean absolute SHAP value and effect direction):
            {summary_df.to_string(index=False)}

            Explain these SHAP results to the user in simple language.

            - Start with one sentence explaining what SHAP values are.
            - Explain the 3-5 most important features and the direction of their effect.
            - If a feature's effect is "mixed / unclear", do not claim a clear direction.
            - Make clear that SHAP shows what the model learned, not cause and effect.
            - Only mention features that appear in the summary.
            """
    return ask_gemini(prompt)

st.session_state.file_type = st.selectbox("File type:", ["CSV", "Excel"], index=None, placeholder="Select a file type")

if st.session_state.file_type == "CSV":
    uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if st.session_state.raw_df is None or not data.equals(st.session_state.raw_df):
            st.session_state.raw_df = data
            st.session_state.processed_df = data.copy()
            reset_state()

elif st.session_state.file_type == "Excel":
    uploaded_file = st.file_uploader("Choose an Excel file:", type=["xlsx", "xls"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        if st.session_state.raw_df is None or not data.equals(st.session_state.raw_df):
            st.session_state.raw_df = data
            st.session_state.processed_df = data.copy()
            reset_state()

if st.session_state.raw_df is not None:
    st.subheader("Uploaded Data")
    st.dataframe(st.session_state.raw_df)

    if st.button("Detailed Analysis (Profiling)"):
        analize_data(st.session_state.processed_df)

    basic_analysis(st.session_state.processed_df)
    st.subheader("Preprocessing Recommendations")
    rest_dict = show_recommendations(st.session_state.processed_df)
    display_recommendations(rest_dict, st.session_state.processed_df)

    if "duplicated" in rest_dict:
        if st.session_state.X_train is None:
            if st.button("Drop duplicate rows"):
                st.session_state.processed_df = st.session_state.processed_df.drop_duplicates()
                st.session_state.preprocess_explanation = None
                st.rerun()
        else:
            st.caption("Duplicate rows can only be dropped before splitting the data.")

    if rest_dict and st.button("Explain Recommendations with Gemini", disabled=client is None):
        explanation = gemini_preprocess_explanation(rest_dict)
        if explanation:
            st.session_state.preprocess_explanation = explanation

    if st.session_state.preprocess_explanation:
        st.subheader("AI Preprocessing Explanation")
        st.write(st.session_state.preprocess_explanation)

    st.subheader("Select Target Column")
    if st.session_state.X_train is None:
        target = st.selectbox("Please select the target column", options=st.session_state.processed_df.columns, key="target")
        if st.button("Split"):
            df = st.session_state.processed_df.copy()
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.target_column = target
            st.rerun()
        st.stop()
    st.success(f"Target column: {st.session_state.target_column}")

    st.subheader("Apply Preprocessing")
    apply_recommendations()

    st.subheader("Processed Data (current state)")
    st.dataframe(st.session_state.X_train)

    st.subheader("Model Selection")
    model_selection()
    display_results()

    if st.button("Explain Results with Gemini", disabled=client is None):

        if (st.session_state.results_df is not None
            and st.session_state.selected_model is not None
            and st.session_state.task_type is not None
            and st.session_state.selected_choice is not None):

            explanation = gemini_explanation(
                st.session_state.results_df,
                st.session_state.task_type,
                st.session_state.selected_model,
                st.session_state.selected_choice
            )

            if explanation:
                st.session_state.results_explanation = explanation

        else:
            st.warning("Please run AutoML before asking Gemini for an explanation.")

    if st.session_state.results_explanation:
        st.subheader("AI Explanation")
        st.write(st.session_state.results_explanation)

    display_shap()

else:
    st.info("Please upload a dataset.")








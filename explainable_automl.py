import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit import dataframe, session_state
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
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

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

st.title("Explainable AutoML")
DEFAULTS = {
    "file_type": None,
    "raw_df": None,
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
}

KEEP_ON_RESET = ["file_type", "raw_df"]

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)

def reset_state():
    for key, value in DEFAULTS.items():
        if key not in KEEP_ON_RESET:
            st.session_state[key] = copy.deepcopy(value)
    for widget_key in ["target", "encoding_column", "imputation_columns"]:
        st.session_state.pop(widget_key, None)

def analize_data(df):
    with st.spinner("Analizing data..."):
        profile = ProfileReport(df, title="Data Analysis")
        components.html(profile.to_html(), height=1000, scrolling=True)

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
            "Number of duplicated rows: ", df.duplicated(keep=False).sum()
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
                    "method": "Label Encoding",
                    "recommendation": "Binary categorical column. Label Encoding is recommended."
                })
            else:
                rec_dict["encoding"].append({
                    "column": column,
                    "unique_values": unique_count,
                    "method": "One-Hot Encoding",
                    "recommendation": "Categorical column with more than two categories. One-Hot Encoding is recommended."
                })
    missing = df.isnull().sum()
    missing_ratio = df.isnull().sum() * 100 / df.shape[0]
    if missing.sum() != 0:
        rec_dict["missing_val"] = []
        for column, percentage in missing_ratio.items():
            if 1 <= percentage <= 10:
                rec_dict["missing_val"].append(
                    {
                        "column": column,
                        "recommendation": "Mean/Median Imputation"
                    }
                )
            elif 10 < percentage <= 50:
                rec_dict["missing_val"].append(
                    {
                        "column": column,
                        "recommendation": "Imputation is recommended."
                    }
                )
            elif 50 < percentage:
                rec_dict["missing_val"].append(
                    {
                        "column": column,
                        "recommendation": "Consider to remove this feature."
                    }
                )
    if df.duplicated(keep=False).sum() != 0:
        rec_dict["duplicated"] = {}
        rec_dict["duplicated"]["recommendation"] = "There are duplicated rows needed to drop."
    return rec_dict

def display_recommendations(rec_dict):
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
            st.write(f"- `{item['column']}`: {item['recommendation']}")

    if "duplicated" in rec_dict:
        st.warning(rec_dict["duplicated"]["recommendation"])

def apply_recommendations():
    X_train = st.session_state.X_train.copy()
    X_test = st.session_state.X_test.copy()
    choice = st.selectbox("Please select the operation: ", ("Scaling", "Encoding", "Imputation"), key="operation_choice")

    if choice == "Encoding":
        column = st.selectbox("Please select the column for encoding: ",  X_train.select_dtypes(include=["object", "category"]).columns,
            key="encoding_column")
        if st.button("Run Encoding"):
            if X_train[column].nunique() <= 2:
                le = LabelEncoder()
                X_train[column] = le.fit_transform(X_train[column])
                X_test[column] = le.transform(X_test[column])
            else:
                ohe = OneHotEncoder()
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
                st.success("Scaling applied.")
                st.rerun()
    elif choice == "Imputation":
        column = st.selectbox("Please select the column for imputation: ",  X_train.columns,
                              key="imputation_columns")
        st.write(f"Missing values in `{column}`: train = {X_train[column].isnull().sum()}, "
                 f"test = {X_test[column].isnull().sum()}")
        
        if st.button("Run Imputation"):
            if X_train[column].dtype in ["int64", "float64"]:
                fill_value = X_train[column].median()
            else:
                fill_value = X_train[column].mode()[0]
            X_train[column] = X_train[column].fillna(fill_value)
            X_test[column] = X_test[column].fillna(fill_value)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.success("Imputation applied.")
            st.rerun()

def get_tasktype(model):
    classification_models = {"Random Forest Classifier", "Support Vector Machine Classifier",
        "Decision Tree Classifier", "Logistic Regressor", "KNN"}

    return "classification" if model in classification_models else "regression"

def model_selection():
    y_train = st.session_state.y_train
    choice = st.radio("Model Selection Preference", ["Manual Selection", "AutoML Selection"])
    if choice == "Manual Selection":
        model = st.selectbox("Please select your model", (
        "Random Forest Regressor", "Random Forest Classifier", "Support Vector Machine Regressor","Support Vector Machine Classifier", "Decision Tree Classifier","Decision Tree Regressor",
        "Logistic Regressor","Gradient Boosting Regressor", "KNN", "Linear Regressor"), key="model_selection")
        if st.button("Apply Model"):

            task_type = get_tasktype(model)
            result = model_process(model, task_type)
            results_df = pd.DataFrame([result])

            st.session_state.results_df = results_df
            st.session_state.task_type = task_type
            st.session_state.selected_model = result
            st.session_state.selected_choice = choice

            st.dataframe(results_df)
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
            st.dataframe(results_df)

            st.success(
                f"Selected Model: {selected_model['Model']}"
            )

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

    if not isinstance(model_object, (LinearRegression, SVR, SVC, KNeighborsClassifier, LogisticRegression)):
        model_object.fit(X_train, y_train)
        prediction = model_object.predict(X_test)
        result = show_result(model, task_type, y_test, prediction)
        return result

    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_object.fit(X_train_scaled, y_train)
        prediction = model_object.predict(X_test_scaled)
        result = show_result(model, task_type, y_test, prediction)
        return result

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

    interaction = client.interactions.create(
        model = "gemini-3.6-flash",
        input = prompt)

    return interaction.output_text

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
        analize_data(st.session_state.raw_df)

    basic_analysis(st.session_state.raw_df)
    st.subheader("Preprocessing Recommendations")
    rest_dict = show_recommendations(st.session_state.raw_df)
    display_recommendations(rest_dict)

    if rest_dict and st.button("Explain Recommendations with Gemini"):
        explanation = gemini_preprocess_explanation(rest_dict)
        if explanation:
            st.subheader("AI Preprocessing Explanation")
            st.write(explanation)

    st.subheader("Select Target Column")
    if st.session_state.X_train is None:
        target = st.selectbox("Please select the target column", options=st.session_state.raw_df.columns, key="target")
        if st.button("Split"):
            df = st.session_state.raw_df.copy()
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

    if st.button("Explain Results with Gemini"):

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
                st.subheader("AI Explanation")
                st.write(explanation)

        else:
            st.warning("Please run AutoML before asking Gemini for an explanation.")

else:
    st.info("Please upload a dataset.")








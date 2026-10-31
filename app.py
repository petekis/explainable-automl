import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from state import init_state, reset_state
from config import RANDOM_STATE
from core.llm import gemini_preprocess_explanation, gemini_explanation, client
from core.analysis import build_recommendations
from ui.gemini_view import explain_with_gemini
from ui.data_view import analize_data, basic_analysis, display_recommendations
from ui.preprocessing_view import apply_recommendations
from ui.model_view import model_selection, display_results 
from ui.shap_view import display_shap

st.title("Explainable AutoML")
if client is None:
    st.warning("GEMINI API key is not set in the .env file. AI explanations are disabled.")

init_state()

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
    rest_dict = build_recommendations(st.session_state.processed_df)
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
        explanation = explain_with_gemini(gemini_preprocess_explanation, rest_dict)
        if explanation:
            st.session_state.preprocess_explanation = explanation

    if st.session_state.preprocess_explanation:
        st.subheader("AI Preprocessing Explanation")
        st.write(st.session_state.preprocess_explanation)

    st.subheader("Select Target Column")
    if st.session_state.X_train is None:
        target = st.selectbox("Please select the target column", options=st.session_state.processed_df.columns, key="target")
        missing_target = st.session_state.processed_df[target].isnull().sum()
        if missing_target > 0:
            st.info(f"{missing_target} rows have a missing value in `{target}`. "
                    "Target values should not be imputed, so these rows will be removed when you split the data.")
        if st.button("Split"):
            df = st.session_state.processed_df.dropna(subset=[target])
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
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

            explanation = explain_with_gemini(
                gemini_explanation,
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
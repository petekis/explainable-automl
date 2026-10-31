import copy
import streamlit as st

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

RESULT_KEYS = ["results_df", "selected_model", "task_type", "selected_choice",
               "trained_models", "results_explanation", "shap_values", "shap_model",
               "shap_explanation"]

def init_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)

def reset_state():
    for key, value in DEFAULTS.items():
        if key not in KEEP_ON_RESET:
            st.session_state[key] = copy.deepcopy(value)
    for widget_key in ["target", "encoding_column", "imputation_columns","drop_column"]:
        st.session_state.pop(widget_key, None)

def clear_results():
    for key in RESULT_KEYS:
        st.session_state[key] = copy.deepcopy(DEFAULTS[key])
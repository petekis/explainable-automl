import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import shap
from state import init_state, reset_state, clear_results
from config import HIGH_MISSING_RATIO, SHAP_SLOW_SAMPLE_SIZE, RANDOM_STATE
from core.llm import gemini_preprocess_explanation, gemini_explanation, gemini_shap_explanation, GeminiError, client
from core.analysis import build_recommendations, missing_value_advice, check_training_readiness
from core.models import MODEL_REGISTRY,get_tasktype, models_for_task, train_model, predict, evaluate
from core.explain import compute_shap, shap_summary, has_fast_explainer

st.title("Explainable AutoML")
if client is None:
    st.warning("GEMINI API key is not set in the .env file. AI explanations are disabled.")

init_state()

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
            "Number of duplicated rows (extra copies): ", df.duplicated().sum()
        )

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

def model_selection():
    problems = check_training_readiness(st.session_state.X_train, st.session_state.X_test)
    if problems:
        st.warning("The data is not ready for model training yet:\n\n" +
                   "\n".join(f"- {problem}" for problem in problems))
        return
    y_train = st.session_state.y_train
    choice = st.radio("Model Selection Preference", ["Manual Selection", "AutoML Selection"])
    if choice == "Manual Selection":
        model = st.selectbox("Please select your model", list(MODEL_REGISTRY), key="model_selection")
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
        with st.expander("Task Type Recommendation"):
            st.write(f"Recommended task type: {recommended_task}")

            task_type = st.radio(
                "Task Type",
                ["classification", "regression"],
                index=0 if recommended_task == "classification" else 1
            )

            model_list = models_for_task(task_type)

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
                    results_df["F1 Score"].idxmax()
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
    model_info = train_model(model, st.session_state.X_train, st.session_state.y_train)
    prediction = predict(model_info, st.session_state.X_test)
    st.session_state.trained_models[model] = model_info
    return evaluate(model, task_type, st.session_state.y_test, prediction)

def show_shap_plot(plot_function, explanation):
    plot_function(explanation, show=False)
    st.pyplot(plt.gcf())
    plt.close()

def display_shap():
    if not st.session_state.trained_models:
        return
    st.subheader("Model Explainability (SHAP)")

    model_names = list(st.session_state.trained_models)
    default_model = st.session_state.selected_model["Model"]
    model_name = st.selectbox("Model to explain", model_names, index=model_names.index(default_model))
    model_object = st.session_state.trained_models[model_name]["model"]

    if not has_fast_explainer(model_object):
        st.caption("This model has no built-in SHAP support, so a slower model-agnostic method is used "
                   f"on {SHAP_SLOW_SAMPLE_SIZE} test rows.")

    if st.button("Compute SHAP values"):
        with st.spinner("Computing SHAP values..."):
            st.session_state.shap_values = compute_shap(
                st.session_state.trained_models[model_name],
                st.session_state.X_train,
                st.session_state.X_test
            )
            st.session_state.shap_model = model_name
            st.session_state.shap_explanation = None
            
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
        explanation = explain_with_gemini(gemini_shap_explanation, summary, model_name, st.session_state.task_type, class_label)
        if explanation:
            st.session_state.shap_explanation = explanation

    if st.session_state.shap_explanation:
        st.subheader("AI SHAP Explanation")
        st.write(st.session_state.shap_explanation)
        with st.expander("Data sent to Gemini"):
            st.dataframe(summary, hide_index=True)

def explain_with_gemini(explain_function, *args):
    try:
        return explain_function(*args)
    except GeminiError as error:
        st.error(str(error))
        return None

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








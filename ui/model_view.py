import streamlit as st
import pandas as pd
from core.analysis import check_training_readiness
from state import clear_results
from core.models import MODEL_REGISTRY, get_tasktype, models_for_task, train_model, predict, evaluate


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
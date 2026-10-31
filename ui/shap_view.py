import streamlit as st
import matplotlib.pyplot as plt
import shap
from core.llm import gemini_shap_explanation, client
from config import SHAP_SLOW_SAMPLE_SIZE
from core.explain import compute_shap, shap_summary, has_fast_explainer
from ui.gemini_view import explain_with_gemini


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

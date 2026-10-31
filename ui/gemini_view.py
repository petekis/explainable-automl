import streamlit as st
from core.llm import GeminiError


def explain_with_gemini(explain_function, *args):
    try:
        return explain_function(*args)
    except GeminiError as error:
        st.error(str(error))
        return None
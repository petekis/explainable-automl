import streamlit as st
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components


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
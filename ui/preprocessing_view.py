import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from state import clear_results
from config import HIGH_MISSING_RATIO
from core.analysis import missing_value_advice

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
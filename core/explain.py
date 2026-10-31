from sklearn.base import is_classifier
import numpy as np
import shap
import pandas as pd
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from config import SHAP_BACKGROUND_SIZE, SHAP_SAMPLE_SIZE, SHAP_SLOW_SAMPLE_SIZE, RANDOM_STATE

TREE_MODELS = (RandomForestClassifier, RandomForestRegressor, DecisionTreeClassifier,
               DecisionTreeRegressor, GradientBoostingRegressor)
LINEAR_MODELS = (LinearRegression, LogisticRegression)

def has_fast_explainer(model_object):
    return isinstance(model_object, TREE_MODELS + LINEAR_MODELS)

def compute_shap(model_info, X_train, X_test):
    model_object = model_info["model"]
    scaler = model_info["scaler"]
    feature_names = model_info["feature_names"]

    X_train = X_train[feature_names]
    X_test = X_test[feature_names]
    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    background = shap.sample(X_train, SHAP_BACKGROUND_SIZE, random_state=RANDOM_STATE)
    sample = shap.sample(X_test, SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)

    if isinstance(model_object, TREE_MODELS):
        explainer = shap.TreeExplainer(model_object)
    elif isinstance(model_object, LINEAR_MODELS):
        explainer = shap.LinearExplainer(model_object, background)
    else:
        background = shap.sample(background, SHAP_SLOW_SAMPLE_SIZE, random_state=RANDOM_STATE)
        sample = shap.sample(sample, SHAP_SLOW_SAMPLE_SIZE, random_state=RANDOM_STATE)
        if is_classifier(model_object):
            predict_function = lambda data: model_object.predict_proba(pd.DataFrame(data, columns=feature_names))
        else:
            predict_function = lambda data: model_object.predict(pd.DataFrame(data, columns=feature_names))
        explainer = shap.Explainer(predict_function, background)
    return explainer(sample)

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
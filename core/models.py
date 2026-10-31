from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE


MODEL_REGISTRY = {
    "Random Forest Classifier": {"task": "classification",
                                 "build": lambda: RandomForestClassifier(random_state=RANDOM_STATE)},
    "Support Vector Machine Classifier": {"task": "classification",
                                          "build": lambda: SVC(probability=True, random_state=RANDOM_STATE)},
    "Decision Tree Classifier": {"task": "classification",
                                 "build": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE)},
    "Logistic Regressor": {"task": "classification", "build": lambda: LogisticRegression()},
    "KNN": {"task": "classification", "build": lambda: KNeighborsClassifier()},
    "Random Forest Regressor": {"task": "regression",
                                "build": lambda: RandomForestRegressor(random_state=RANDOM_STATE)},
    "Support Vector Machine Regressor": {"task": "regression", "build": lambda: SVR()},
    "Decision Tree Regressor": {"task": "regression",
                                "build": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE)},
    "Linear Regressor": {"task": "regression", "build": lambda: LinearRegression()},
    "Gradient Boosting Regressor": {"task": "regression",
                                    "build": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE)},
}

SCALED_MODELS = (LinearRegression, SVR, SVC, KNeighborsClassifier, LogisticRegression)

def get_tasktype(model_name):
    return MODEL_REGISTRY[model_name]["task"]

def models_for_task(task_type):
    return [name for name, info in MODEL_REGISTRY.items() if info["task"] == task_type]

def train_model(model_name, X_train, y_train):
    model_object = MODEL_REGISTRY[model_name]["build"]()
    feature_names = X_train.columns.tolist()

    scaler = None
    if isinstance(model_object, SCALED_MODELS):
        scaler = StandardScaler().set_output(transform="pandas")
        X_train = scaler.fit_transform(X_train)

    model_object.fit(X_train, y_train)
    return {"model": model_object, "scaler": scaler, "feature_names": feature_names}

def predict(model_info, X):
    X = X[model_info["feature_names"]]
    if model_info["scaler"] is not None:
        X = model_info["scaler"].transform(X)
    return model_info["model"].predict(X)

def evaluate(model, task_type, y_test, prediction):
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
        result_dict = {
            "Model": model_name,
            "Accuracy Score": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, prediction, average="weighted", zero_division=0)
        }
        return result_dict




# =============================================================
# Invasive Ventilator Days Prediction
# Full Regression ML Pipeline
# =============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from scipy.stats import pearsonr, spearmanr

# =============================================================
# LOAD DATA
# =============================================================

df = pd.read_csv("clinical_data.csv")

TARGET = "invasive_vent_days"

# =============================================================
# CLEAN TARGET VARIABLE
# =============================================================

# convert to numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

# remove missing or invalid targets
df = df[df[TARGET].notna()]
df = df[df[TARGET] >= 0]

df = df.reset_index(drop=True)

print("Final dataset size:", df.shape)

# =============================================================
# DROP IDENTIFIER COLUMNS
# =============================================================

id_cols = [c for c in df.columns if "id" in c.lower()]
df = df.drop(columns=id_cols, errors="ignore")

# =============================================================
# SPLIT FEATURES / LABEL
# =============================================================

y = df[TARGET]
X = df.drop(columns=[TARGET])

# =============================================================
# FEATURE TYPES
# =============================================================

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

# =============================================================
# HANDLE MISSING VALUES
# =============================================================

# numeric → median
X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])

# categorical → most frequent
X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

# =============================================================
# ONE-HOT ENCODING
# =============================================================

X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

# =============================================================
# TRAIN TEST SPLIT
# =============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =============================================================
# SCALE FEATURES
# =============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================
# METRIC FUNCTION
# =============================================================

def evaluate(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    pcc, _ = pearsonr(y_test, preds)
    scc, _ = spearmanr(y_test, preds)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PCC": pcc,
        "SCC": scc
    }

# =============================================================
# MODELS
# =============================================================

models = [
    (LinearRegression(), "Linear Regression"),
    (Ridge(alpha=1.0), "Ridge Regression"),
    (Lasso(alpha=0.001), "Lasso Regression"),
    (ElasticNet(alpha=0.001, l1_ratio=0.5), "ElasticNet"),
    (SVR(kernel="rbf", C=10, epsilon=0.1), "SVR"),
    (KNeighborsRegressor(n_neighbors=7), "KNN"),
    (RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ), "Random Forest"),
    (GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ), "Gradient Boosting")
]

# =============================================================
# RUN ALL MODELS
# =============================================================

results = []

for model, name in models:
    print(f"Training → {name}")
    results.append(evaluate(model, name))

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\n================ REGRESSION RESULTS ================\n")
print(results_df)

results_df.to_csv("invasive_vent_days_regression_results.csv", index=False)

# =============================================================
# DONE
# =============================================================

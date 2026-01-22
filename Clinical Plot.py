import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Train best model again
# ===============================

best_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

print("Computing SHAP values...")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# ---------------------------------
# 1️⃣ SHAP Summary Plot (GLOBAL)
# ---------------------------------

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    show=False
)
plt.title("SHAP Summary Plot – Feature Importance")
plt.tight_layout()
plt.show()

# ---------------------------------
# 2️⃣ SHAP Bar Plot
# ---------------------------------

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.show()

# ---------------------------------
# 3️⃣ SHAP Dependence Plot
# ---------------------------------

top_feature = np.argsort(np.abs(shap_values).mean(0))[-1]
top_feature_name = X.columns[top_feature]

shap.dependence_plot(
    top_feature_name,
    shap_values,
    X_test,
    feature_names=X.columns
)

# =====================================================
# PREDICTED vs TRUE PLOTS
# =====================================================

# ---------------------------------
# 4️⃣ Predicted vs True Scatter
# ---------------------------------

plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.xlabel("True Vent Days")
plt.ylabel("Predicted Vent Days")
plt.title("Predicted vs True Invasive Vent Days")
plt.grid(True)
plt.show()

# ---------------------------------
# 5️⃣ Regression Scatter Plot
# ---------------------------------

plt.figure(figsize=(7,6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"alpha":0.5})
plt.xlabel("True Vent Days")
plt.ylabel("Predicted Vent Days")
plt.title("Regression Relationship")
plt.show()

# ---------------------------------
# 6️⃣ Residual Plot
# ---------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(7,6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Vent Days")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ---------------------------------
# 7️⃣ Error Distribution
# ---------------------------------

plt.figure(figsize=(7,6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Prediction Error (Days)")
plt.title("Residual Error Distribution")
plt.show()

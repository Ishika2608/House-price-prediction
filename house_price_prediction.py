# ============================================================
#  House Price Prediction — End-to-End ML Project
#  Dataset  : California Housing (built into scikit-learn)
#  Author   : Ishika
#  Tech     : Python, Pandas, Scikit-learn, Matplotlib, Seaborn
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create output folders
os.makedirs("plots", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("=" * 50)
print("Step 1: Loading Dataset")
print("=" * 50)

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Shape  : {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# ── 2. EDA ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 2: Exploratory Data Analysis")
print("=" * 50)
print(df.describe().round(2))
print("\nMissing values:\n", df.isnull().sum())

# Plot 1: Price distribution
plt.figure(figsize=(8, 4))
plt.hist(df["MedHouseVal"], bins=50, color="#185FA5", edgecolor="white")
plt.title("Distribution of House Prices")
plt.xlabel("Median House Value ($100K)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/01_price_distribution.png", dpi=150)
plt.show()

# Plot 2: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/02_correlation_heatmap.png", dpi=150)
plt.show()

# Plot 3: Key features vs price
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feat in zip(axes.flatten(), ["MedInc", "AveRooms", "HouseAge", "AveOccup"]):
    ax.scatter(df[feat], df["MedHouseVal"], alpha=0.15, s=5, color="#185FA5")
    ax.set_xlabel(feat)
    ax.set_ylabel("Price ($100K)")
    ax.set_title(f"{feat} vs Price")
plt.tight_layout()
plt.savefig("plots/03_feature_vs_price.png", dpi=150)
plt.show()

# ── 3. PREPROCESSING ─────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 3: Preprocessing")
print("=" * 50)

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── 4. TRAIN MODELS ──────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 4: Training Models")
print("=" * 50)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
print("Linear Regression trained ✓")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest trained ✓")

# ── 5. EVALUATION ────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 5: Evaluation")
print("=" * 50)

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.4f}  (~${mae*100000:,.0f})")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f} ({r2*100:.1f}%)")

evaluate("Linear Regression", y_test, lr_preds)
evaluate("Random Forest",     y_test, rf_preds)

# Plot 4: Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, name, color in zip(
    axes, [lr_preds, rf_preds],
    ["Linear Regression", "Random Forest"],
    ["#185FA5", "#0F6E56"]
):
    ax.scatter(y_test, preds, alpha=0.25, s=8, color=color)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"{name} | R²={r2_score(y_test, preds):.3f}")
plt.tight_layout()
plt.savefig("plots/04_predicted_vs_actual.png", dpi=150)
plt.show()

# Plot 5: Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
feat_imp.plot(kind="barh", figsize=(8, 5), color="#185FA5")
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig("plots/05_feature_importance.png", dpi=150)
plt.show()

# ── 6. SAVE MODEL ────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 6: Saving Model")
print("=" * 50)

import joblib
joblib.dump(rf, "model/random_forest_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model saved → model/random_forest_model.pkl")
print("Scaler saved → model/scaler.pkl")

# ── 7. SAMPLE PREDICTION ─────────────────────────────────────
print("\n" + "=" * 50)
print("Step 7: Sample Prediction")
print("=" * 50)

sample    = X_test.iloc[[0]]
predicted = rf.predict(sample)[0]
actual    = y_test.iloc[0]
print(f"Predicted : ${predicted * 100000:,.0f}")
print(f"Actual    : ${actual    * 100000:,.0f}")
print(f"Difference: ${abs(predicted - actual) * 100000:,.0f}")

print("\n✅ Done! Check /plots for all charts.")

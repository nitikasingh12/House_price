
# Dataset: Ames Housing Dataset

from altair import Step
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

print("Libraries loaded successfully!")


print("\n--- Step 1: Loading Data ---")

test = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")


df = test.merge(sample_submission, on="Id")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns[:10])} ...")
print(f"\nSalePrice stats:")
print(df["SalePrice"].describe())


print("\n--- Step 2: Exploratory Data Analysis ---")

# Check missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\nColumns with missing values: {len(missing)}")
print(missing.head(10))

# Plot 1 - SalePrice distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df["SalePrice"], bins=40, color="steelblue", edgecolor="white")
plt.title("SalePrice Distribution")
plt.xlabel("Price")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df["SalePrice"]), bins=40, color="coral", edgecolor="white")
plt.title("Log(SalePrice) Distribution")
plt.xlabel("log(Price)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("plots/01_price_distribution.png", dpi=120)
plt.close()
print("Saved: plots/01_price_distribution.png")

# Plot 2 - Overall Quality vs Price
plt.figure(figsize=(8, 5))
qual_price = df.groupby("OverallQual")["SalePrice"].median()
plt.bar(qual_price.index, qual_price.values, color="teal", edgecolor="white")
plt.title("Median Sale Price by Overall Quality")
plt.xlabel("Overall Quality (1=Poor, 10=Excellent)")
plt.ylabel("Median Sale Price ($)")
plt.tight_layout()
plt.savefig("plots/02_quality_vs_price.png", dpi=120)
plt.close()
print("Saved: plots/02_quality_vs_price.png")

# Plot 3 - Correlation heatmap (top numeric features)
num_df = df.select_dtypes(exclude="object")
corr = num_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
top_features = corr.head(10)

plt.figure(figsize=(8, 5))
plt.barh(top_features.index[::-1], top_features.values[::-1], color="steelblue")
plt.title("Top 10 Features Correlated with SalePrice")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig("plots/03_top_correlations.png", dpi=120)
plt.close()
print("Saved: plots/03_top_correlations.png")


print("\n--- Step 3: Feature Engineering ---")

def add_features(data):
    data = data.copy()
    data["HouseAge"]   = data["YrSold"] - data["YearBuilt"]
    data["RemodAge"]   = data["YrSold"] - data["YearRemodAdd"]
    data["TotalSF"]    = data["TotalBsmtSF"].fillna(0) + data["1stFlrSF"] + data["2ndFlrSF"]
    data["TotalBath"]  = data["FullBath"] + 0.5 * data["HalfBath"] + data["BsmtFullBath"].fillna(0)
    data["TotalPorch"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["ScreenPorch"]
    data["HasGarage"]  = (data["GarageArea"].fillna(0) > 0).astype(int)
    data["HasPool"]    = (data["PoolArea"] > 0).astype(int)
    data["QualXCond"]  = data["OverallQual"] * data["OverallCond"]
    return data

df = add_features(df)
test = add_features(test)

print(f"New features added: HouseAge, RemodAge, TotalSF, TotalBath, TotalPorch, HasGarage, HasPool, QualXCond")
print(f"Dataset shape after feature engineering: {df.shape}")


print("\n--- Step 4: Preprocessing ---")

# Separate features and target
X = df.drop(columns=["Id", "SalePrice"])
y = np.log1p(df["SalePrice"])   # log transform to normalize target
X_test = test.drop(columns=["Id"])

# Split into numeric and categorical
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print(f"Numeric features:     {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# Fill missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")

X[num_cols]      = num_imputer.fit_transform(X[num_cols])
X[cat_cols]      = cat_imputer.fit_transform(X[cat_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# Encode categoricals
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols]      = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

print("Missing values handled. Categorical features encoded.")

#Step 5: Train Model 
print("\n--- Step 5: Training Model ---")

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

# Cross validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
rmse = np.sqrt(-cv_scores.mean())
print(f"5-Fold CV RMSE (log scale): {rmse:.4f}")
print(f"Individual fold RMSEs: {[round(np.sqrt(-s), 4) for s in cv_scores]}")

# Train on full data
model.fit(X, y)
print("Model trained successfully!")

# ── Step 6: Generate Predictions ──────────────────────────────────────────────
print("\n--- Step 6: Generating Predictions ---")

# Train predictions
train_preds = np.expm1(model.predict(X))
train_output = pd.DataFrame({
    "Id": df["Id"],
    "SalePrice": train_preds.round(2)
})
train_output.to_csv("outputs/train_predictions.csv", index=False)
print(f"Train predictions saved → outputs/train_predictions.csv")
print(f"  Rows: {len(train_output)}")
print(f"  Mean SalePrice: ${train_preds.mean():,.0f}")

# Test predictions
test_preds = np.expm1(model.predict(X_test))
test_output = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_preds.round(2)
})
test_output.to_csv("outputs/test_predictions.csv", index=False)
print(f"\nTest predictions saved  → outputs/test_predictions.csv")
print(f"  Rows: {len(test_output)}")
print(f"  Mean SalePrice: ${test_preds.mean():,.0f}")

#  Step 7: Feature Importance Plot 
print("\n--- Step 7: Feature Importance ---")

importances = pd.Series(model.feature_importances_, index=X.columns)
top15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 6))
plt.barh(top15.index[::-1], top15.values[::-1], color="teal", edgecolor="white")
plt.title("Top 15 Feature Importances (Gradient Boosting)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/04_feature_importance.png", dpi=120)
plt.close()
print("Saved: plots/04_feature_importance.png")

print("\n All done!")
print("Check the outputs/ folder for predictions and plots/ for charts.")

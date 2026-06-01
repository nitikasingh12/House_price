
# Ames Housing Dataset - House Price Prediction
# tried to keep this clean and well structured

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------------
# Step 1 - load the data
# -------------------------------------------------------

print("loading data...")

df_train = pd.read_csv("data/train.csv")   # actual training data with sale prices
df_test  = pd.read_csv("data/test.csv")    # test data - no labels here

print(f"train shape : {df_train.shape}")
print(f"test shape  : {df_test.shape}")

print("\nquick look at sale prices:")
print(df_train["SalePrice"].describe())

# -------------------------------------------------------
# Step 2 - basic EDA
# -------------------------------------------------------

print("\ndoing some EDA...")

# check missing values
missing_vals = df_train.isnull().sum()
missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)
print(f"\nfeatures with missing values: {len(missing_vals)}")
print(missing_vals.head(10))

# plot 1 - price distribution (raw and log)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df_train["SalePrice"], bins=40, color="steelblue", edgecolor="white")
plt.title("SalePrice Distribution")
plt.xlabel("Price")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df_train["SalePrice"]), bins=40, color="coral", edgecolor="white")
plt.title("Log(SalePrice) - more normal looking")
plt.xlabel("log(Price)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("plots/01_price_distribution.png", dpi=120)
plt.close()
print("saved price distribution plot")

# plot 2 - how quality affects price
qual_price = df_train.groupby("OverallQual")["SalePrice"].median()

plt.figure(figsize=(8, 5))
plt.bar(qual_price.index, qual_price.values, color="teal", edgecolor="white")
plt.title("Median SalePrice by Overall Quality")
plt.xlabel("Overall Quality (1=Poor to 10=Excellent)")
plt.ylabel("Median Sale Price ($)")
plt.tight_layout()
plt.savefig("plots/02_quality_vs_price.png", dpi=120)
plt.close()
print("saved quality vs price plot")

# plot 3 - top correlated features
numeric_df = df_train.select_dtypes(exclude="object")
correlations = numeric_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
top10 = correlations.head(10)

plt.figure(figsize=(8, 5))
plt.barh(top10.index[::-1], top10.values[::-1], color="steelblue")
plt.title("Top 10 Features Correlated with SalePrice")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig("plots/03_top_correlations.png", dpi=120)
plt.close()
print("saved correlations plot")

# -------------------------------------------------------
# Step 3 - feature engineering
# I created a few extra features that made intuitive sense
# -------------------------------------------------------

print("\nengineering new features...")

def make_features(data):
    df = data.copy()

    df["HouseAge"]   = df["YrSold"] - df["YearBuilt"]         # age when sold
    df["RemodAge"]   = df["YrSold"] - df["YearRemodAdd"]       # how recently remodeled
    df["TotalSF"]    = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]   # total area
    df["TotalBath"]  = df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"].fillna(0)
    df["TotalPorch"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"]
    df["HasGarage"]  = (df["GarageArea"].fillna(0) > 0).astype(int)
    df["HasPool"]    = (df["PoolArea"] > 0).astype(int)
    df["QualXCond"]  = df["OverallQual"] * df["OverallCond"]   # combined quality score

    return df

df_train = make_features(df_train)
df_test  = make_features(df_test)

print("new features: HouseAge, RemodAge, TotalSF, TotalBath, TotalPorch, HasGarage, HasPool, QualXCond")
print(f"train shape after feature engineering: {df_train.shape}")

# -------------------------------------------------------
# Step 4 - preprocessing
# -------------------------------------------------------

print("\npreprocessing...")

X = df_train.drop(columns=["Id", "SalePrice"])
y = np.log1p(df_train["SalePrice"])    # log transform - helps with skewed target

X_test_final = df_test.drop(columns=["Id"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print(f"numeric features: {len(num_cols)}")
print(f"categorical features: {len(cat_cols)}")

# fill missing values
num_imp = SimpleImputer(strategy="median")
cat_imp = SimpleImputer(strategy="constant", fill_value="Missing")

X[num_cols] = num_imp.fit_transform(X[num_cols])
X[cat_cols] = cat_imp.fit_transform(X[cat_cols])

X_test_final[num_cols] = num_imp.transform(X_test_final[num_cols])
X_test_final[cat_cols] = cat_imp.transform(X_test_final[cat_cols])

# encode categorical columns
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols]            = enc.fit_transform(X[cat_cols])
X_test_final[cat_cols] = enc.transform(X_test_final[cat_cols])

print("imputation and encoding done")

# train validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\ntrain size : {X_train.shape[0]}")
print(f"val size   : {X_val.shape[0]}")

# -------------------------------------------------------
# Step 5 - train and compare models
# tried 3 different ones to see which works best
# -------------------------------------------------------

print("\n--- comparing models ---")

model_dict = {
    "Gradient Boosting" : GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=4, min_samples_leaf=5,
        subsample=0.8, random_state=42
    ),
    "Random Forest" : RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ),
    "Ridge Regression" : Ridge(alpha=10.0),
}

results = {}

for model_name, model in model_dict.items():
    # cross validation
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    cv_rmse = np.sqrt(-cv.mean())

    # fit and check on validation set
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    val_rmse = np.sqrt(mean_squared_error(y_val, preds))
    val_r2   = r2_score(y_val, preds)
    val_mae  = mean_absolute_error(y_val, preds)

    results[model_name] = {
        "CV RMSE"  : round(cv_rmse, 4),
        "Val RMSE" : round(val_rmse, 4),
        "Val R2"   : round(val_r2, 4),
        "Val MAE"  : round(val_mae, 4),
    }

    print(f"\n{model_name}")
    print(f"  CV RMSE  : {cv_rmse:.4f}")
    print(f"  Val RMSE : {val_rmse:.4f}")
    print(f"  Val R2   : {val_r2:.4f}")
    print(f"  Val MAE  : {val_mae:.4f}")

# summary table
print("\n--- model comparison ---")
print(pd.DataFrame(results).T.to_string())

# pick best model by validation RMSE
best_name  = min(results, key=lambda k: results[k]["Val RMSE"])
best_model = model_dict[best_name]
print(f"\nbest model: {best_name}")

# -------------------------------------------------------
# Step 6 - retrain best model on full data and save it
# -------------------------------------------------------

print(f"\nretraining {best_name} on full training data...")
best_model.fit(X, y)

joblib.dump(best_model, "outputs/house_price_model.pkl")
print("model saved to outputs/house_price_model.pkl")

# -------------------------------------------------------
# Step 7 - generate predictions
# -------------------------------------------------------

print("\ngenerating predictions...")

# train set predictions
train_pred_vals = np.expm1(best_model.predict(X))
train_out = pd.DataFrame({
    "Id"        : df_train["Id"],
    "SalePrice" : train_pred_vals.round(2)
})
train_out.to_csv("outputs/train_predictions.csv", index=False)
print(f"train predictions saved  | rows: {len(train_out)} | avg price: ${train_pred_vals.mean():,.0f}")

# test set predictions
test_pred_vals = np.expm1(best_model.predict(X_test_final))
test_out = pd.DataFrame({
    "Id"        : df_test["Id"],
    "SalePrice" : test_pred_vals.round(2)
})
test_out.to_csv("outputs/test_predictions.csv", index=False)
print(f"test predictions saved   | rows: {len(test_out)} | avg price: ${test_pred_vals.mean():,.0f}")

# -------------------------------------------------------
# Step 8 - feature importance plot
# -------------------------------------------------------

print("\nplotting feature importances...")

if hasattr(best_model, "feature_importances_"):
    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    top15    = feat_imp.sort_values(ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    plt.barh(top15.index[::-1], top15.values[::-1], color="teal", edgecolor="white")
    plt.title(f"Top 15 Feature Importances ({best_name})")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/04_feature_importance.png", dpi=120)
    plt.close()
    print("saved feature importance plot")

print("\ndone! check outputs/ for predictions and plots/ for charts.")

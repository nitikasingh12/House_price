# Housing Price Prediction
Predicting house sale prices using the Ames Housing Dataset.

##    Google colab Notebook
link:https://colab.research.google.com/drive/1gQnG7bF17h5Yx9moc--ZIy42eLFsKVZ0?usp=sharing


## What I did
1. **Loaded the data** — test.csv with 79 features per house
2. **Explored the data** — checked missing values, distributions, correlations
3. **Feature Engineering** — created 8 new features like TotalSF, HouseAge, TotalBath etc.
4. **Preprocessing** — filled missing values, encoded categorical columns
5. **Trained a model** — Gradient Boosting Regressor
6. **Generated predictions** — for both train and test sets

---

## Results

| Metric | Value |
|---|---|
| 5-Fold Cross Validation RMSE (log scale) | 0.0116 |
| Mean Predicted Price | $179,184 |

---

## Project Structure

```
housing-price-prediction/
│
├── data/
│   ├── test.csv                  # Raw test data (79 features)
│   ├── sample_submission.csv     # Labels used for training
│   └── data_description.txt      # Feature descriptions
│
├── plots/
│   ├── 01_price_distribution.png
│   ├── 02_quality_vs_price.png
│   ├── 03_top_correlations.png
│   └── 04_feature_importance.png
│
├── outputs/
│   ├── train_predictions.csv     # Predictions on training data
│   └── test_predictions.csv      # Predictions on test data
│
├── housing_price_prediction.py   # Main script (everything is here)
├── requirements.txt
└── README.md
```

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python housing_price_prediction.py
---

## Features I engineered

| Feature | What it means |
|---|---|
| HouseAge | How old the house is at time of sale |
| RemodAge | Years since last remodel |
| TotalSF | Total area (basement + 1st + 2nd floor) |
| TotalBath | Total bathrooms (full + half) |
| TotalPorch | Total porch area across all types |
| HasGarage | 1 if house has a garage, else 0 |
| HasPool | 1 if house has a pool, else 0 |
| QualXCond | Overall Quality × Overall Condition |

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
```

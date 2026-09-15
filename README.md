# Spotter Freight Rate Prediction — Machine Learning Assessment

Machine learning solution for predicting freight load rates using historical shipment data.

The project includes:

- exploratory data analysis and data-quality checks
- chronological model validation
- preprocessing and feature engineering
- model comparison and Ridge regularization tuning
- rolling time-based backtesting
- final predictions for 12,000 validation loads
- separate reduced-feature model for the fixed December scenario
- reproducible training pipeline
- automated tests for preprocessing, features, and submission formatting

---

## Model Summary

The primary model is **Ridge Regression with alpha = 100**.

A chronological validation strategy was used because the final prediction data occurs after the labeled development period.

Models evaluated included:

- Naive median baseline
- Ridge Regression
- HistGradientBoostingRegressor
- Ridge with additional freight interaction and lane features

The original Ridge model generalized best.

### October Holdout

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge, alpha=100 | $140.46 | $651.30 | 0.8185 |
| Enhanced Ridge | $153.90 | $669.69 | 0.8081 |
| HistGradientBoosting | $162.62 | $663.18 | 0.8118 |
| Naive Median | $1,146.79 | $1,567.97 | -0.0522 |

---

## Rolling Chronological Validation

To reduce dependence on a single validation month, Ridge regularization was evaluated using July, August, September, and October 2025 as sequential holdouts.

For `alpha=100`:

- Mean MAE: **$150.49**
- Median MAE: **$145.33**
- Mean RMSE: **$632.92**

This regularization strength achieved the lowest mean MAE across the tested values.

---

## Data Quality

The labeled development dataset contains **48,000 rows**.

The final validation dataset contains **12,000 rows**.

Important data-quality findings included:

- missing `weight` values
- missing `market_index` values
- negative freight weights that appeared to be sign errors
- unseen pickup/delivery locations in future data
- right-skewed freight rates with a small number of extreme observations

Processing decisions:

- negative weights are converted to absolute values
- missing-value indicators are retained
- medians are learned from training data only
- categorical values use `OneHotEncoder(handle_unknown="ignore")`
- `load_id` is excluded from predictive features
- target-derived EDA features are excluded from training

---

## Main Model Features

The final validation model uses:

### Numeric

- pickup latitude / longitude
- delivery latitude / longitude
- distance
- weight
- market index
- quote signal
- month
- day of week
- day of month
- ISO week
- continuous days since January 1, 2025
- missing-value indicators

### Categorical

- pickup
- delivery
- equipment

---

## December Scenario

The supplied December scenario contains:

- Lexington → Fort Wayne
- 360 miles
- Dry Van
- 32,000 lb
- December 1–31, 2025

The December input does not contain `market_index`, `quote_signal`, or geographic coordinates, so a separate reduced-feature Ridge model is used rather than inventing unavailable values.

The exact Lexington → Fort Wayne lane appears in the historical development data.

A lane-aware reduced Ridge model was selected using chronological backtesting.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── december-chart-inputs.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── outputs/
├── reports/
│   └── figures/
├── scorer_results/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_prediction_format.py
├── score.py
├── requirements.txt
├── validation_predictions.csv
└── README.md
```

---

## Setup

Python 3.12 was used during development.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Input Files

Place the assessment datasets inside:

```text
data/raw/
```

Expected files:

```text
train-test.csv
validation.csv
validation-predictions-template.csv
december-chart-inputs.csv
```

The assessment datasets are not committed to the repository.

---

## Run the Full Pipeline

From the repository root:

```bash
python -m src.train
```

The pipeline:

1. loads the raw datasets
2. cleans and prepares features
3. learns imputation statistics from labeled training data
4. trains the final Ridge model
5. generates the 12,000 validation predictions
6. trains the reduced December model
7. generates the 31 December predictions

Reproduced files are written to:

```text
outputs/validation_predictions_reproduced.csv
outputs/december_chart_inputs_reproduced.csv
```

---

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Current test suite:

```text
13 tests
13 passed
```

The tests cover:

- preprocessing behavior
- missing-value handling
- feature definitions
- leakage prevention
- lane construction
- prediction schemas
- duplicate-date detection
- non-positive prediction rejection

---

## Run the Provided Spotter Scorer

For the final submission files:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

For reproduced pipeline outputs:

```bash
python score.py --predictions outputs/validation_predictions_reproduced.csv --december-predictions outputs/december_chart_inputs_reproduced.csv --output-dir outputs/scorer_results_reproduced
```

Successful validation produces:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
```

and generates:

```text
candidate_december.png
```

The final hidden validation metrics are calculated by Spotter after submission.

---

## Final Submission Output

`validation_predictions.csv` contains exactly:

```text
load_id,predicted_rate
```

for all **12,000 validation loads**.

The December prediction file contains one prediction for every day from December 1 through December 31, 2025.

---

## Author

**Muhammad Talha Tahir**

Machine Learning Engineer Assessment
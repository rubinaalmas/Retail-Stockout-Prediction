
# Retail Stockout Prediction Using Machine Learning

### Predicting Next-Day Inventory Stockouts with XGBoost

A machine learning project focused on forecasting next-day retail stockout events using historical sales, product availability, and temporal patterns.

The project applies feature engineering, gradient-boosted decision trees, and time-based model evaluation to identify products at risk of becoming unavailable, helping explore data-driven approaches to inventory management.

## 1. Project Overview

Retail stockouts can result in lost sales, dissatisfied customers, and inefficient replenishment decisions. Identifying stockout risks in advance can help retailers prioritize inventory monitoring and operational interventions.

**Objective:** Develop a binary classification model that predicts whether a store-product combination will experience at least one recorded stockout hour on the following day.

The project explores:

- Historical sales and inventory availability patterns
- Lag-based and rolling-window feature engineering
- XGBoost classification for stockout prediction
- Chronological training, validation, and evaluation
- Precision-recall tradeoffs and model interpretability

## 2. Dataset

**Source:** FreshRetailNet-50K by Dingdong-Inc

[Access the dataset on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)

FreshRetailNet-50K contains real-world retail sales and product availability observations collected across multiple stores and products.

### Dataset Characteristics

| Property | Value |
|---|---|
| Training subset analyzed | 450,000 observations |
| Store-product combinations | 5,000 |
| Unique stores | 57 |
| Unique product IDs | 352 |
| Historical observations | 90 days per store-product combination |
| Missing values | None in selected subset |
| Initial observed stockout rate | 44.28% |
| Official evaluation subset | 35,000 next-day predictions |

The experiment uses the first 10% of the dataset's official training split, representing 5,000 store-product combinations.

The official evaluation is restricted to those same combinations to measure future-period predictive performance.

### Relevant Variables

- `sale_amount`: Daily sales
- `stock_hour6_22_cnt`: Recorded stockout hours
- `store_id`, `product_id`: Store and product identifiers
- Product category identifiers
- `discount`, `activity_flag`: Pricing and promotional information
- `dt`: Observation date

The target is derived from actual recorded stockout hours rather than a synthetic inventory-risk rule.

## 3. Methodology

### A. Target Definition

A stockout event is defined as a day with at least one recorded stockout hour.

The model predicts the next day's stockout event using observations available through the end of the current day.

**Target:**

- 1: At least one stockout hour tomorrow
- 0: No recorded stockout hours tomorrow

This represents a rolling next-day forecasting task.

### B. Feature Engineering

Historical features were created independently for each store-product combination.

**Sales features**
- Current-day sales
- Sales lags of 1, 2, 3, and 7 days
- Rolling average sales over 3 and 7 days

**Stockout history**
- Current-day recorded stockout hours
- Previous stockout indicators at 1, 2, 3, and 7-day lags
- Rolling stockout rates over 3 and 7 days

**Product and calendar features**
- Store and product identifiers
- Product category identifiers
- Promotional indicators
- Tomorrow's day of week and month

These features allow the model to examine historical patterns and differences across products and stores.

### C. Model Development

**Algorithm:** XGBoost Classifier

XGBoost was selected for its ability to model nonlinear relationships and interactions in structured tabular data.

The experiment included:

1. Chronological training, validation, and internal test splits
2. Historical feature engineering
3. XGBoost model training
4. Classification threshold selection using validation F1-score
5. Comparison with a persistence baseline
6. Evaluation on later, out-of-time observations

The classification threshold selected on validation data was **0.30**.

### D. Leakage-Aware Evaluation

The original project approach was revised to avoid predicting a same-day target directly defined by the input variables.

The updated experiment:

- Predicts the following day's observed stockout outcome
- Uses historical data available at prediction time
- Separates training and validation chronologically
- Selects the operating threshold on validation data
- Evaluates on later dates without retraining on those future observations

This design evaluates forecasts for store-product combinations already observed during training.

## 4. Model Performance

### Out-of-Time Evaluation Results

Evaluation period: June 26–July 2, 2024

Evaluation records: **35,000**

| Metric | XGBoost | Persistence Baseline |
|---|---:|---:|
| Precision | 48.74% | 56.85% |
| Recall | **79.72%** | 55.74% |
| F1-score | **60.50%** | 56.29% |
| PR-AUC | 0.5517 | Not evaluated |
| ROC-AUC | 0.6430 | Not evaluated |

### Key Result

**XGBoost improved next-day stockout recall by 23.98 percentage points over the persistence baseline.**

The model correctly identified 11,752 of 14,741 observed stockout events during the evaluation period.

The improved recall came with lower precision, reflecting a tradeoff between detecting stockouts and generating false alarms.

### Confusion Matrix

| | Predicted No Stockout | Predicted Stockout |
|---|---:|---:|
| Actual No Stockout | 7,901 | 12,358 |
| Actual Stockout | 2,989 | 11,752 |

**Interpretation:** The model prioritizes detecting potential inventory shortages. Although this improves recall, a substantial number of predictions are false positives.

The operating threshold should therefore be chosen according to the costs of missed stockouts, unnecessary replenishment actions, and additional inventory monitoring.

## 5. Feature Importance

The most influential features identified by XGBoost were:

| Feature | Importance |
|---|---:|
| Current-day stockout hours | 35.67% |
| 3-day rolling stockout rate | 21.38% |
| Stockout indicator, 2-day lag | 5.61% |
| Product identifier | 4.85% |
| 7-day rolling stockout rate | 4.77% |

Recent stockout history contributed strongly to model predictions.

Feature importance represents the model's relative reliance on predictors. It does not establish causation.

## 6. Technologies Used

**Language:** Python

**Machine Learning:** XGBoost, Scikit-learn

**Data Processing:** Pandas, NumPy

**Visualization:** Matplotlib

**Development Environment:** Google Colab, Jupyter Notebook

**Techniques:** Feature Engineering, Binary Classification, Time-Series Features, Model Evaluation, Threshold Optimization

## 7. Limitations

Several limitations should be considered when interpreting the results.

**Precision-recall tradeoff:** The model achieves relatively high recall but generates many false-positive alerts.

**Limited training subset:** The experiment uses 5,000 of the dataset's 50,000 store-product series. Performance may differ across the full dataset.

**Existing-product evaluation:** The final evaluation covers previously observed store-product combinations, rather than measuring generalization to entirely unseen products or stores.

**Prediction timing:** Forecasts assume that the current day's sales and stockout information are available before predicting the following day.

**Operational impact:** Improvements in classification metrics do not directly establish reductions in lost sales, replenishment costs, or actual business stockouts.

## 8. Future Improvements

Potential improvements include:

- Expanding training to the complete FreshRetailNet-50K dataset
- Hyperparameter tuning using chronological cross-validation
- Comparing XGBoost with LightGBM and other models
- Incorporating additional lagged promotional and weather features
- Evaluating calibration and cost-sensitive thresholds
- Testing generalization across previously unseen stores and products
- Developing a dashboard for stockout-risk monitoring

## 9. How to Reproduce

1. Open the project notebook in Google Colab or Jupyter.
2. Install the dependencies used in the notebook.
3. Load FreshRetailNet-50K from Hugging Face.
4. Select the documented 10% training subset.
5. Generate historical features and next-day targets.
6. Train XGBoost using chronological data splits.
7. Select the classification threshold on validation data.
8. Evaluate on the documented future-period subset.

The dataset is retrieved from its original public source rather than redistributed within this repository.

## 10. Dataset Attribution

Dataset: **FreshRetailNet-50K**

Publisher: Dingdong-Inc

Source: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

Dataset license: CC BY 4.0

The dataset belongs to its respective creators and is used here for educational and research purposes.

---

**Project Focus:** Applied Machine Learning | Predictive Modeling | Retail Analytics | XGBoost | Time-Based Evaluation

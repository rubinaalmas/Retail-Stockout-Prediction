# Retail Stockout Risk Prediction with Machine Learning

## Project Overview

Retailers lose millions every year because they run out of high-demand products.
This project builds a machine learning model that predicts stockout risk before it happens , helping stores avoid lost sales and improve customer satisfaction.

I worked with a real-world retail dataset including inventory levels, demand forecasts, pricing, promotions, weather conditions, and more.

## Business Problem

Traditional inventory rules fail during:

* Seasonal spikes
* Rapid demand changes
* Competitive price shifts
* Weather-driven demand

This model proactively identifies which products in which stores are at risk of stockout so teams can replenish faster.

## Techniques Used

* Data Processing:	Feature Engineering, Rolling Windows, Time-based Signals
* Modeling:	Random Forest , XGBoost
* Imbalance Handling:	SMOTE Oversampling + Class Weights
* Evaluation:	Confusion Matrix, Classification Report, Feature Importance
* Deployment-ready Design:	Scikit-learn Pipelines

## Results That Matter

* Improved recall for stockout cases from 0.2% to ~90%
* Overall model accuracy: ~99%
* Fully interpretable drivers of stockout risk 

### This solution enables stores to reduce lost revenue, better plan replenishment, and prioritize high-risk products.

## Key Insights from Feature Importance

Top contributors to stockout risk:

1. Demand Forecast
2. Real-time Sales (Units Sold)
3. Inventory Levels
4. Product Category (Clothing, Electronics)
5. Region + Seasonal Factors (Spring, Sunny weather : more demand)

### These reveal where retailers should urgently shift stock.

##  Project Structure

Retail-Stockout-Prediction/.

├── Retail_Stockout_Prediction.ipynb   # Main notebook.

├── data_sample.csv                    # Sample dataset for reference.

├── feature_importance_plot.png        # Top drivers visual.

├── README.md                          # Project documentation

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, imblearn, Seaborn, Matplotlib

### Author

Rubina Almas
Data Scientist & Machine Learning Researcher
📍 Arizona, USA


## Future Enhancements

> Model deployment as a real-time alert dashboard (Streamlit / FastAPI)

> SKU-level demand forecasting integration

> Cost-based stockout risk prioritization

⭐ If you like this project, please star the repo!

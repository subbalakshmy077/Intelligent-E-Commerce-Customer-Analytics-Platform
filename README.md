# Intelligent E-Commerce Customer Analytics Platform

## Overview
The **Intelligent E-Commerce Customer Analytics Platform** is an end-to-end Machine Learning and Business Intelligence application designed to transform complex transactional data into actionable operational insights. Built on the **Brazilian E-Commerce Public Dataset by Olist**, the system processes multiple interconnected relational tables to profile customer behavior, segment buyers, predict repeat purchase likelihood, forecast Customer Lifetime Value (CLV), and provide explainable model decisions.

The project features an interactive **Streamlit Dashboard** that empowers marketing teams, business analysts, and decision-makers to optimize retention strategies and revenue performance.

---

## Key Features & Use Cases
* **Customer Segmentation:** Groups customers into distinct behavioral cohorts (e.g., Premium, Regular, Bargain Hunters, Dormant) using **K-Means Clustering**.
* **Purchase Prediction:** Predicts 90-day repeat purchase probabilities using ensemble classifiers (**XGBoost, LightGBM, CatBoost**).
* **Customer Lifetime Value (CLV):** Forecasts expected future revenue per customer using advanced regression pipelines.
* **Explainable AI (XAI):** Uses **SHAP (SHapley Additive exPlanations)** summary, force, and waterfall plots to interpret the primary drivers behind predictions.
* **Interactive Dashboard:** Deploys a multi-page Streamlit application with live KPI tracking, sales analytics, and dynamic customer profiling.

---

## Tech Stack
* **Language:** Python
* **Data Engineering & Analysis:** Pandas, NumPy, SQL
* **Machine Learning & Pipelines:** Scikit-Learn, XGBoost, LightGBM, CatBoost, Imbalanced-Learn (SMOTE)
* **Model Explainability:** SHAP
* **Visualization & Web App:** Streamlit, Matplotlib, Seaborn, Plotly

---

├── models/                  # Saved model pipelines and artifacts (.pkl)
├── app.py                   # Main Streamlit Dashboard entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

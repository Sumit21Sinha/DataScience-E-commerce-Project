# 🛒 E-Commerce Delivery Intelligence using Machine Learning

## 📌 Project Overview

This project analyzes the Brazilian E-Commerce Public Dataset by Olist and develops end-to-end Machine Learning solutions for two real-world logistics problems:

1. **Regression:** Predict the number of days required to deliver an order.
2. **Classification:** Predict whether an order will be delivered late.

The project demonstrates a complete Machine Learning workflow including data cleaning, feature engineering, handling imbalanced datasets, model development, hyperparameter tuning, threshold optimization, and model evaluation.

---

# 📂 Dataset

**Source:** Olist Brazilian E-Commerce Dataset

The project uses multiple relational datasets including:

* Customers
* Orders
* Order Items
* Products
* Sellers
* Payments
* Reviews
* Geolocation
* Product Category Translation

These datasets were merged to create a single order-level master dataset.

---

# 🎯 Objectives

## Regression

Predict

> **Delivery Days**

(Number of days between purchase and customer delivery)

---

## Classification

Predict

> **Late Delivery**

Binary Classification

* 0 → Delivered On Time
* 1 → Delivered Late

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

Algorithms

* Linear Regression
* Decision Tree
* Random Forest
* XGBoost
* Logistic Regression

---

# 📊 Data Preprocessing

The following preprocessing pipeline was implemented.

## Data Cleaning

* Merged all Olist datasets into a master dataset
* Converted timestamp columns into datetime format
* Removed invalid delivered orders having missing delivery timestamps
* Filled missing categorical values
* Removed infinite values
* Removed remaining missing values

---

## Feature Engineering

### Time Features

* Purchase Year
* Purchase Month
* Purchase Day
* Purchase Hour
* Purchase Quarter
* Weekend Indicator

---

### Delivery Features

* Delivery Days
* Approval Days
* Estimated Delivery Days
* Delivery Delay

---

### Product Features

* Total Weight
* Average Weight
* Maximum Weight
* Total Volume
* Average Volume
* Product Density
* Weight per Item
* Volume per Item

---

### Pricing Features

* Total Price
* Total Freight
* Freight Ratio
* Price per Item
* Freight per Item

---

### Order Features

* Number of Items
* Number of Sellers
* Payment Installments

---

### Geographical Features

* Customer Latitude
* Customer Longitude
* Seller Latitude
* Seller Longitude

Distance between Seller and Customer was calculated using the **Haversine Distance Formula**.

---

### Business Features

* Same State Indicator
* Weekend Purchase Indicator
* Expensive Order Indicator

---

# 📈 Regression Pipeline

Target Variable

```
delivery_days
```

### Models Implemented

* Linear Regression
* Decision Tree Regressor
* Tuned Decision Tree Regressor
* Random Forest Regressor
* Tuned Random Forest Regressor
* XGBoost Regressor

---

## Hyperparameter Tuning

GridSearchCV was used for

* Decision Tree
* Random Forest

Cross Validation

* 4-Fold
* 5-Fold

---

## Evaluation Metrics

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## Final Regression Model

🏆 **Random Forest Regressor**

Best Test R²

```
≈ 0.45
```

The Random Forest model outperformed all other regression models by effectively capturing nonlinear relationships among engineered logistics and geographical features.

---

# 📊 Classification Pipeline

Target Variable

```
late_delivery
```

Late Delivery was defined as

```
delivery_delay > 0
```

---

## Class Distribution

Original Dataset

* On-Time Deliveries ≈ 93%
* Late Deliveries ≈ 7%

The dataset was highly imbalanced.

---

## Handling Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique) was applied **only on the training dataset** to balance both classes before model training.

---

## Models Implemented

* Logistic Regression
* Decision Tree Classifier
* Tuned Decision Tree Classifier
* Random Forest Classifier
* Tuned Random Forest Classifier
* XGBoost Classifier

---

## Hyperparameter Tuning

GridSearchCV

* Decision Tree
* Random Forest

Cross Validation

* Stratified K-Fold Cross Validation

---

## Threshold Optimization

Instead of using the default prediction threshold of **0.5**, multiple thresholds were evaluated.

Thresholds tested

```
0.20
0.30
0.40
0.50
0.60
```

Threshold optimization significantly improved the model's ability to identify late deliveries.

The optimal threshold was found to be

```
0.30
```

---

# 📋 Model Comparison

| Model                                  |   Accuracy | Precision |   Recall | F1 Score |
| -------------------------------------- | ---------: | --------: | -------: | -------: |
| Logistic Regression                    |     88.76% |      0.18 |     0.18 |     0.18 |
| Decision Tree                          |     87.17% |      0.18 |     0.25 |     0.21 |
| Tuned Decision Tree                    |     87.25% |      0.23 |     0.38 |     0.29 |
| Random Forest                          |     92.65% |      0.40 |     0.17 |     0.23 |
| Random Forest (Threshold = 0.30)       |     87.23% |      0.24 |     0.42 |     0.31 |
| XGBoost                                |     92.71% |      0.40 |     0.15 |     0.22 |
| **XGBoost (Threshold = 0.30)**         | **88.96%** |  **0.28** | **0.39** | **0.32** |
| Tuned Random Forest (Threshold = 0.30) |     76.63% |      0.17 |     0.62 |     0.26 |

---

# 🏆 Final Classification Model

**XGBoost Classifier**

Reason for Selection

* Highest F1 Score
* Balanced Precision and Recall
* Better overall performance than Logistic Regression, Decision Tree, and Random Forest after threshold optimization

---

# 📌 Key Learnings

During this project, the following Machine Learning concepts were implemented and analyzed:

* Data Cleaning
* Feature Engineering
* Dataset Merging
* Haversine Distance Calculation
* One-Hot Encoding
* Feature Scaling
* Regression Modeling
* Classification Modeling
* Hyperparameter Tuning
* Cross Validation
* Stratified K-Fold
* SMOTE
* Threshold Optimization
* Confusion Matrix Analysis
* Precision, Recall and F1 Score
* Model Comparison
* Business-Oriented Model Selection

---

# 🚀 Future Improvements

* Experiment with LightGBM and CatBoost
* Deploy the model using FastAPI
* Build an interactive Streamlit dashboard
* Perform automated hyperparameter optimization using Optuna
* Incorporate customer review sentiment as an additional predictive feature

---

# 📬 Author

**Sumit Sinha**

Electronics and Computer Engineering

Thapar Institute of Engineering & Technology

GitHub: https://github.com/Sumit21Sinha

Linkedin: https://www.linkedin.com/in/sumit-sinha-454a232a6/

---

## ⭐ If you found this project useful, consider giving it a star!


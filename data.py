#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Importing all the CSVs
customers = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_customers_dataset.csv")
geolocation = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_geolocation_dataset.csv")
orderitems = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_items_dataset.csv")
payments = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_payments_dataset.csv")
reviews = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_reviews_dataset.csv")
orders = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_orders_dataset.csv")
products = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_products_dataset.csv")
sellers = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_sellers_dataset.csv")
product_cat = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\product_category_name_translation.csv")

files = [customers, geolocation, orderitems, payments, reviews, orders, products, sellers]
#Statistics and data cleaning
'''for file in files:
    print(file.isnull().sum())
    print(file.duplicated().sum())
    print(file.info())
    print(file.describe())'''

#Master table merging
master = orders.merge(customers, on="customer_id", how="left")
#print(master.shape)
master = master.merge(orderitems, on="order_id", how="left")
#print(master.shape)
master = master.merge(products, on="product_id", how="left")
#print(master.shape)
master = master.merge(product_cat, on="product_category_name", how="left")
#print(master.shape)
master = master.merge(sellers, on="seller_id", how="left")
#print(master.shape)
master = master.merge(payments, on="order_id", how="left")
#print(master.shape)
#print(master["order_id"].nunique())
#print(master["order_id"].duplicated().sum())

#Master table interpretation & Data cleaning
'''print(master.duplicated().sum())'''
master["order_purchase_timestamp"] = pd.to_datetime(master["order_purchase_timestamp"])
master["order_approved_at"] = pd.to_datetime(master["order_approved_at"])
master["order_delivered_carrier_date"] = pd.to_datetime(master["order_delivered_carrier_date"])
master["order_delivered_customer_date"] = pd.to_datetime(master["order_delivered_customer_date"])
master["order_estimated_delivery_date"] = pd.to_datetime(master["order_estimated_delivery_date"])
'''print(master.info())
print(master.isnull().sum().sort_values(ascending=False))
print(master[master["order_delivered_customer_date"].isnull()]["order_status"].value_counts())
print(master[(master["order_status"] == "delivered") &(master["order_delivered_customer_date"].isnull())
][["order_id", "order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"]])'''
master = master.drop(master[(master["order_status"] == "delivered") & (master["order_delivered_customer_date"].isnull())].index)
master["shipping_limit_date"] = pd.to_datetime(master["shipping_limit_date"])

#Feature Engineering
master["purchase_year"] = master["order_purchase_timestamp"].dt.year
master["purchase_month"] = master["order_purchase_timestamp"].dt.month
master["purchase_day"] = master["order_purchase_timestamp"].dt.day
master["purchase_hour"] = master["order_purchase_timestamp"].dt.hour
master["purchase_dayname"] = master["order_purchase_timestamp"].dt.day_name()
master["purchase_quarter"] = master["order_purchase_timestamp"].dt.quarter

master["is_weekend"] = master["purchase_dayname"].isin(["Saturday","Sunday"])
master["delivery_days"] = (master["order_delivered_customer_date"]-master["order_purchase_timestamp"]).dt.days
master["approval_days"] = (master["order_approved_at"]-master["order_purchase_timestamp"]).dt.days
master["estimated_delivery_days"] = (master["order_estimated_delivery_date"]-master["order_purchase_timestamp"]).dt.days
master["delivery_delay"] = (master["order_delivered_customer_date"]-master["order_estimated_delivery_date"]).dt.days

#New attributes statistics
'''print(master[["delivery_days", "approval_days", "estimated_delivery_days", "delivery_delay"]].describe())'''
#---- Outlier ----
features = ["delivery_days", "approval_days", "estimated_delivery_days", "delivery_delay"]
'''for feature in features:
    sns.boxplot(data=master, x=feature)
    plt.title(feature)
    plt.show()'''

'''numerical_cols = master.select_dtypes(include=np.number)
corr = numerical_cols.corr()
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()'''

#Regression Model
reg_data=master.copy()
# ---Feature Selection---
reg_data = reg_data[reg_data["order_status"] == "delivered"].copy()
numerical_features = ["price","freight_value","payment_value","payment_installments","product_weight_g",
    "product_length_cm","product_height_cm","product_width_cm","approval_days","purchase_month",
    "purchase_day","purchase_hour","purchase_quarter"]
categorical_features = ["is_weekend","customer_state","seller_state","payment_type","product_category_name_english"]
reg_data["product_category_name_english"]=reg_data["product_category_name_english"].fillna("Unknown")
reg_data=reg_data.dropna()
x = reg_data[numerical_features + categorical_features]
y = reg_data["delivery_days"]
'''print("Shape of X:", x.shape)
print("Shape of y:", y.shape)
print("\nMissing Values:")
print(x.isnull().sum())'''
# ---Encoding---
x = pd.get_dummies(x, columns=categorical_features, drop_first=True, dtype=int)
'''print(x.head(5))
print(x.shape)'''
# ---Training-test data splitting---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
'''print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)'''
# ---Scaling---
scaler = StandardScaler()
x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
x_test[numerical_features] = scaler.transform(x_test[numerical_features])
# ---Training---
lr = LinearRegression()
lr.fit(x_train, y_train)
'''print("Linear Regression gives an accuracy of :", lr.score(x_test, y_test)) #21%'''
dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)
'''print("Decision Tree gives an accuracy of :", dt.score(x_test, y_test)) #overfitting(-6%)
print("Linear Regression on train :", lr.score(x_train, y_train)) #21%
print("Decision Tree on train :", dt.score(x_train, y_train)) #99%(overfitting)
print(reg_data["delivery_days"].describe())'''
# ---DecisionTree tuning(Depth)---
param_grid = {"max_depth": [3, 5, 7, 10, 15, 20],"min_samples_split": [2, 5, 10, 20],"min_samples_leaf": [1, 2, 5, 10]}
'''dt2 = DecisionTreeRegressor(random_state=42)
grid = GridSearchCV(estimator=dt2, param_grid=param_grid, scoring="r2", cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_) #{'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 2}'''
dt3 = DecisionTreeRegressor(max_depth=10, min_samples_split=2, min_samples_leaf=10, random_state=42)
dt3.fit(x_train, y_train)
print(dt3.score(x_test, y_test)) #28

# ---Working on Outliers---
Q1 = reg_data["delivery_days"].quantile(0.25)
Q3 = reg_data["delivery_days"].quantile(0.75)
IQR = Q3 - Q1
delivery_upper = Q3 + 3 * IQR

Q1 = reg_data["approval_days"].quantile(0.25)
Q3 = reg_data["approval_days"].quantile(0.75)
IQR = Q3 - Q1
approval_upper = Q3 + 3 * IQR

reg_data = reg_data[(reg_data["delivery_days"] <= delivery_upper) & (reg_data["approval_days"] <= approval_upper)].copy()

#Recreating ML pipeline again
x = reg_data[numerical_features + categorical_features]
y = reg_data["delivery_days"]
x = pd.get_dummies(x, columns=categorical_features, drop_first=True, dtype=int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
x_test[numerical_features] = scaler.transform(x_test[numerical_features])
# ---Linear Regression---
lr = LinearRegression()
lr.fit(x_train, y_train)
print("Re-created Linear Regression :", lr.score(x_test, y_test)) #26%
dt = DecisionTreeRegressor(max_depth=10,min_samples_leaf=10,min_samples_split=2,random_state=42)
dt.fit(x_train, y_train)
print("Re-created Decision Tree :", dt.score(x_test, y_test)) #36%

#Re-feature Engineering

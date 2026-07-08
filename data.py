import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Importing Datasets
customers = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_customers_dataset.csv")
orders = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_orders_dataset.csv")
order_items = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_items_dataset.csv")
payments = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_payments_dataset.csv")
products = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_products_dataset.csv")
sellers = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_sellers_dataset.csv")
reviews = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_reviews_dataset.csv")
geolocation = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_geolocation_dataset.csv")
product_translation = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\product_category_name_translation.csv")

#Merging Datasets
geo = geolocation.groupby("geolocation_zip_code_prefix").agg(latitude=("geolocation_lat", "mean"), longitude=("geolocation_lng", "mean")).reset_index()
seller_geo = sellers.merge(geo, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")
products = products.merge(product_translation, on="product_category_name", how="left")

products["product_volume"] = (products["product_length_cm"]*products["product_width_cm"]*products["product_height_cm"])
products["product_density"] = (products["product_weight_g"]/products["product_volume"])

order_items = order_items.merge(products[["product_id", "product_weight_g", "product_length_cm", "product_width_cm", "product_height_cm",
            "product_volume", "product_density", "product_category_name_english"]], on="product_id", how="left")
order_items = order_items.merge(seller_geo[["seller_id", "latitude", "longitude"]], on="seller_id", how="left")
order_items.rename(columns={"latitude":"seller_lat", "longitude":"seller_lng"}, inplace=True)
order_items = order_items.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

order_items_agg = (order_items.groupby("order_id")
    .agg(max_length=("product_length_cm","max"), max_width=("product_width_cm","max"), max_height=("product_height_cm","max"), seller_lat=("seller_lat","mean"),
        seller_lng=("seller_lng","mean"), total_price=("price", "sum"), total_freight=("freight_value", "sum"),
        num_items=("product_id", "count"), num_sellers=("seller_id", "nunique"), total_weight=("product_weight_g", "sum"),
        avg_weight=("product_weight_g", "mean"), max_weight=("product_weight_g", "max"), total_volume=("product_volume", "sum"),
        avg_volume=("product_volume", "mean"), avg_length=("product_length_cm", "mean"), avg_width=("product_width_cm", "mean"), avg_height=("product_height_cm", "mean"),
        seller_state=("seller_state", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        product_category_name_english=("product_category_name_english", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")).reset_index())

payments_agg = (payments.groupby("order_id")
    .agg(payment_value=("payment_value", "sum"),payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda x: x.mode().iloc[0])).reset_index())

master = (orders.merge(customers, on="customer_id", how="left")
    .merge(order_items_agg, on="order_id", how="left").merge(payments_agg, on="order_id", how="left"))
master = master.merge(geo, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")

master.rename(columns={"latitude":"customer_lat", "longitude":"customer_lng"}, inplace=True)
master.drop(columns="geolocation_zip_code_prefix", inplace=True)

from math import radians
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
master["distance_km"] = haversine(master["customer_lat"], master["customer_lng"], master["seller_lat"], master["seller_lng"])

print("Shape :", master.shape)
print("Unique Orders :", master["order_id"].nunique())
print("Duplicate Orders :", master.duplicated("order_id").sum())
print(master.head())

#Data Cleaning
datecol = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
for i in datecol:
    master[i] = pd.to_datetime(master[i])
master = master.drop(master[(master["order_status"] == "delivered") & (master["order_delivered_customer_date"].isna())].index)
master = master[master["order_status"] == "delivered"].copy()
master["product_category_name_english"] = master["product_category_name_english"].fillna("Unknown")
master["seller_state"] = master["seller_state"].fillna("Unknown")
master["payment_type"] = master["payment_type"].fillna("Unknown")

#Feature Engineering
master["purchase_year"] = master["order_purchase_timestamp"].dt.year
master["purchase_month"] = master["order_purchase_timestamp"].dt.month
master["purchase_day"] = master["order_purchase_timestamp"].dt.day
master["purchase_hour"] = master["order_purchase_timestamp"].dt.hour
master["purchase_weekday"] = master["order_purchase_timestamp"].dt.day_name()
master["purchase_quarter"] = master["order_purchase_timestamp"].dt.quarter

master["is_weekend"] = (master["purchase_weekday"].isin(["Saturday", "Sunday"]).astype(int))
master["delivery_days"] = (master["order_delivered_customer_date"]-master["order_purchase_timestamp"]).dt.days
master["approval_days"] = (master["order_approved_at"]-master["order_purchase_timestamp"]).dt.days
master["estimated_delivery_days"] = (master["order_estimated_delivery_date"]-master["order_purchase_timestamp"]).dt.days
master["delivery_delay"] = (master["order_delivered_customer_date"]-master["order_estimated_delivery_date"]).dt.days

master["same_state"] = (master["customer_state"]==master["seller_state"]).astype(int)
master["freight_ratio"] = master["total_freight"]/master["total_price"]
master["is_expensive"] = (master["total_price"]>master["total_price"].median()).astype(int)
master["weight_per_item"] = master["total_weight"]/master["num_items"]
master["volume_per_item"] = master["total_volume"]/master["num_items"]
master["price_per_item"] = master["total_price"]/master["num_items"]
master["freight_per_item"] = (master["total_freight"]/master["num_items"])

master.replace([np.inf, -np.inf], np.nan, inplace=True)
master.dropna(inplace=True)

clf_data = master.copy() #Copying Dataset for later work on classification

#Removing Outliers
'''def remove_outlier(df, column, m=3):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3-q1
    lower = q1 - m*iqr
    upper = q3 + m*iqr
    return df[(df[column]>=lower) & (df[column]<=upper)]
master = remove_outlier(master, "delivery_days")
master = remove_outlier(master, "approval_days")
master = remove_outlier(master, "estimated_delivery_days")

print(master.shape)
print(master.isnull().sum().sum())
print(master["delivery_days"].describe())
print(master.head(3))

#Feature Selection
numerical_features = ["distance_km", "total_price", "total_freight",
    "payment_value", "payment_installments", "total_weight", "avg_weight", "max_weight", "total_volume",
    "avg_volume", "avg_length", "avg_width", "avg_height", "max_length", "max_width", "max_height", "approval_days",
    "estimated_delivery_days", "purchase_month", "purchase_day", "purchase_hour","purchase_quarter", "num_items",
    "num_sellers", "freight_ratio", "weight_per_item", "volume_per_item", "price_per_item", "freight_per_item"]
categorical_features = ["customer_state", "seller_state", "payment_type", "product_category_name_english", "is_weekend", "same_state", "is_expensive"]

#Regression Preparation
X = master[numerical_features + categorical_features]
y = master["delivery_days"]

#Encoding
X = pd.get_dummies(X, columns=categorical_features, drop_first=True, dtype=int)
print(X.shape)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Scaling
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression stat :", lr.score(X_test, y_test))

#DecisionTree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
print("Decision Tree stat :", dt.score(X_test, y_test))

#Tuning Decision Tree
param_grid = {"max_depth":[5,8,10,12,15], "min_samples_split":[2,5,10], "min_samples_leaf":[1,2,5]}
gridcv = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),param_grid=param_grid, scoring="r2", cv=5, n_jobs=-1)
gridcv.fit(X_train, y_train)
best_dt = gridcv.best_estimator_
best_dt_pred = best_dt.predict(X_test)

print("Best Parameters")
print(gridcv.best_params_)
print("CV Score")
print(gridcv.best_score_)
print("Train R²")
print(best_dt.score(X_train,y_train))
print("Test R²")
print(best_dt.score(X_test,y_test))

#Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
print("Random Forest stat :". rf.score(X_test, y_test))

#Tuning Random Forest
param_grid = {"n_estimators":[100,200], "max_depth":[10,15,20], "min_samples_split":[2,5], "min_samples_leaf":[1,2,5]}
grid_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, scoring="r2", cv=4, n_jobs=-1, verbose=2)
grid_rf.fit(X_train, y_train)

print(grid_rf.best_params_)
best_rf = grid_rf.best_estimator_
best_rf_pred = best_rf.predict(X_test)
print("Tuned random forest stat :", best_rf.score(X_train,y_train))

#XGBoost
xgb = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror")
xgb.fit(X_train, y_train)
print("XGBoost stat :",xgb.score(X_test, y_test))'''

#=======================================================================================================================
#Classification Model
#Column creation for classification
clf_data["late_delivery"] = (clf_data["delivery_delay"] > 0).astype(int)
'''print(clf_data["late_delivery"].value_counts())
print()
print(clf_data["late_delivery"].value_counts(normalize=True))'''

dropcol = ["delivery_days", "delivery_delay", "order_delivered_customer_date", "order_estimated_delivery_date", "order_delivered_carrier_date"]
clf_data.drop(columns=dropcol, inplace=True)

#Feature Selection
numerical_features = ["distance_km","total_price","total_freight","payment_value","payment_installments","total_weight",
    "avg_weight","max_weight","total_volume","avg_volume","avg_length","avg_width","avg_height",
    "max_length","max_width","max_height","approval_days","estimated_delivery_days","purchase_month","purchase_day","purchase_hour",
    "purchase_quarter","num_items","num_sellers","freight_ratio","weight_per_item","volume_per_item","price_per_item","freight_per_item"]
categorical_features = ["customer_state","seller_state","product_category_name_english","is_weekend","same_state","is_expensive"]

X = clf_data[numerical_features + categorical_features]
y = clf_data["late_delivery"]

#Encoding
X = pd.get_dummies(X, columns=categorical_features, drop_first=True, dtype=int)
'''print(X.shape)'''

#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

#Scaling
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
'''print(y_train.value_counts())'''

#Logistic Regression
'''lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''
'''[[16802  1093]
    [ 1065   233]]'''

#Decision Tree
'''dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Train Accuracy :", dt.score(X_train, y_train))
print("Test Accuracy :", dt.score(X_test, y_test))'''
'''[[16400  1495]
    [  968   330]]'''

#Tuning Decision Tree
'''param_grid = {"max_depth": [5, 8, 10, 15], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
grid = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, scoring="r2", cv=cv, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
best_dt = grid.best_estimator_
y_pred = best_dt.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''
'''[[16260  1635]
    [  810   488]]'''

#Random Forest
'''rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''
'''[[17568   327]
     [ 1083   215]]'''

#XGBoost
'''xgb = XGBClassifier(random_state=42,n_estimators=200,max_depth=6,learning_rate=0.1,objective="binary:logistic",eval_metric="logloss")
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''
'''[[17593   302]
    [ 1097   201]]'''

#Tuning Random Forest at Threshold 0.3
param_grid = {"n_estimators": [100, 200], "max_depth": [10, 15, 20], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2, 5]}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42,n_jobs=-1), param_grid=param_grid,scoring="f1",cv=cv,n_jobs=-1,verbose=2)
grid_rf.fit(X_train, y_train)
print(grid_rf.best_params)
print(grid_rf.best_score_)
best_rf = grid_rf.best_estimator_
rf_prob = best_rf.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (rf_prob >= threshold).astype(int)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
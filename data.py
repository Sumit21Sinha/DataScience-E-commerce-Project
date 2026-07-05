#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing all the CSVs
customers = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_customers_dataset.csv")
geolocation = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_customers_dataset.csv")
orderitems = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_items_dataset.csv")
payments = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_payments_dataset.csv")
reviews = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_order_reviews_dataset.csv")
orders = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_orders_dataset.csv")
products = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_products_dataset.csv")
sellers = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\olist_sellers_dataset.csv")
product_cat = pd.read_csv(r"C:\Users\Sumit Sinha\Desktop\DataScience E-Commerce Project\DataSets\product_category_name_translation.csv")

#Statistics and data cleaning
print("Customers :", customers.isnull().sum(), customers.describe())
print("geolocation :", geolocation.isnull().sum(), geolocation.describe())
print("orderitems :", orderitems.isnull().sum(), orderitems.describe())
print("payments :", payments.isnull().sum(), payments.describe())
print("reviews :", reviews.isnull().sum(), reviews.describe())
print("orders :", orders.isnull().sum(), orders.describe())
print("products :", products.isnull().sum(), products.describe())
print("sellers :", sellers.isnull().sum(), sellers.describe())
print("product_cat :", product_cat.isnull().sum(), product_cat.describe())
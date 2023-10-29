############################################################################################################################
# PART 1 Data Preparation
###########################################################################################################################
# Import necessary modules 
import pandas as pd 
import numpy as np 
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score,confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')

# call all dataset 
customer = pd.read_csv(r'D:/Rakamin Academy/Job Accelaration Program (JAP)/VIX Data Scientist at Kalbe/Final Project/Dataset/Case Study - Customer.csv',delimiter = ';')
product = pd.read_csv(r'D:/Rakamin Academy/Job Accelaration Program (JAP)/VIX Data Scientist at Kalbe/Final Project/Dataset/Case Study - Product.csv',delimiter = ';')
store = pd.read_csv(r'D:/Rakamin Academy/Job Accelaration Program (JAP)/VIX Data Scientist at Kalbe/Final Project/Dataset/Case Study - Store.csv',delimiter = ';')
transaction = pd.read_csv(r'D:/Rakamin Academy/Job Accelaration Program (JAP)/VIX Data Scientist at Kalbe/Final Project/Dataset/Case Study - Transaction.csv',delimiter = ';')

# Perform data merging into one dataframe
cust_transac = pd.merge(transaction, customer, how = 'inner', on  = 'CustomerID')
cust_transac_product = pd.merge(cust_transac, product, how = 'inner', on ='ProductID')
df = pd.merge(cust_transac_product, store, how = 'inner', on = 'StoreID')

# Fill the missing values with modus
df['Marital Status'] = df['Marital Status'].fillna(df['Marital Status'].mode()[0])

# Transform date feature into datetime object 
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format = '%d/%m/%Y' )

############################################################################################################################
# PART 2 Regression / Forecasting Using ARIMA Models
###########################################################################################################################

# order data position by the date
df = df.sort_values('Date', ascending = True)

# Extract Qty Values 
qty = df.groupby('Date')['Qty'].sum().reset_index()
qty = qty.drop('Date', axis=1)

# Create train & test : 80% train data & 20% test data
split_index = int(0.8 * len(qty))
qty_train = qty.iloc[:split_index] 
qty_test = qty.iloc[split_index:]

# Train ARIMA Model
model = ARIMA(qty_train, order=(3,1,0))
model_fit = model.fit()
predicted_values = model_fit.forecast(len(qty_test))

# Forecast for the next 365 days
forecast_horizon = 365
forecast = model_fit.forecast(steps=forecast_horizon)

# Transform predicted data into excels 
forecast.to_csv('Next Year Forecasting.csv')

############################################################################################################################
# PART 3 Clusterization Using KMeans
###########################################################################################################################
# Perform Aggregation Based on CustomerID toward Quantity & Amount
cluster = df.groupby('CustomerID')['Qty', 'TotalAmount'].sum().reset_index()
cluster_scaled = cluster.drop('CustomerID', axis=1)

# Perform normalization on Qty & Total Amount feature
scaler = MinMaxScaler()
cluster_scaled = pd.DataFrame(scaler.fit_transform(cluster_scaled), columns = cluster_scaled.columns)

# Fitting clustering model using KMeans Clustering 
kmeans = KMeans(n_clusters=3, random_state = 42).fit(cluster_scaled)
preds = kmeans.predict(cluster_scaled)

cluster_scaled['Cluster'] = kmeans.labels_

# Evaluate clusterization results using Silhoutte Score
silhouette_score(cluster_scaled, preds, metric='euclidean')

# Put the cluster into the dataframe
df['Cluster'] = cluster_scaled['Cluster']
df['Cluster'] = df['Cluster'].fillna(0.0)
df.head()

# Transform clusterized data into csv
df.to_csv('Clusterized Data.csv')
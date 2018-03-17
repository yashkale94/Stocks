import os 
import pandas as pd
import csv
import numpy as np
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import urllib.request
import requests
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import json
import sqlite3
from sklearn.externals import joblib
import pickle
import datetime
import re
from sklearn.decomposition import PCA



predicted_stocks_list = []

def predict_price(stockname):

	df = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stockname+'\"' , connection, index_col = 'Date')
	TrainData = df.iloc[39:-10]
	PredictData = df.iloc[-1:]

	X_train = []
	Y_train = []

	exclude_index_list = [0,1,6,9,10,50,51,52,53,54,55,56,57,58,59,60,65,71,72]

	for i,row in TrainData.iterrows():
		record = row.values
		
		X = []
		for j in range(len(record)):
			if j not in exclude_index_list:
				X.append(float(record[j]))

		if float(record[72]) <= 5:
			Y_train.append('R20')
			
		elif float(record[72]) > 5:
			Y_train.append('R25')

		X_train.append(X)

	for i,row in PredictData.iterrows():
		record = row.values

		X = []
		predict_list = []
		for j in range(len(record)):
			if j not in exclude_index_list:
				X.append(record[j])

		predict_list.append(X)


	LR = MLPClassifier(solver = 'lbfgs', max_iter = 500, hidden_layer_sizes= (1024,512,256,128) )
	# LR = SVC(kernel = 'rbf', C=1e3 , gamma = 0.01)
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	predict_list = scaler.transform(predict_list)
	pca = PCA()
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	predict_list = pca.transform(predict_list)
	LR.fit(X_train, Y_train)
	Y_pred = LR.predict(predict_list)
	if Y_pred == 'R25':
		Y_pred = (predict_prob(predict_list))
		prob = max(Y_pred)
		if prob > 0.7:
			predicted_stocks_list.append((stock,prob))
	
	# Y_pred = (LR.predict_proba(predict_list))
	# prob = max(Y_pred[0])
	# predicted_stocks_list.append((stockname,prob))
	# print(max(Y_pred))
	# input()

def initialise():

	global connection
	global cursor
	connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
	cursor = connection.cursor()
	cursor.execute('SELECT distinct StockSymbol from Stocks')
	currentstocks = cursor.fetchall()
	count = 0
	for stock in currentstocks:
		count = count + 1
		stockname = re.findall(r"[A-Z]+",str(stock))[0]
		print(stockname)
		predict_price(stockname)
		if count ==4:
			break

	dump_to_json(predicted_stocks_list)

def dump_to_json(predicted_stocks_list):

	shortlisted_stock = []
	sorted_list = sorted(predicted_stocks_list, key = lambda x:x[1], reverse = True)
	for entry in range(0,15):
		shortlisted_stock.append(predicted_stocks_list[entry][0])


	with open('TrueStocks.json', 'w') as outfile:
		json.dump(shortlisted_stock, outfile)

initialise()
# today = str(datetime.datetime.now().timestamp())
# today = today.split(".")[0]
# print(today)
# input()
	# return(Y_pred[0])
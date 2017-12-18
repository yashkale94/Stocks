import os
import pandas as pd
import csv
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time

start = time.time()

####This function reads the stock csv file into a dataframe. The data is then converted into a trasaction list format to pass it to the apriori function####
def read_input():
	df = pd.read_csv("Y:\Work\StocksSelector\StocksDisp1\AAN-NYSE.csv", index_col = 'Date')

	testdaata = df.ix['06-01-2016':'20-09-2017']
	print(df['High'][0])
	temp = []
	for i,row in testdaata.iterrows():
		X = []
		record = row.values
		if float(record[5]) > 0:
			X.append('Pricechange+')
		elif float(record[5]) < 0:
			X.append('Pricechange-')
		elif float(record[5]) == 0:
			X.append('No Price Change')

		if float(record[6]) > 0:
			X.append('Volumechange+')
		elif float(record[6]) < 0:
			X.append('Volumechange-')
		elif float(record[6]) == 0:
			X.append('No Volume change')

		for j in range(51,75):
			if float(record[j]) > 0:
				X.append('Above' + str(j))
			else:
				X.append('Below' + str(j))
		
		if float(record[575]) <= 30:
			X.append('RSIbelow30')
		elif float(record[575]) > 30 and float(record[575]) <=70:
			X.append('RSImidrange')
		elif float(record[575]) > 70:
			X.append('RSIabove70')

		if float(record[576]) >= -100 and float(record[576]) <= -40:
			X.append('LowSMI')
		elif float(record[576]) >= -40 and float(record[576]) <= 40:
			X.append('MidSMI')
		elif float(record[576]) >= 40 and float(record[576]) <= 100:
			X.append('HighSMI')

		if float(record[9]) < 0:
			X.append('R10')
		elif float(record[9]) > 0 and float(record[9]) <= 3:
			X.append('R11')
		elif float(record[9]) > 3 and float(record[9]) <= 7:
			X.append('R12')
		elif float(record[9]) > 7 and float(record[9]) <= 15:
			X.append('R13')
		elif float(record[9]) > 15:
			X.append('R14')
		
		temp.append(X)

	return(temp)

####This function returns the frequent itemsets with their support and stores it in a csv file for every stock. The transactions list(inputdata) is converted to 
####one hot transactions and then passed as a parameter to the apriori function#### 
def get_itemsets(inputdata):
	oht = OnehotTransactions()
	oht_ary = oht.fit(inputdata).transform(inputdata)
	df = pd.DataFrame(oht_ary, columns=oht.columns_)
	frequent_itemsets = apriori(df, min_support = 0.1, max_len = 4, use_colnames = True)
	#rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
	frequent_itemsets.to_csv('AANFrequentItemsets.csv')


inputdata = read_input()
get_itemsets(inputdata)

print((time.time() - start)/60)

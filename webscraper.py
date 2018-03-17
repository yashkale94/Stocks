import os
import json
import pandas as pd
import sqlite3
import datetime
import urllib.request
import requests



def scrape_data():
	mainstocks = []

	FOUND = 1

	while FOUND:

		intermediate_session = requests.session()
		http_request = intermediate_session.get("https://in.finance.yahoo.com/quote/FL/history?p=FL")
		main_text = http_request.text
		main_text = str(main_text)
		search_string = "\"CrumbStore\":{\"crumb\":\""
		index = main_text.find(search_string)+len(search_string)
		new_str = main_text[index:]
		index2 = index + new_str.find("\"")
		crumb = main_text[index:index2]
		if "\\u002" not in crumb:
			FOUND = 0

	today = str(datetime.datetime.now().timestamp())
	today = today.split(".")[0]

	yesterday = str(datetime.datetime.now() - datetime.timedelta(days = 1))
	yesterday = yesterday.split(".")[0]
	connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
	cursor = connection.cursor()
	with open('TrueStocks.json') as json_file:
		data = json.load(json_file)
	
	for stock in data:
		print(stock)
		df = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stock+'\"', connection)
		close = list(df['Close'])[-1]
		http_query = "https://query1.finance.yahoo.com/v7/finance/download/"+stock+"?period1="+ str(1519635286) + "&period2="+str(1519635286)+"&interval=1d&events=history&crumb="+crumb
		symbol_text = intermediate_session.get(http_query)
		symbol_text = symbol_text.text.split("\n")
		todaysclose = float(symbol_text[1].split(",")[4])
		if ((float(todaysclose) - float(close))/float(close))*100 < -3:
			status = 'STOP LOSS TRIGGERED'
		elif ((float(todaysclose) - float(close))/float(close))*100 > 5:
			status = 'TARGET ACHIEVED'
		else:
			status = 'HOLD'

		gain = ((float(todaysclose) - float(close))/float(close))*100
		mainstocks.append((stock,close,round(todaysclose,2),status,round(gain,2)))

	with open('YYY.json', 'w') as outfile:
		json.dump(mainstocks, outfile)
scrape_data()
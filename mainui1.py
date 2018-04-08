import os
from flask import Flask, request, redirect, url_for, render_template, jsonify, session, abort
from werkzeug.utils import secure_filename
import json
import pandas as pd
import sqlite3
import datetime
import urllib.request
import requests
import plotly.plotly as py
import plotly.graph_objs as go
from  plotly.offline import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
import numpy as np
import urllib
import datetime as dt
import re
import time
import math

app = Flask(__name__)

@app.route('/')
def render():

	connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
	cursor = connection.cursor()
	newstocks = set()
	oldstocks = set()

	with open('YYY.json') as json_file:
		stockswithlatestclose = json.load(json_file)

	with open('TrueStocks.json') as json_file:
		newdata = json.load(json_file)

	with open('StocksQuantity.json') as json_file:
		olddata = json.load(json_file)

	buydates = []
	for stock in newdata:
		buydates.append(stock[0])
	# print(newdata)

	for stock in newdata:
		newstocks.add(stock[1])
	for stock in olddata:
		oldstocks.add(stock[1])

	stocks_to_be_removed = oldstocks - newstocks
	# print(stocks_to_be_removed)
	# input()
	# print(olddata[0])
	# print(len(stocks_to_be_removed))
	# print(oldstocks)
	# input()
	keep_list = []
	if len(stocks_to_be_removed) == 0:
		for stock in olddata:
			for i in stockswithlatestclose:
				if i[0] == stock[1]:
					X = []
					X.append(stock[0])
					X.append(i[0])
					X.append(stock[2])
					X.append(stock[2] * i[1])
					X.append(i[1])
					keep_list.append(X)
					# stock[3] = i[1]
					# stock[2] = stock[1] * stock[3]

					break
		# print(keep_list)
		# input()
		olddata = []
		olddata = keep_list
		# print(olddata[0])
		# input()
		displaylist = []
		for stock in newdata:
			for i in olddata:
				if stock[1] == i[1]:
					X = []
					# X.append(newdata[0])
					X.append(i[0])
					X.append(i[1])
					stockdf = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+i[1]+'\"' , connection, index_col = 'Date')
					# print(stockdf)
					buyclose = stockdf.ix[i[0]]['Close']
					# print(buyclose)
					# input()
					X.append(buyclose)
					X.append(i[2])
					X.append(i[3])
					X.append(i[4])
					if (i[4] - buyclose)/buyclose*100 < -3:
						X.append('STOP LOSS TRIGGERED')
					elif (i[4] - buyclose)/buyclose*100 > 5:
						X.append('TARGET ACHIEVED')
					else:
						X.append('HOLD')
					gain = (i[4] - buyclose)/buyclose*100
					# print(repr(gain))
					X.append(gain)
					displaylist.append(X)
					break

		with open('StocksQuantity.json', 'w') as outfile:
			json.dump(olddata, outfile)

	# print(stocks_to_be_removed)
	# print(olddata)
	# input()
	if stocks_to_be_removed == oldstocks:
		# print('high')
		freed_value = 0
		for stock in olddata:
			freed_value = freed_value + stock[3]

		stockscount = 0
		for stock in newdata:
			stockscount = stockscount + 1

		amount_per_stock = round(freed_value/stockscount,2)

		displaylist = []
		stockquantity = []
		for stock in newdata:
			Y = []
			X = []
			stockdf = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stock[1]+'\"' , connection, index_col = 'Date')
			X.append(stock[0])
			X.append(stock[1])
			buyclose = stockdf.ix[stock[0]]['Close']
			X.append(buyclose)
			quantity = math.floor(amount_per_stock/buyclose)
			X.append(quantity)
			value = quantity*buyclose
			X.append(value)
			X.append(buyclose)
			X.append('HOLD')
			X.append(0)
			displaylist.append(X)

			Y.append(stock[0])
			Y.append(stock[1])
			Y.append(quantity)
			Y.append(value)
			Y.append(buyclose)
			stockquantity.append(Y)

		with open('StocksQuantity.json', 'w') as outfile:
			json.dump(stockquantity, outfile)

	if stocks_to_be_removed != oldstocks and stocks_to_be_removed:

		freed_value = 0
		for stock in stocks_to_be_removed:
			for i in olddata:
				if i[1] == stock:
					freed_value = freed_value + i[3]

		stockscount = 0
		for stock in newstocks - oldstocks:
			stockscount = stockscount + 1

		amount_per_stock = round(freed_value/stockscount,2)

		keep_list = []
		for i in olddata:
			if i[1] not in stocks_to_be_removed:
				keep_list.append(i)

		displaylist = []
		stockquantity = []
		newstocksadded = newstocks - oldstocks
		# print(newstocksadded)
		# print(freed_value)
		# print(amount_per_stock)
		for i in newstocksadded:
			for stock in newdata:
				if stock[1] == i:
					X = []
					Y = []
					X.append(stock[0])
					X.append(stock[1])
					stockdf = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stock[1]+'\"' , connection, index_col = 'Date')
					buyclose = stockdf.ix[stock[0]]['Close']
					X.append(buyclose)
					quantity = math.floor(amount_per_stock/buyclose)
					X.append(quantity)
					value = quantity*buyclose
					X.append(value)
					X.append(buyclose)
					X.append('HOLD')
					X.append(0)

					Y.append(stock[0])
					Y.append(stock[1])
					Y.append(quantity)
					Y.append(value)
					Y.append(buyclose)
					stockquantity.append(Y)
					displaylist.append(X)

		# print(displaylist)
		# input()

		for stock in olddata:
			if stock[1] not in stocks_to_be_removed:
				for i in newdata:
					if i[1] == stock[1]:
						Y = []
						X = []
						X.append(stock[0])
						X.append(stock[1])
						stockdf = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stock[1]+'\"' , connection, index_col = 'Date')
						# print(stockdf.tail(0))
						# input()
						buyclose = stockdf.ix[i[0]]['Close']
						X.append(buyclose)
						X.append(stock[2])
						X.append(stock[3])
						X.append(stock[4])
						if (stock[4] - buyclose)/buyclose*100 < -3:
							X.append('STOP LOSS TRIGGERED')
						elif (stock[4] - buyclose)/buyclose*100 > 5:
							X.append('TARGET ACHIEVED')
						else:
							X.append('HOLD')
						gain = (stock[4] - buyclose)/buyclose*100
						X.append(gain)

						Y.append(stock[0])
						Y.append(stock[1])
						Y.append(stock[2])
						Y.append(stock[3])
						Y.append(stock[4])
						stockquantity.append(Y)
						displaylist.append(X)

		with open('StocksQuantity.json', 'w') as outfile:
			json.dump(stockquantity, outfile)


	total_value = 0
	for stock in displaylist:
		total_value = total_value + stock[4]



###############################################################






	# connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
	# cursor = connection.cursor()
	# with open('TrueStocks.json') as json_file:
	# 	stockwithbuydate = json.load(json_file)

	# total_value = 15000
	# netvalue = 15000
	# amount_per_stock = 1000
	# oldstocks = []
	# for stock in stockwithbuydate:
	# 	X = []
	# 	stockdf = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stock[1]+'\"' , connection, index_col = 'Date')
	# 	# print(stockdf.ix[stock[0]])
	# 	buy_price = stockdf.ix[stock[0]]['Close']
	# 	quantity = round(1000/buy_price,0)
	# 	value = 1000
	# 	X.append(stock[0])
	# 	X.append(stock[1])
	# 	X.append(buy_price)
	# 	X.append(quantity)
	# 	X.append(value)
	# 	X.append(buy_price)
	# 	X.append('HOLD')
	# 	X.append(0)
	# 	oldstocks.append(X)

	# stockquantity = []

	# for stock in oldstocks:
	# 	X = []
	# 	X.append(stock[0])
	# 	X.append(stock[1])
	# 	X.append(stock[3])
	# 	X.append(stock[4])
	# 	X.append(stock[5])
	# 	stockquantity.append(X)

	# with open('StocksQuantity.json', 'w') as outfile:
	# 	json.dump(stockquantity, outfile)




	# newdata = set()
	# olddata = set()

	# with open('YYY.json') as json_file:
	# 	newstocks = json.load(json_file)

	# with open('OLD.json') as json_file:
	# 	oldstocks = json.load(json_file)

	# for stock in newstocks:
	# 	newdata.add(stock[0])
	# for stock in oldstocks:
	# 	olddata.add(stock[0])

	# print(newdata,olddata)

	# # stocks_to_be_removed = olddata - newdata

	# freed_value = 0

	# stocks_to_be_removed = olddata - newdata
	# print(stocks_to_be_removed)
	# if stocks_to_be_removed != olddata and stocks_to_be_removed:
	# 	for stock in stocks_to_be_removed:
	# 		for i in oldstocks:
	# 			if stock == i[0]:
	# 				freed_value = freed_value + i[3]
	# 				break

	# 	print(freed_value)


		
	# 	keep_list = []
	# 	for i in oldstocks:
	# 		if i[0] not in stocks_to_be_removed:
	# 			keep_list.append(i)


	# 	oldstocks = []
	# 	oldstocks = keep_list
	# 	# print(oldstocks)
	# 	# input()		

	# 	# print(len(oldstocks))
	# 	# for i in indices:
	# 	# 	del oldstocks[i]


	# 	# print(oldstocks)
	# 	# input()

	# 	count = 0
	# 	for stock in newdata - olddata:
	# 		count = count + 1

	# 	if count!= 0:
	# 		amount_per_stock = freed_value/count

	# 	for stock in newdata - olddata:
	# 		for i in newstocks:
	# 			if i[0] == stock:
	# 				X = [] 
	# 				X.append(i[0])
	# 				X.append(i[1])
	# 				X.append(round(amount_per_stock/i[2]))
	# 				X.append(amount_per_stock)
	# 				X.append(i[2])
	# 				if (i[2] - i[1])/i[1]*100 < -3:
	# 					X.append('STOP LOSS TRIGGERED')
	# 				elif (i[2] - i[1])/i[1]*100 > 5:
	# 					X.append('TARGET ACHIEVED')
	# 				else:
	# 					X.append('HOLD')
	# 				X.append((i[2] - i[1])/i[1]*100)
	# 				oldstocks.append(X)
	# 				break
	# 	with open('OLD.json', 'w') as outfile:
	# 	    json.dump(oldstocks, outfile)
	# 	# else:
	# 	# 	for i in newstocks:
	# 	# 		X = [] 
	# 	# 		X.append(i[0])
	# 	# 		X.append(i[1])
	# 	# 		X.append(round(amount_per_stock/i[2]))
	# 	# 		X.append(amount_per_stock)
	# 	# 		X.append(i[2])
	# 	# 		if (i[2] - i[1])/i[1]*100 < -3:
	# 	# 			X.append('STOP LOSS TRIGGERED')
	# 	# 		elif (i[2] - i[1])/i[1]*100 > 5:
	# 	# 			X.append('TARGET ACHIEVED')
	# 	# 		else:
	# 	# 			X.append('HOLD')
	# 	# 		X.append((i[2] - i[1])/i[1]*100)
	# 	# 		oldstocks.append(X)
	# elif stocks_to_be_removed == olddata:
	# 	freed_value = 0
	# 	for stock in oldstocks:
	# 		freed_value = freed_value + stock[3]

	# 	amount_per_stock = freed_value/len(newdata)
	# 	oldstocks = []
	# 	for i in newstocks:
	# 		X = [] 
	# 		X.append(i[0])
	# 		X.append(i[1])
	# 		X.append(round(amount_per_stock/i[2]))
	# 		X.append(amount_per_stock)
	# 		X.append(i[2])
	# 		if (i[2] - i[1])/i[1]*100 < -3:
	# 			X.append('STOP LOSS TRIGGERED')
	# 		elif (i[2] - i[1])/i[1]*100 > 5:
	# 			X.append('TARGET ACHIEVED')
	# 		else:
	# 			X.append('HOLD')
	# 		X.append((i[2] - i[1])/i[1]*100)
	# 		oldstocks.append(X)
	# 	with open('OLD.json', 'w') as outfile:
	# 	    json.dump(oldstocks, outfile)
	# elif len(stocks_to_be_removed) == 0:
	# 	new_close = []
	# 	for stock in newstocks:
	# 		new_close.append(stock[2])

	# 	count = 0
	# 	for stock in oldstocks:
	# 		stock[3] = stock[2] * new_close[count]
	# 		stock[4] = new_close[count]
	# 		if (stock[4] - stock[1])/stock[1]*100 < -3:
	# 			stock[5] = 'STOP LOSS TRIGGERED'
	# 		elif (stock[4] - stock[1])/stock[1]*100 > 5:
	# 			stock[5] = 'TARGET ACHIEVED'
	# 		else:
	# 			stock[5] = 'HOLD'
	# 		stock[6] = (stock[4] - stock[1])/stock[1]*100

	# 		count = count + 1

	# 	with open('OLD.json', 'w') as outfile:
	# 	    json.dump(oldstocks, outfile)



	# newdata = set()
	# olddata = set()
	# displaydata = []
	# with open('temp.json') as json_file:
	# 	newstocks = json.load(json_file)

	# with open('OLD.json') as json_file:
	# 	oldstocks = json.load(json_file)

	# for stock in newstocks:
	# 	newdata.add(stock[0])
	# for stock in oldstocks:
	# 	olddata.add(stock[0])

	# freed_value = 0

	# stocks_to_be_removed = olddata - newdata
	# for stock in stocks_to_be_removed:
	# 	for i in oldstocks:
	# 		if stock == i[0]:
	# 			freed_value = freed_value + i[3]
	# 			break

	# count = 0
	# for stock in newdata - olddata:
	# 	count = count + 1

	# if count!= 0:
	# 	amount_per_stock = freed_value/count

	# for stock in stocks_to_be_removed:
	# 	del oldstocks[oldstocks[0] == stock]

	# for stock in newdata - olddata:
	# 	for i in newstocks:
	# 		if stock == i[0]:
	# 			X = []
	# 			X.append(i[0])
	# 			X.append(i[1])
	# 			X.append(amount_per_stock/i[4])
	# 			X.append(amount_per_stock/i[4]*)
	# 			displaydata.append(i)
	# 		break
	# for stock in newdata & olddata:
	# 	for i in oldstocks:
	# 		if stock == i[0]:
	# 			displaydata.append(i)

	# with open('OLD.json', 'w') as outfile:
	# 	json.dump(displaydata, outfile)


	# for stock in data:
	# 	X = []
	# 	stock[1] = round(stock[1],2)
	# 	number_of_shares = round(float(1000/stock[1]),0)
	# 	value = number_of_shares*stock[4]
	# 	X.append(stock[0])
	# 	X.append(stock[1])
	# 	X.append(number_of_shares)
	# 	X.append(value)
	# 	X.append(stock[2])
	# 	X.append(stock[3])
	# 	X.append(stock[4])
	# 	displaylist.append(X)


	


	return render_template('./MainPage1.html', mainstocks = displaylist, netvalue = total_value, gainloss = (total_value - 15000)/15000*100)

def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter

@app.route('/Stocks/<stockname>')
def show_image(stockname):
	connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
	cursor = connection.cursor()

	stockdf = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stockname+'\"' , connection)
	displaydf = stockdf.iloc[-500:]
	fig = plt.figure()
	ax1 = plt.subplot2grid((1,1), (0,0))
	stock_data = []
	x = 0
	y = len(displaydf['Date'])
	print(y)
	dates = list(displaydf['Date'])
	print(dates[0])
	opens = list(displaydf['Open'])
	highs = list(displaydf['High'])
	lows = list(displaydf['Low'])
	closes = list(displaydf['Close'])
	volumes = list(displaydf['Volume'])
	for i in range(len(dates)):
		X = dates[i]+','+str(opens[i])+','+str(highs[i])+','+str(lows[i])+','+str(closes[i])+','+str(volumes[i])
		stock_data.append(X)
	# print(stock_data)
	# input()

	# date, closep, highp, lowp, openp, closep, adjclose, volume 
	temp_data= np.loadtxt(stock_data,delimiter=',',unpack=True,converters={0: bytespdate2num('%Y-%m-%d')})
	# print(temp_data)
	# input()
	ohlc = []

	# print('HI')
	while x < y:
		append_me = temp_data[0][x], temp_data[1][x], temp_data[2][x], temp_data[3][x], temp_data[4][x], temp_data[5][x]
		ohlc.append(append_me)
		# print(append_me)
		# input()
		x+=1

	candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')

	for label in ax1.xaxis.get_ticklabels():
		label.set_rotation(45)

	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax1.grid(True)

	# print('HI')

	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(stockname)
	plt.legend()
	plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
	plt.savefig('.\static\StockImages\\'+stockname+'.png')
	# print('hi')
    # plt.show()



	return render_template('./Display.html', image = stockname+'.png')

@app.route('/MyPortfolio')
def create_portfolio():

	stockslist = []

	connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
	cursor = connection.cursor()

	cursor.execute('SELECT distinct StockSymbol from Stocks')
	currentstocks = cursor.fetchall()

	for stock in currentstocks:
		stockname = re.findall(r"[A-Z]+",str(stock))[0]
		stockslist.append(stockname)

	print(stockslist)



	return render_template('./MyPortfolio.html' , displaylist = stockslist)

@app.route('/Quantity/<stock>')
def enter_quantity(stock):

	return render_template('MyQuantity.html' , stockname = stock)


@app.route('/DataDump/<stockname>', methods = ['POST','GET'])
def datatump(stockname):
	final_list = []
	if request.method == 'POST':
		POST_QUANTITY = str(request.form['quantity'])
		POST_PRICE = str(request.form['price'])
		connection = sqlite3.connect("C:\Sqlite\\users.db")
		cursor = connection.cursor()
		final_list.append('ABC')
		final_list.append(stockname)
		final_list.append(round(float(POST_PRICE)*float(POST_QUANTITY),2))
		final_list.append(POST_QUANTITY)
		cursor.execute('SELECT * FROM Users where StockSymbol = \"'+stockname+'\"')
		stockname_check = cursor.fetchall()
		if stockname_check:
			investedamount_new = round(float(POST_PRICE)*float(POST_QUANTITY),2)

			cursor.execute('UPDATE Users SET InvestedAmount = InvestedAmount + ? where StockSymbol = \"'+stockname+'\"',(investedamount_new,))
			cursor.execute('UPDATE Users SET Quantity = Quantity + ? where StockSymbol = \"'+stockname+'\"',(POST_QUANTITY,))
			connection.commit()
		else:
			final_columns = ['Name','StockSymbol','InvestedAmount','Quantity']
			query = 'insert into Users({0}) values ({1})'
			query = query.format(','.join(final_columns), ','.join('?' * len(final_columns)))
			cursor.execute(query, final_list)
			connection.commit()
		if str(request.form['submit']) == 'Add Stock':
			return redirect(url_for('create_portfolio'))
		else:
			return redirect(url_for('render'))

@app.route('/DisplayPortfolio/<user>')
def displayportfolio(user):
	templist = []
	current_time = int(time.time())
	connection = sqlite3.connect("C:\Sqlite\\users.db")
	cursor = connection.cursor()
	df = pd.read_sql_query('SELECT * FROM Users where Name = \"'+user+'\"' , connection)
	total = 0
	networth = 0
	today_date = datetime.datetime.now()
	yesterday_date = today_date - datetime.timedelta(days = 1)
	today = int(today_date.timestamp())
	yesterday = int(yesterday_date.timestamp())
	yes = 0
	while yes != 1:
		http_query = "https://query1.finance.yahoo.com/v7/finance/download/"+'MSFT'+"?period1="+ str(yesterday) + "&period2="+str(today)+"&interval=1d&events=history&crumb="+crumb
		symbol_text = intermediate_session.get(http_query)
		symbol_text = symbol_text.text.split("\n")
		if symbol_text[0] == '{"chart":{"result":null,"error":{"code":"Not Found","description":"Encountered an error when generating the download data"}}}':
			yesterday_date = yesterday_date - datetime.timedelta(days = 1)
			today_date = today_date -  datetime.timedelta(days = 1)
			today = int(today_date.timestamp())
			yesterday = int(yesterday_date.timestamp())
		else:
			yes = 1



	for i,row in df.iterrows():
		http_query = "https://query1.finance.yahoo.com/v7/finance/download/"+row['StockSymbol']+"?period1="+ str(yesterday) + "&period2="+str(today)+"&interval=1d&events=history&crumb="+crumb
		symbol_text = intermediate_session.get(http_query)
		symbol_text = symbol_text.text.split("\n")
		# print(symbol_text)

		close = round(float(symbol_text[1].split(',')[4]),2)
		# gain = round((close - float(row['Price']))/float(row['Price'])*100,2)
		X =[]
		# print(row['StockSymbol'])
		X.append(row['StockSymbol'])
		X.append(row['InvestedAmount'])
		X.append(row['Quantity'])
		currentvalue = round(close*row['Quantity'],2)
		X.append(currentvalue)
		gain = (currentvalue - row['InvestedAmount'])/row['InvestedAmount']*100
		X.append(gain)
		total = total + row['InvestedAmount']
		networth = networth + currentvalue
		templist.append(X)
	gain_loss = round((networth - total)/total*100,2)


	return render_template('./DisplayPortfolio1.html', mainstocks = templist, investment = total, networth = round(networth,2), totalgl = gain_loss)




if __name__ == '__main__':
	FOUND = 1
	global crumb
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

	app.secret_key = os.urandom(12)

	app.run(host='0.0.0.0',port=2000)
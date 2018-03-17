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

app = Flask(__name__)

@app.route('/')
def render():
	
	with open('YYY.json') as json_file:
		data = json.load(json_file)
	


	return render_template('./Arightstock.html', mainstocks = data)

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

	print('HI')
	while x < y:
		append_me = temp_data[0][x], temp_data[1][x], temp_data[2][x], temp_data[3][x], temp_data[4][x], temp_data[5][x]
		ohlc.append(append_me)
		x+=1

	candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')

	for label in ax1.xaxis.get_ticklabels():
		label.set_rotation(45)

	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax1.grid(True)

	print('HI')

	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(stockname)
	plt.legend()
	plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
	plt.savefig('.\static\StockImages\\'+stockname+'.png')
	print('hi')
    # plt.show()



	return render_template('./Display.html', image = stockname+'.png')


if __name__ == '__main__':

    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0',port=2000)
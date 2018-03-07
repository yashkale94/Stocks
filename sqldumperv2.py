import os
import csv
import pandas as pd
import sqlite3
import datetime
import pandas_datareader.data as web
import urllib.request
import requests
import numpy as np
import ast
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import json
import re

FILES = os.listdir("Y:\Work\StocksSelectorNewData\FinalStocksTable\\")

# connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
# cursor = connection.cursor()

# #cursor.execute('drop table Stocks')
# query1 = """create table Stocks ( ID INT AUTO_INCREMENT PRIMARY KEY , StockSymbol VARCHAR, Date VARCHAR, Open FLOAT, High FLOAT, Low FLOAT, Close FLOAT, Volume FLOAT, PriceChange FLOAT, VolumeChange FLOAT, OBV FLOAT, OBVChange Float, TwoDayEMA FLOAT,ThreeDayEMA FLOAT,FourDayEMA FLOAT,FiveDayEMA FLOAT,SixDayEMA FLOAT,SevenDayEMA FLOAT,EightDayEMA FLOAT,
# NineDayEMA FLOAT,TenDayEMA FLOAT,ElevenDayEMA FLOAT,TwelveDayEMA FLOAT,ThirteenDayEMA FLOAT,FourteenDayEMA FLOAT,FifteenDayEMA FLOAT,SixteenDayEMA FLOAT,SeventeenDayEMA FLOAT,EighteenDayEMA FLOAT,NineteenDayEMA FLOAT,TwentyDayEMA FLOAT,TwentyOneDayEMA FLOAT,TwentyTwoDayEMA FLOAT,TwentyThreeDayEMA FLOAT,TwentyFourDayEMA FLOAT,TwentyFiveDayEMA FLOAT,TwentySixDayEMA FLOAT, TwentySevenDayEMA FLOAT, TwentyEightDayEMA FLOAT, TwentyNineDayEMA FLOAT,   
# ThirtyDayEMA FLOAT,ThirtyOneDayEMA FLOAT,ThirtyTwoDayEMA FLOAT,ThirtyThreeDayEMA FLOAT,ThirtyFourDayEMA FLOAT,ThirtyFiveDayEMA FLOAT,ThirtySixDayEMA FLOAT,ThirtySevenDayEMA FLOAT,ThirtyEightDayEMA FLOAT,ThirtyNineDayEMA FLOAT,FortyDayEMA FLOAT,AverageGain FLOAT,AverageLoss FLOAT,Gain FLOAT,Loss FLOAT,HC FLOAT,H FLOAT,HS1 FLOAT,HS2 FLOAT,C FLOAT,DHL1 FLOAT,DHL2 FLOAT,RSI FLOAT,SMI FLOAT,MACDLine FLOAT,Signal FLOAT,Histogram FLOAT,Google FLOAT,Apple FLOAT,Amazon FLOAT,Microsoft FLOAT, NasdaqComposite FLOAT, R1Target FLOAT, R2Target FLOAT);""" 
# cursor.execute(query1)
# final_columns = ['StockSymbol','Date','Open','High','Low','Close','Volume','PriceChange','VolumeChange','OBV','OBVChange','TwoDayEMA','ThreeDayEMA','FourDayEMA','FiveDayEMA','SixDayEMA','SevenDayEMA','EightDayEMA','NineDayEMA','TenDayEMA','ElevenDayEMA','TwelveDayEMA','ThirteenDayEMA','FourteenDayEMA','FifteenDayEMA','SixteenDayEMA','SeventeenDayEMA','EighteenDayEMA','NineteenDayEMA','TwentyDayEMA','TwentyOneDayEMA','TwentyTwoDayEMA','TwentyThreeDayEMA','TwentyFourDayEMA','TwentyFiveDayEMA','TwentySixDayEMA', 'TwentySevenDayEMA', 'TwentyEightDayEMA','TwentyNineDayEMA','ThirtyDayEMA','ThirtyOneDayEMA','ThirtyTwoDayEMA','ThirtyThreeDayEMA','ThirtyFourDayEMA','ThirtyFiveDayEMA','ThirtySixDayEMA','ThirtySevenDayEMA','ThirtyEightDayEMA','ThirtyNineDayEMA','FortyDayEMA','AverageGain','AverageLoss','Gain','Loss','HC','H','HS1','HS2','C','DHL1','DHL2','RSI','SMI','MACDLine','Signal','Histogram','Google','Apple','Amazon','Microsoft','NasdaqComposite','R1Target','R2Target']

# query = 'insert into Stocks({0}) values ({1})'
# query = query.format(','.join(final_columns), ','.join('?' * len(final_columns)))
# for file in FILES:
# 	print(file)
# 	count = 0
# 	with open("Y:\Work\StocksSelectorNewData\FinalStocksTable\\"+file,'r') as f:
# 		reader = csv.reader(f)
# 		name = file.split('-')[0]
# 		for data in reader:
# 			if count != 0:
# 				data.insert(0,name)
# 				cursor.execute(query, data)
# 			count+=1

# connection.commit()

def predict_price(stockdf):

	TrainData = stockdf.iloc[39:-40]
	PredictData = stockdf.iloc[-1:]

	X_train = []
	Y_train = []
	# print(PredictData.columns)
	# input()

	exclude_index_list = [0,1,6,9,10,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,65,71,72]

	for i,row in TrainData.iterrows():
		record = row.values
		
		X = []
		for j in range(len(record)):
			if j not in exclude_index_list:
				X.append(float(record[j]))

		if float(record[72]) <= 0:
			Y_train.append(0)
			
		elif float(record[72]) >0 and float(record[72]) <= 2:
			Y_train.append(1)
			
		elif float(record[72]) > 2  and float(record[72]) <=5:
			Y_train.append(2)
			
		elif float(record[72]) > 5 and float(record[72]) <=10:
			Y_train.append(3)
			
		elif float(record[72]) > 10:
			Y_train.append(4)

		X_train.append(X)

	for i,row in PredictData.iterrows():
		record = row.values

		X = []
		predict_list = []
		for j in range(len(record)):
			if j not in exclude_index_list:
				X.append(record[j])

		predict_list.append(X)



	LR = SVC(kernel = 'rbf', C=1e3 , gamma = 0.01)
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	predict_list = scaler.transform(predict_list)
	LR.fit(X_train, Y_train)
	Y_pred = LR.predict(predict_list)

	return(Y_pred[0])


def insert_affinity_data(prev_day,df,today_day,final_list,index_list,olddf,affinitydf):
	
	affinity_list = []

	if final_list[5] > 0:
		affinity_list.append('PC+')
	elif final_list[5] < 0:
		affinity_list.append('PC-')
	else:
		affinity_list.append('PCNC')

	if final_list[6] > 0:
		affinity_list.append('VC+')
	elif final_list[6] < 0:
		affinity_list.append('VC+')
	else:
		affinity_list.append('VCNC+')


	if final_list[8] > 0:
		affinity_list.append('OBVP')
	elif final_list[8] < 0:
		affinity_list.append('OBVN')
	else:
		affinity_list.append('OBVNC')


	for i in range(2,41):
		if today_day['Close'] > final_list[i+7]:
			affinity_list.append('Above'+str(i)+'DayEMA')
		elif today_day['Close'] < final_list[i+7]:
			affinity_list.append('Below'+str(i)+'DayEMA')
		elif today_day['Close'] == final_list[i+7]:
			affinity_list.append('Same'+str(i)+'DayEMA')

	if float(final_list[61]) <= -21:
		affinity_list.append('MACD1')
	elif float(final_list[61]) > -21 and float(final_list[61]) <= -0.3:
		affinity_list.append('MACD2')
	elif float(final_list[61]) > -0.3 and float(final_list[61]) <= 0:
		affinity_list.append('MACD3')
	elif float(final_list[61]) > 0 and float(final_list[61]) <= 0.4:
		affinity_list.append('MACD4')
	elif float(final_list[61]) > 0.4 and float(final_list[61]) <= 8:
		affinity_list.append('MACD5')
	elif float(final_list[61]) > 8:
		affinity_list.append('MACD6')

	if float(final_list[62]) <= -18:
		affinity_list.append('Signal1')
	elif float(final_list[62]) > -18 and float(final_list[62]) <= -0.3:
		affinity_list.append('Signal2')
	elif float(final_list[62]) > -0.3 and float(final_list[62]) <= 0:
		affinity_list.append('Signal3')
	elif float(final_list[62]) > 0 and float(final_list[62]) <= 0.4:
		affinity_list.append('Signal4')
	elif float(final_list[62]) > 0.4 and float(final_list[62]) <= 7:
		affinity_list.append('Signal5')
	elif float(final_list[62]) > 7:
		affinity_list.append('Signal6')

	if float(final_list[59]) <= 30:
		affinity_list.append('RSIbelow30')
	elif float(final_list[59]) > 30 and float(final_list[59]) <=70:
		affinity_list.append('RSImidrange')
	elif float(final_list[59]) > 70:
		affinity_list.append('RSIabove70')

	if float(final_list[60]) >= -100 and float(final_list[60]) <= -40:
		affinity_list.append('LowSMI')
	elif float(final_list[60]) >= -40 and float(final_list[60]) <= 40:
		affinity_list.append('MidSMI')
	elif float(final_list[60]) > 40 and float(final_list[60]) <= 100:
		affinity_list.append('HighSMI')


	R1TargetDate = today_date - datetime.timedelta(days=7)
	R2TargetDate = today_date - datetime.timedelta(days=14)

	R1TargetDate = R1TargetDate.strftime('%Y-%m-%d')
	R2TargetDate = R2TargetDate.strftime('%Y-%m-%d')

	R1Target = ''
	R2Target = ''
	while(R1Target == ''):
		try:
			R1Target = (today_day['Close'] - olddf.ix[R1TargetDate]['Close'])/olddf.ix[R1TargetDate]['Close']*100

		except:
			R1TargetDate = datetime.datetime(int(R1TargetDate.split('-')[0]),int(R1TargetDate.split('-')[1]),int(R1TargetDate.split('-')[2])).timestamp()
			R1TargetDate = datetime.datetime.fromtimestamp(R1TargetDate)
			R1TargetDate = R1TargetDate - datetime.timedelta(days=1)
			R1TargetDate = R1TargetDate.strftime('%Y-%m-%d')


	while(R2Target == ''):
		try:
			R2Target = (today_day['Close'] - olddf.ix[R2TargetDate]['Close'])/olddf.ix[R2TargetDate]['Close']*100
		except:
			R2TargetDate = datetime.datetime(int(R2TargetDate.split('-')[0]),int(R2TargetDate.split('-')[1]),int(R2TargetDate.split('-')[2])).timestamp()
			R2TargetDate = datetime.datetime.fromtimestamp(R2TargetDate)
			R2TargetDate = R2TargetDate - datetime.timedelta(days=1)
			R2TargetDate = R2TargetDate.strftime('%Y-%m-%d')


	target_list = []

	if R1Target <=-2:
		target_list.append('R10')
	elif R1Target > -2 and R1Target <= -1:
		target_list.append('R11')
	elif R1Target > -1 and R1Target <= 0:
		target_list.append('R12')
	elif R1Target > 0 and R1Target <= 1.5:
		target_list.append('R13')
	elif R1Target > 1.5 and R1Target <= 2.5:
		target_list.append('R14')
	elif R1Target > 2.5:
		target_list.append('R15')

	if R2Target <=-4:
		target_list.append('R20')
	elif R2Target > -4 and R2Target <= -2:
		target_list.append('R21')
	elif R2Target > -2 and R2Target <= 0:
		target_list.append('R22')
	elif R2Target > 0 and R2Target <= 2:
		target_list.append('R23')
	elif R2Target > 2 and R2Target <= 5:
		target_list.append('R24')
	elif R2Target > 5:
		target_list.append('R25')


	return(R1TargetDate, R2TargetDate, R1Target,R2Target,affinity_list,target_list)	

def check_rule(affinity_list):
	rulesdf = pd.read_csv("Y:\Work\StocksSelectorNewData\AllStockEMArulesR2.csv")
	for lhs,rhs in zip(rulesdf.LHS,rulesdf.RHS):
		accept_list = []
		if 'R25' in rhs:
			lhs = ast.literal_eval(lhs)
			for entry in lhs:
				entry = entry.replace(" ","")
				entry = entry.replace("Ema","EMA")
				if entry in affinity_list:
					accept_list.append(1)
				else:
					accept_list.append(0)
			if 0 in accept_list:
				continue
				#print('Rule'+str(lhs)+" =>"+str(rhs)+" not satisfied")
			else:
				print('Rule satisfied')
				return(1)
	return(0)

def calculate_technicals(prev_day,df,today_day):
	High_values = list(df['High'].values)
	High_values.append(today_day['High'])
	
	Low_values = list(df['High'].values)
	Low_values.append(today_day['Low'])

	HC = (max(High_values) + min(Low_values))/2
	H = today_day['Close'] - HC 

	HS1 = ((H - prev_day['HS1'] ) * 2/4) + prev_day['HS1']

	HS2 = ((HS1 - prev_day['HS2'] ) * 2/4) + prev_day['HS2']

	C = (max(High_values) - min(Low_values))/2

	DHL1 = ((C - prev_day['DHL1']) * 2/4) + prev_day['DHL1']

	DHL2 = ((DHL1 - prev_day['DHL2']) * 2/4) + prev_day['DHL2']

	SMI = HS2/DHL2

	difference = today_day['Close'] - prev_day['Close']
	if difference > 0:
		Gain = difference
		Loss = 0
	elif difference < 0:
		Loss = abs(difference)
		Gain = 0
	else:
		Gain = 0
		Loss = 0

	Average_Gain = (prev_day['AverageGain'] * 13 + Gain) / 14
	Average_Loss = (prev_day['AverageLoss'] * 13 + Loss) / 14

	if Average_Loss == 0:
		RSI = 0
	else:
		RSI = 100 - (100/(1+(Average_Gain/Average_Loss)))


	price_change = ((today_day['Close'] - prev_day['Close'])/prev_day['Close'])*100
	volume_change = ((today_day['Volume'] - prev_day['Volume'])/prev_day['Volume'])*100
	two_day_ema = ((today_day['Close'] - prev_day['TwoDayEMA']) * 2/3) + prev_day['TwoDayEMA']
	three_day_ema = ((today_day['Close'] - prev_day['ThreeDayEMA']) * 2/4) + prev_day['ThreeDayEMA']
	four_day_ema = ((today_day['Close'] - prev_day['FourDayEMA']) * 2/5) + prev_day['FourDayEMA']
	five_day_ema = ((today_day['Close'] - prev_day['FiveDayEMA']) * 2/6) + prev_day['FiveDayEMA']
	six_day_ema = ((today_day['Close'] - prev_day['SixDayEMA']) * 2/7) + prev_day['SixDayEMA']
	seven_day_ema = ((today_day['Close'] - prev_day['SevenDayEMA']) * 2/8) + prev_day['SevenDayEMA']
	eight_day_ema = ((today_day['Close'] - prev_day['EightDayEMA']) * 2/9) + prev_day['EightDayEMA']
	nine_day_ema = ((today_day['Close'] - prev_day['NineDayEMA']) * 2/10) + prev_day['NineDayEMA']
	ten_day_ema = ((today_day['Close'] - prev_day['TenDayEMA']) * 2/11) + prev_day['TenDayEMA']
	eleven_day_ema = ((today_day['Close'] - prev_day['ElevenDayEMA']) * 2/12) + prev_day['ElevenDayEMA']
	twelve_day_ema = ((today_day['Close'] - prev_day['TwelveDayEMA']) * 2/13) + prev_day['TwelveDayEMA']
	thirteen_day_ema = ((today_day['Close'] - prev_day['ThirteenDayEMA']) * 2/14) + prev_day['ThirteenDayEMA']
	fourteen_day_ema = ((today_day['Close'] - prev_day['FourteenDayEMA']) * 2/15) + prev_day['FourteenDayEMA']
	fifteen_day_ema = ((today_day['Close'] - prev_day['FifteenDayEMA']) * 2/16) + prev_day['FifteenDayEMA']
	sixteen_day_ema = ((today_day['Close'] - prev_day['SixteenDayEMA']) * 2/17) + prev_day['SixteenDayEMA']
	seventeen_day_ema = ((today_day['Close'] - prev_day['SeventeenDayEMA']) * 2/18) + prev_day['SeventeenDayEMA']
	eighteen_day_ema = ((today_day['Close'] - prev_day['EighteenDayEMA']) * 2/19) + prev_day['EighteenDayEMA']
	nineteen_day_ema = ((today_day['Close'] - prev_day['NineteenDayEMA']) * 2/20) + prev_day['NineteenDayEMA']
	twenty_day_ema = ((today_day['Close'] - prev_day['TwentyDayEMA']) * 2/21) + prev_day['TwentyDayEMA']
	twentyone_day_ema = ((today_day['Close'] - prev_day['TwentyOneDayEMA']) * 2/22) + prev_day['TwentyOneDayEMA']
	twentytwo_day_ema = ((today_day['Close'] - prev_day['TwentyTwoDayEMA']) * 2/23) + prev_day['TwentyTwoDayEMA']
	twentythree_day_ema = ((today_day['Close'] - prev_day['TwentyThreeDayEMA']) * 2/24) + prev_day['TwentyThreeDayEMA']
	twentyfour_day_ema = ((today_day['Close'] - prev_day['TwentyFourDayEMA']) * 2/25) + prev_day['TwentyFourDayEMA']
	twentyfive_day_ema = ((today_day['Close'] - prev_day['TwentyFiveDayEMA']) * 2/26) + prev_day['TwentyFiveDayEMA']
	twentysix_day_ema = ((today_day['Close'] - prev_day['TwentySixDayEMA']) * 2/27) + prev_day['TwentySixDayEMA']
	twentyseven_day_ema = ((today_day['Close'] - prev_day['TwentySevenDayEMA']) * 2/28) + prev_day['TwentySevenDayEMA']
	twentyeight_day_ema = ((today_day['Close'] - prev_day['TwentyEightDayEMA']) * 2/29) + prev_day['TwentyEightDayEMA']
	twentynine_day_ema = ((today_day['Close'] - prev_day['TwentyNineDayEMA']) * 2/30) + prev_day['TwentyNineDayEMA']
	thirty_day_ema = ((today_day['Close'] - prev_day['ThirtyDayEMA']) * 2/31) + prev_day['ThirtyDayEMA']
	thirtyone_day_ema = ((today_day['Close'] - prev_day['ThirtyOneDayEMA']) * 2/32) + prev_day['ThirtyOneDayEMA']
	thirtytwo_day_ema = ((today_day['Close'] - prev_day['ThirtyTwoDayEMA']) * 2/33) + prev_day['ThirtyTwoDayEMA']
	thirtythree_day_ema = ((today_day['Close'] - prev_day['ThirtyThreeDayEMA']) * 2/34) + prev_day['ThirtyThreeDayEMA']
	thirtyfour_day_ema = ((today_day['Close'] - prev_day['ThirtyFourDayEMA']) * 2/35) + prev_day['ThirtyFourDayEMA']
	thirtyfive_day_ema = ((today_day['Close'] - prev_day['ThirtyFiveDayEMA']) * 2/36) + prev_day['ThirtyFiveDayEMA']
	thirtysix_day_ema = ((today_day['Close'] - prev_day['ThirtySixDayEMA']) * 2/37) + prev_day['ThirtySixDayEMA']
	thirtyseven_day_ema = ((today_day['Close'] - prev_day['ThirtySevenDayEMA']) * 2/38) + prev_day['ThirtySevenDayEMA']
	thirtyeight_day_ema = ((today_day['Close'] - prev_day['ThirtyEightDayEMA']) * 2/39) + prev_day['ThirtyEightDayEMA']
	thirtynine_day_ema = ((today_day['Close'] - prev_day['ThirtyNineDayEMA']) * 2/40) + prev_day['ThirtyNineDayEMA']
	forty_day_ema = ((today_day['Close'] - prev_day['FortyDayEMA']) * 2/41) + prev_day['FortyDayEMA']

	if today_day['Close'] > prev_day['Close']:
		OBV = prev_day['OBV'] + today_day['Volume']
	elif today_day['Close'] < prev_day['Close']:
		OBV = prev_day['OBV'] + today_day['Volume']
	elif today_day['Close'] == prev_day['Close']:
		OBV = 0

	if prev_day['OBV'] != 0:
		OBVchange = (OBV - prev_day['OBV'])/prev_day['OBV']*100
	else:
		OBVchange = OBV


	MACD_Line = twelve_day_ema - twentysix_day_ema

	Signal = ((MACD_Line - prev_day['Signal']) * 2/11 ) + prev_day['Signal']

	Histogram = MACD_Line - Signal

	return([today_day['Open'],today_day['High'],today_day['Low'],today_day['Close'],today_day['Volume'],price_change,volume_change,OBV,OBVchange,two_day_ema,three_day_ema,four_day_ema,
		five_day_ema,six_day_ema,seven_day_ema,eight_day_ema,nine_day_ema,ten_day_ema,eleven_day_ema,twelve_day_ema,thirteen_day_ema,fourteen_day_ema,fifteen_day_ema,sixteen_day_ema,
		seventeen_day_ema,eighteen_day_ema,nineteen_day_ema,twenty_day_ema,twentyone_day_ema,twentytwo_day_ema,twentythree_day_ema,twentyfour_day_ema,twentyfive_day_ema,
		twentysix_day_ema,twentyseven_day_ema,twentyeight_day_ema,twentynine_day_ema,thirty_day_ema,thirtyone_day_ema,thirtytwo_day_ema,thirtythree_day_ema,thirtyfour_day_ema,
		thirtyfive_day_ema,thirtysix_day_ema,thirtyseven_day_ema,thirtyeight_day_ema,thirtynine_day_ema,forty_day_ema,Average_Gain,Average_Loss,Gain,Loss,HC,H,HS1,HS2,C,DHL1,DHL2,RSI,SMI,MACD_Line,Signal,Histogram])

def insert_data(filename):
	stock_Name = filename
	dfcloses = None
	while dfcloses is None:
		try:
			dfcloses = web.DataReader(stock_Name, 'yahoo', period1, period2)
		except:
			pass

	df = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+stock_Name+'\"' , connection, index_col = 'Date')
	affinitydf = pd.read_sql_query('SELECT * FROM Affinity where StockSymbol = \"'+stock_Name+'\"' , connection, index_col = 'Date')
	dfs = df.iloc[-9:]
	index_list = df.index.values
	final_list = calculate_technicals(df.iloc[-1],dfs,dfcloses.ix[today_date])
	R1TargetDate, R2TargetDate, R1Target, R2Target, affinity_list, target_list = insert_affinity_data(df.iloc[-1],dfs,dfcloses.ix[today_date],final_list,index_list,df,affinitydf)

	cursor.execute('update Stocks set R1Target = '+str(R1Target)+ ' where Date = \"'+R1TargetDate+'\" and StockSymbol = \"'+stock_Name+'\"')
	cursor.execute('update Stocks set R2Target = '+str(R2Target)+ ' where Date = \"'+R2TargetDate+'\"and StockSymbol = \"'+stock_Name+'\"')

	cursor.execute('update Affinity set R1Target = \"'+str(target_list[0])+'\" where Date = \"'+R1TargetDate+'\" and StockSymbol = \"'+stock_Name+'\"')
	cursor.execute('update Affinity set R2Target = \"'+str(target_list[1])+'\" where Date = \"'+R2TargetDate+'\"and StockSymbol = \"'+stock_Name+'\"')

	final_affinity_columns = ['StockSymbol','Date','PricechangeCategory', 'VolumechangeCategory','OBVchangeCategory', 'ClosevsTwoDayEMA','ClosevsThreeDayEMA', 'ClosevsFourDayEMA','ClosevsFiveDayEMA', 'ClosevsSixDayEMA','ClosevsSevenDayEMA', 'ClosevsEightDayEMA', 'ClosevsNineDayEMA', 'ClosevsTenDayEMA','ClosevsElevenDayEMA','ClosevsTwelveDayEMA','ClosevsThirteenDayEMA','ClosevsFourteenDayEMA','ClosevsFifteenDayEMA',
	'ClosevsSixteenDayEMA','ClosevsSeventeenDayEMA','ClosevsEighteenDayEMA','ClosevsNineteenDayEMA','ClosevsTwentyDayEMA','ClosevsTwentyOneDayEMA','ClosevsTwentyTwoDayEMA','ClosevsTwentyThreeDayEMA','ClosevsTwentyFourDayEMA','ClosevsTwentyFiveDayEMA','ClosevsTwentySixDayEMA','ClosevsTwentySevenDayEMA','ClosevsTwentyEightDayEMA','ClosevsTwentyNineDayEMA','ClosevsThirtyDayEMA','ClosevsThirtyOneDayEMA','ClosevsThirtyTwoDayEMA',
	'ClosevsThirtyThreeDayEMA','ClosevsThirtyFourDayEMA','ClosevsThirtyFiveDayEMA','ClosevsThirtySixDayEMA','ClosevsThirtySevenDayEMA','ClosevsThirtyEightDayEMA','ClosevsThirtyNineDayEMA','ClosevsFortyDayEMA','MACDLine','Signal','RSI','SMI','R1Target','R2Target']

	final_columns = ['StockSymbol','Date','Open','High','Low','Close','Volume','PriceChange','VolumeChange','OBV','OBVChange','TwoDayEMA','ThreeDayEMA','FourDayEMA','FiveDayEMA','SixDayEMA','SevenDayEMA','EightDayEMA','NineDayEMA','TenDayEMA','ElevenDayEMA','TwelveDayEMA','ThirteenDayEMA','FourteenDayEMA','FifteenDayEMA','SixteenDayEMA','SeventeenDayEMA','EighteenDayEMA','NineteenDayEMA','TwentyDayEMA','TwentyOneDayEMA','TwentyTwoDayEMA','TwentyThreeDayEMA','TwentyFourDayEMA','TwentyFiveDayEMA','TwentySixDayEMA', 'TwentySevenDayEMA', 'TwentyEightDayEMA','TwentyNineDayEMA','ThirtyDayEMA','ThirtyOneDayEMA','ThirtyTwoDayEMA','ThirtyThreeDayEMA','ThirtyFourDayEMA','ThirtyFiveDayEMA','ThirtySixDayEMA','ThirtySevenDayEMA','ThirtyEightDayEMA','ThirtyNineDayEMA','FortyDayEMA','AverageGain','AverageLoss','Gain','Loss','HC','H','HS1','HS2','C','DHL1','DHL2','RSI','SMI','MACDLine','Signal','Histogram','Google','Apple','Amazon','Microsoft','NasdaqComposite','R1Target','R2Target']
	query = 'insert into Stocks({0}) values ({1})'
	query = query.format(','.join(final_columns), ','.join('?' * len(final_columns)))

	affinity_query = 'insert into Affinity({0}) values({1})'
	affinity_query = affinity_query.format(','.join(final_affinity_columns), ','.join('?' * len(final_affinity_columns)))

	flag = check_rule(affinity_list)
	if flag == 1:
		predicted_price = predict_price(df)
		predicted_stocks_list.append(stock_Name)
	# else:
	# 	print('DNQ')

	for i in priceleaders_list:
		final_list.append(i)

	for i in range(2):
		final_list.append(np.nan)
		affinity_list.append(np.nan)

	final_list.insert(0,today_date.strftime('%Y-%m-%d'))
	affinity_list.insert(0,today_date.strftime('%Y-%m-%d'))
	final_list.insert(0,stock_Name)
	affinity_list.insert(0,stock_Name)
	cursor.execute(query, final_list)
	cursor.execute(affinity_query, affinity_list)
	connection.commit()





connection = sqlite3.connect("C:\Sqlite\\RightStock.db")
cursor = connection.cursor()
cursor.execute('SELECT distinct StockSymbol from Stocks')
currentstocks = cursor.fetchall()
temp_stock = re.findall(r"[A-Z]+",str(currentstocks[0]))[0]

df = pd.read_sql_query('SELECT * FROM Stocks where StockSymbol = \"'+temp_stock+'\"', connection)

HH_LL_day = df.iloc[-9:]['Date']
old_date_list = df.iloc[-1]['Date'].split('-')

prev_date = datetime.datetime(int(old_date_list[0]),int(old_date_list[1]),int(old_date_list[2])).timestamp()
prev_date = datetime.datetime.fromtimestamp(prev_date)
session_prev_date = prev_date + datetime.timedelta(days=1)
session_period_1 = str(session_prev_date.timestamp()).replace(".0","")
today_date = prev_date + datetime.timedelta(days=1)
session_today_date = prev_date + datetime.timedelta(days=2)

session_period_2 = str(session_today_date.timestamp()).replace(".0","")
today_date_list = today_date.strftime('%Y-%m-%d').split('-')

period1 = datetime.datetime(int(old_date_list[0]),int(old_date_list[1]),int(old_date_list[2]))
period2 = datetime.datetime(int(today_date_list[0]),int(today_date_list[1]),int(today_date_list[2]))

priceleaders_list = []
predicted_stocks_list = []
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

symbollist = ['GOOG','AAPL','AMZN','MSFT','^IXIC']

final = 0
while final == 0:
	dfcloses = None
	while dfcloses is None:
		try:
			dfcloses = web.DataReader('AA', 'yahoo', period1, period2)
		except:
			pass

	if dfcloses.index[len(dfcloses.index)-1].strftime('%Y-%m-%d') != today_date.strftime('%Y-%m-%d'):
		prev_date = prev_date + datetime.timedelta(days = 1)
		old_date_list = prev_date.strftime('%Y-%m-%d').split('-')
		period1 = datetime.datetime(int(old_date_list[0]),int(old_date_list[1]),int(old_date_list[2]))
		
		today_date = today_date + datetime.timedelta(days = 1)
		today_date_list = today_date.strftime('%Y-%m-%d').split('-')
		period2 = datetime.datetime(int(today_date_list[0]),int(today_date_list[1]),int(today_date_list[2]))
		session_today_date = session_today_date + datetime.timedelta(days = 1)
		session_prev_date = session_prev_date + datetime.timedelta(days=1)

	else:
		final = 1
		session_period_1 = str(session_prev_date.timestamp()).replace(".0","")
		session_period_2 = str(session_today_date.timestamp()).replace(".0","")
		today_date_list = today_date.strftime('%Y-%m-%d').split('-')
		period2 = datetime.datetime(int(today_date_list[0]),int(today_date_list[1]),int(today_date_list[2]))

for i in symbollist:
	http_query = "https://query1.finance.yahoo.com/v7/finance/download/"+i+"?period1="+ str(session_period_1) + "&period2="+str(session_period_2)+"&interval=1d&events=history&crumb="+crumb
	symbol_text = intermediate_session.get(http_query)
	symbol_text = symbol_text.text.split("\n")

	with open('intermediatefile.csv','w',newline = '') as csvfile:
		spamwriter = csv.writer(csvfile,delimiter = ',')
		for j in (symbol_text):
			j = j.split(",")
			if "" not in j:
				spamwriter.writerow(j)


	
	df = pd.read_csv('intermediatefile.csv', index_col = 'Date')


	priceleaders_list.append(df.ix[today_date.strftime('%Y-%m-%d')]['Close'])

for stock in currentstocks:
	stockname = re.findall(r"[A-Z]+",str(stock))[0]
	print(stockname)
	insert_data(stockname)

print("A buy call has been initiated on the following stocks")
for stock in predicted_stocks_list:
	print(stock)

with open('MainStocks.json', 'w') as outfile:
    json.dump(predicted_stocks_list, outfile)
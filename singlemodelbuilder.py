import csv
import os
import random
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
logistic = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
# clf = MLPClassifier(hidden_layer_sizes=(500,250,), learning_rate_init = 0.01, tol = 0.000001, max_iter = 200)
# lda = LDA()
svm = SVC()
from sklearn.metrics import accuracy_score


#Reading the respectice Stock file
Files = os.listdir("Y:\Work\StocksSelector\StocksDisp")
#Files = ['DKS-S&P.csv']
#Feature Indices refers to the features we want to include in the training data

# with open('.\StocksDisp\AAN-NYSE.csv') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter = ',')
# 	for row in spamreader:
# 		if row[0] == '21-09-2017':				#Adding the date '21-09-2017' to validate the model against
# 			temp_list = []
# 			for i in range(len(row)):
# 				if i in Feature_indices:
# 					temp_list.append(float(row[i]))
# 			validate_list.append(temp_list)


confusion_matrix_actual = []
confusion_matrix_predicted = []




def predict_return(filename):

	returns_dict = {}
	#random.seed(100)
	Feature_indices = [1,2,3,4,6,7,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
	validate_list = []
	stock_list = []
	count = 0
	target_list = []
	with open('.\StocksDisp\\'+filename) as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',')
		for row in spamreader:
			if '' not in row:
				if row[0] != 'Date':
					stock_list.append(row)
			if row[0] == '21-09-2017' or row[0] == '22-09-2017' or row[0] == '25-09-2017' or row[0] == '26-09-2017' or row[0] == '27-09-2017':
				target_list.append(row[8:16])
				temp_list = []
				for i in range(len(row)):
					if i in Feature_indices:
						temp_list.append(float(row[i]))
				validate_list.append(temp_list)


	#random.shuffle(stock_list)
	# print(stock_list[0:5])
	# input()
	TrainingDataIndex = int(0.8*len(stock_list))	#Training Data and Test Data are being split in the ratio 80:20
	TrainingData = stock_list[:TrainingDataIndex]	
	TestData = stock_list[TrainingDataIndex:]

	for index in range(8,16):
	# if returnweek == 1:
	# 	index = 8
	# elif returnweek == 2:
	# 	index = 9
	# elif returnweek == 3:
	# 	index = 10
	# elif returnweek == 4:
	# 	index = 11
	# elif returnweek == 5:
	# 	index = 12
	# elif returnweek == 6:
	# 	index = 13
	# elif returnweek == 7:
	# 	index = 14
	# elif returnweek == 8:
	# 	index = 15

		X_train = []
		Y_train = []


		for record in TrainingData:
				X = []
				if float(record[index]) <= 2:
					target_variable = 0
				elif float(record[index]) > 2 and float(record[index]) <= 5:
					target_variable = 1
				elif float(record[index]) > 5 and float(record[index]) <= 8:
					target_variable = 2
				elif float(record[index]) > 8 and float(record[index]) <= 12:
					target_variable = 3
				elif float(record[index]) > 12 and float(record[index]) <= 15:
					target_variable = 4
				elif float(record[index]) > 15 and float(record[index]) <= 20:
					target_variable = 5
				else:
					target_variable = 6

				X.append(float(record[1]))
				X.append(float(record[2]))
				X.append(float(record[3]))
				X.append(float(record[4]))
				X.append(float(record[6]))
				X.append(float(record[7]))
				X.append(float(record[40]))
				X.append(float(record[41]))
				X.append(float(record[42]))
				X.append(float(record[43]))
				X.append(float(record[44]))
				X.append(float(record[45]))
				X.append(float(record[46]))
				X.append(float(record[47]))
				X.append(float(record[48]))
				X.append(float(record[49]))
				X.append(float(record[50]))
				X.append(float(record[51]))
				X.append(float(record[52]))
				X.append(float(record[53]))
				X.append(float(record[54]))
				X.append(float(record[55]))
				X.append(float(record[56]))
				X.append(float(record[57]))
				X.append(float(record[58]))
				X.append(float(record[59]))
				X.append(float(record[60]))
				X.append(float(record[61]))
				X.append(float(record[62]))
				X.append(float(record[63]))
				X.append(float(record[64]))
				X.append(float(record[65]))
				X.append(float(record[66]))
				X.append(float(record[67]))
				X.append(float(record[68]))
				X.append(float(record[69]))
				X.append(float(record[70]))
				X.append(float(record[71]))
				X.append(float(record[72]))
				X.append(float(record[73]))
				X.append(float(record[74]))
				X.append(float(record[75]))
				X.append(float(record[76]))
				X.append(float(record[77]))
				X.append(float(record[78]))
				X.append(float(record[79]))
				X.append(float(record[80]))
				X.append(float(record[81]))
				X.append(float(record[82]))
				X.append(float(record[83]))
				X.append(float(record[84]))
				X.append(float(record[85]))
				X.append(float(record[86]))
				X.append(float(record[87]))
				X.append(float(record[88]))
				X.append(float(record[89]))
				X.append(float(record[90]))
				X.append(float(record[91]))
				X.append(float(record[92]))
				X.append(float(record[93]))
				X.append(float(record[94]))
				X.append(float(record[95]))
				X.append(float(record[96]))
				X.append(float(record[97]))
				X.append(float(record[98]))
				X.append(float(record[99]))
				X.append(float(record[100]))
				X_train.append(X)
				Y_train.append(target_variable)


		X_test = []
		Y_test = []
		for record in TestData:
				X = []
				if float(record[index]) <= 2:
					target_variable = 0
				elif float(record[index]) > 2 and float(record[index]) <= 5:
					target_variable = 1
				elif float(record[index]) > 5 and float(record[index]) <= 8:
					target_variable = 2
				elif float(record[index]) > 8 and float(record[index]) <= 12:
					target_variable = 3
				elif float(record[index]) > 12 and float(record[index]) <= 15:
					target_variable = 4
				elif float(record[index]) > 15 and float(record[index]) <= 20:
					target_variable = 5
				else:
					target_variable = 6

				X.append(float(record[1]))
				X.append(float(record[2]))
				X.append(float(record[3]))
				X.append(float(record[4]))
				X.append(float(record[6]))
				X.append(float(record[7]))
				X.append(float(record[40]))
				X.append(float(record[41]))
				X.append(float(record[42]))
				X.append(float(record[43]))
				X.append(float(record[44]))
				X.append(float(record[45]))
				X.append(float(record[46]))
				X.append(float(record[47]))
				X.append(float(record[48]))
				X.append(float(record[49]))
				X.append(float(record[50]))
				X.append(float(record[51]))
				X.append(float(record[52]))
				X.append(float(record[53]))
				X.append(float(record[54]))
				X.append(float(record[55]))
				X.append(float(record[56]))
				X.append(float(record[57]))
				X.append(float(record[58]))
				X.append(float(record[59]))
				X.append(float(record[60]))
				X.append(float(record[61]))
				X.append(float(record[62]))
				X.append(float(record[63]))
				X.append(float(record[64]))
				X.append(float(record[65]))
				X.append(float(record[66]))
				X.append(float(record[67]))
				X.append(float(record[68]))
				X.append(float(record[69]))
				X.append(float(record[70]))
				X.append(float(record[71]))
				X.append(float(record[72]))
				X.append(float(record[73]))
				X.append(float(record[74]))
				X.append(float(record[75]))
				X.append(float(record[76]))
				X.append(float(record[77]))
				X.append(float(record[78]))
				X.append(float(record[79]))
				X.append(float(record[80]))
				X.append(float(record[81]))
				X.append(float(record[82]))
				X.append(float(record[83]))
				X.append(float(record[84]))
				X.append(float(record[85]))
				X.append(float(record[86]))
				X.append(float(record[87]))
				X.append(float(record[88]))
				X.append(float(record[89]))
				X.append(float(record[90]))
				X.append(float(record[91]))
				X.append(float(record[92]))
				X.append(float(record[93]))
				X.append(float(record[94]))
				X.append(float(record[95]))
				X.append(float(record[96]))
				X.append(float(record[97]))
				X.append(float(record[98]))
				X.append(float(record[99]))
				X.append(float(record[100]))
				X_test.append(X)
				Y_test.append(target_variable)

		# scaler = MinMaxScaler()
		# scaler.fit(X_train)
		# X_train = scaler.transform(X_train)
		test = SelectKBest(score_func=chi2, k=50)
		fit = test.fit(X_train, Y_train)
		print(fit)
		input()
		
		pca = PCA()
		pca.fit(X_train)
		X_train = pca.fit_transform(X_train)

		logistic.fit(X_train,Y_train)
		svm.fit(X_train, Y_train)
		#clf.fit(X_train,Y_train)
		# lda.fit(X_train,Y_train)

		# clf_prediction = clf.predict(X_test)
		logistic_prediction = logistic.predict(X_test)
		svm_prediction = svm.predict(X_test)

		print(accuracy_score(Y_test,svm_prediction))
		# input()
		# print(index)

		answer_list = logistic.predict(validate_list)
		temp = []
		for i in range(len(target_list)):
			if index != 15:
				if float(target_list[i][index-8]) <= 2:
					target_variable = 0
				elif float(target_list[i][index-8]) > 2 and float(target_list[i][index-8]) <= 5:
					target_variable = 1
				elif float(target_list[i][index-8]) > 5 and float(target_list[i][index-8]) <= 8:
					target_variable = 2
				elif float(target_list[i][index-8]) > 8 and float(target_list[i][index-8]) <= 12:
					target_variable = 3
				elif float(target_list[i][index-8]) > 12 and float(target_list[i][index-8]) <= 15:
					target_variable = 4
				elif float(target_list[i][index-8]) > 15 and float(target_list[i][index-8]) <= 20:
					target_variable = 5
				else:
					target_variable = 6
				temp.append((answer_list[i],target_variable))
				confusion_matrix_predicted.append(answer_list[i])
				confusion_matrix_actual.append(target_variable)
			else:
				temp.append(answer_list[i])
		returns_dict['R'+str(index-7)] = temp

	return(returns_dict)


for file in Files:
	print(file)
	main_dict = predict_return(file)
	with open('LogisticResults.csv','a',newline = '') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow([file.replace('.csv','')])
		writer.writerow(['Date','R1(Predicted,Actual)','R2(Predicted,Actual)','R3(Predicted,Actual)','R4(Predicted,Actual)','R5(Predicted,Actual)','R6(Predicted,Actual)','R7(Predicted,Actual)','R8(Predicted,Actual)'])
		R1_list = main_dict['R1']
		R2_list = main_dict['R2']
		R3_list = main_dict['R3']
		R4_list = main_dict['R4']
		R5_list = main_dict['R5']
		R6_list = main_dict['R6']
		R7_list = main_dict['R7']
		R8_list = main_dict['R8']
		for i in range(5):
			if i == 0:
				writer.writerow(['21-09-2017',R1_list[i],R2_list[i],R3_list[i],R4_list[i],R5_list[i],R6_list[i],R7_list[i],R8_list[i]])
			if i == 1:
				writer.writerow(['22-09-2017',R1_list[i],R2_list[i],R3_list[i],R4_list[i],R5_list[i],R6_list[i],R7_list[i],R8_list[i]])
			if i == 2:
				writer.writerow(['25-09-2017',R1_list[i],R2_list[i],R3_list[i],R4_list[i],R5_list[i],R6_list[i],R7_list[i],R8_list[i]])
			if i == 3:
				writer.writerow(['26-09-2017',R1_list[i],R2_list[i],R3_list[i],R4_list[i],R5_list[i],R6_list[i],R7_list[i],R8_list[i]])
			if i == 4:
				writer.writerow(['27-09-2017',R1_list[i],R2_list[i],R3_list[i],R4_list[i],R5_list[i],R6_list[i],R7_list[i],R8_list[i]])

print(confusion_matrix(confusion_matrix_actual,confusion_matrix_predicted))
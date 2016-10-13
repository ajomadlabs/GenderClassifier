'''------------------------------------------------------------------
	A simple classifier which distinguishes between male and
	female based on the features and training the model using 
	DecisionTreeClassifier model
-------------------------------------------------------------------'''

'''------------------------------------------------------------------
	Importing tree classifier from scikit
-------------------------------------------------------------------'''

from sklearn import tree						#Tree is classifier which is imported from scikit

'''----------------------------------------------------------------'''

'''-------------------------------------------------------------------
	Defining The Dataset
-------------------------------------------------------------------'''

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],	#X denotes the features for distinguishing between male and female
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']		#Y denotes the corresponding label

'''-----------------------------------------------------------------'''

'''--------------------------------------------------------------------
	Creating the classifier and training the model
---------------------------------------------------------------------'''
    
classifier = tree.DecisionTreeClassifier()				#Creating an empty classifier which is DecisionTreeClassifier 
classifier = classifier.fit(X,Y)					#Training the model by giving the features and corresponding labels

'''------------------------------------------------------------------'''

'''---------------------------------------------------------------------
	Defining the test data set
---------------------------------------------------------------------'''

test_data = ['male']							#Defining the testdata

'''------------------------------------------------------------------'''

'''---------------------------------------------------------------------
	Defining the prediction of the system
---------------------------------------------------------------------'''

print(test_data)							#Printing the test_data
prediction = classifier.predict([[195,125,43]])				#Assining the prediction for the test data given to the system to a prediciton variable
print(prediction)							#Printing the prediction

'''------------------------------------------------------------------'''

'''---------------------------------------------------------------------
	Importing accuracy_scroe from scikit
---------------------------------------------------------------------'''

from sklearn.metrics import accuracy_score				#Importing the accuracy_score
print(accuracy_score(test_data,prediction))				#Printing the accuracy between the predicition and actual value

'''------------------------------------------------------------------'''

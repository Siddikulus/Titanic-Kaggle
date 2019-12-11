
#-------------------------------80% Accuracy using Random Forest Classifier (100 estimators).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

resultdf = pd.read_csv('../01. Titanic ML/Titanic Data/titanic/gender_submission.csv')
traindf = pd.read_csv('../01. Titanic ML/Titanic Data/titanic/train.csv')
testdf = pd.read_csv('../01. Titanic ML/Titanic Data/titanic/test.csv')
traindfcopy = [traindf, testdf]
# print(traindf.columns.values)
# print(traindf.info()) #Shows total present counts of rows of each feature (Check to see if missing values). Also shows
#                       #data type for conversion purposes.
# print('_'*100)
# print(testdf.info())
# print(sum(traindf['Survived'])) #People who survived out of 891.

# print(traindf.describe())#For numerical data
# print(traindf.describe(include = [np.object]))
# print(traindf.describe(include = 'all')) #Describe all columns, both categorical and numerical
# print(traindf.isna().sum())
# print(testdf.isna().sum())
# print(traindf.head())
# print(testdf.head())

#-----------------Handling missing data (khud kiya)
imp = SimpleImputer(missing_values = np.NaN, strategy = 'median')
#imputing 'Age' of traindf
imputer = imp.fit(traindf.iloc[:, 5:6])
traindf.iloc[:, 5:6] = imputer.transform(traindf.iloc[:, 5:6])
#imputing 'Embarked' of traindf
impcat = SimpleImputer(missing_values = np.NaN, strategy = 'most_frequent')
impemb = impcat.fit(traindf.iloc[:, 11:12])
traindf.iloc[:, 11:12] = impemb.transform(traindf.iloc[:, 11:12])
#imputing 'Age' of testdf
imputertest = imp.fit(testdf.iloc[:, 4:5])
testdf.iloc[:, 4:5] = imputertest.transform(testdf.iloc[:, 4:5])
#imputing 'Fare' of testdf
testdf.iloc[:, 8:9] = imp.fit_transform(testdf.iloc[:, 8:9])
#Removing 'Cabin' column from both train and test
traindf = traindf.drop(['Cabin'], axis = 1)
testdf = testdf.drop(['Cabin'], axis = 1)

# print(traindf.info())

#----------------Correlation Matrix
# corrmat = traindf.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(10,10))
# #plot heat map
# g=sns.heatmap(traindf[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()

#----------------Using Group-By to check impact on prediction.

# print(traindf[['Pclass', 'Survived']].groupby(['Pclass'],as_index = False).mean())

# print(traindf[['Age', 'Survived']].groupby(['Age'],as_index = False).mean()) #Since there are alot of values, we will
# distibute in different categories(bins)
# !!!!Creating multiple categories for Age!!!!
traindf['CategoryAge'] = pd.qcut(traindf['Age'], 6)
testdf['CategoryAge'] = pd.qcut(testdf['Age'], 6)
# print(traindf[['CategoryAge', 'Survived']].groupby(['CategoryAge'], as_index=False).mean())

# print(traindf[['Fare', 'Survived']].groupby(['Fare'], as_index=False). mean())#Since lots of values, distibute to bins.
# !!!!Creating multiple categories for Fare!!!!
traindf['CategoryFare'] = pd.qcut(traindf['Fare'], 4)
testdf['CategoryFare'] = pd.qcut(testdf['Fare'], 4)
# print(traindf[['CategoryFare', 'Survived']].groupby('CategoryFare').mean())

# print(traindf[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean()) #as_index gives a index column.
# print(traindf[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean()) #as_index gives a index column.
# print(traindf[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean()) #as_index gives a index column.

#!!!!Creating new feature Family Size!!!!
traindf['FamilySize'] = traindf['SibSp'] + traindf['Parch'] + 1
testdf['FamilySize'] = testdf['SibSp'] + traindf['Parch'] + 1 # (+1) cause khud ko bhi to family mein include krega
# print(traindf[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean()) #as_index gives a index column.

#!!!!Creating new feature to check if member is alone or not!!!!
traindf['Isalone'] = 0
traindf.loc[traindf['FamilySize'] == 1, 'Isalone'] = 1
testdf['Isalone'] = 0
testdf.loc[testdf['FamilySize'] == 1, 'Isalone'] = 1
# print(traindf[['Isalone', 'Survived']].groupby(['Isalone'], as_index = False).mean()) #as_index gives a index column.
# print (traindf[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


#---------------------Splitting Title from Name
traindf['Title'] = traindf['Name'].str.split(',', expand = True)[1].str.split('.', expand=True)[0]
testdf['Title'] = testdf['Name'].str.split(',', expand = True)[1].str.split('.', expand=True)[0]

# print(pd.crosstab(traindf['Title'], traindf['Sex']))
traindf['Title'].replace(['Lady', 'the Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace = True, regex = True)
traindf['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss', inplace = True, regex = True)

testdf['Title'].replace(['Lady', 'the Countess','Capt', 'Col',
 	'Dona', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Don'], 'Rare', inplace = True, regex = True)
testdf['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss', inplace = True, regex = True)
# print(traindf[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
# print(testdf['Title'])

# ------------------------Cleaning Data
traindf.loc[traindf['Age']<=19, 'Age'] = 0
traindf.loc[(traindf['Age']>19) & (traindf['Age']<=25), 'Age'] = 1
traindf.loc[(traindf['Age']>25) & (traindf['Age']<=28), 'Age'] = 2
traindf.loc[(traindf['Age']>28) & (traindf['Age']<=31), 'Age'] = 3
traindf.loc[(traindf['Age']>31) & (traindf['Age']<=40.5), 'Age'] = 4
traindf.loc[(traindf['Age']>40.5) & (traindf['Age']<=80), 'Age'] = 5

testdf.loc[testdf['Age']<=19, 'Age'] = 0
testdf.loc[(testdf['Age']>19) & (testdf['Age']<=25), 'Age'] = 1
testdf.loc[(testdf['Age']>25) & (testdf['Age']<=28), 'Age'] = 2
testdf.loc[(testdf['Age']>28) & (testdf['Age']<=31), 'Age'] = 3
testdf.loc[(testdf['Age']>31) & (testdf['Age']<=40.5), 'Age'] = 4
testdf.loc[(testdf['Age']>40.5) & (testdf['Age']<=80), 'Age'] = 5

traindf.loc[traindf['Fare']<=7.91, 'Fare'] = 0
traindf.loc[(traindf['Fare']>7.91) & (traindf['Fare']<=14.454), 'Fare'] = 1
traindf.loc[(traindf['Fare']>14.454) & (traindf['Fare']<=31), 'Fare'] = 2
traindf.loc[(traindf['Fare']>31) & (traindf['Fare']<=512.329), 'Fare'] = 3

testdf.loc[testdf['Fare']<=7.91, 'Fare'] = 0
testdf.loc[(testdf['Fare']>7.91) & (testdf['Fare']<=14.454), 'Fare'] = 1
testdf.loc[(testdf['Fare']>14.454) & (testdf['Fare']<=31), 'Fare'] = 2
testdf.loc[(testdf['Fare']>31) & (testdf['Fare']<=512.329), 'Fare'] = 3

traindf['Sex'] = traindf['Sex'].map({'female':0, 'male':1}).astype(int) #Encoding Sex Column
testdf['Sex'] = testdf['Sex'].map({'female':0, 'male':1}).astype(int)


traindf.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'CategoryAge', 'CategoryFare','FamilySize'], axis = 1, inplace = True)


#!!!!!!!!!!Dummying traindf(OneHotEncoder not working efficiently for multiple columns.
traindf = pd.get_dummies(traindf, columns = ['Embarked', 'Title'], drop_first=True)

testdf.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'CategoryAge', 'CategoryFare', 'FamilySize'], axis = 1, inplace = True)
#!!!!!!!!!!Dummying testdf
testdf = pd.get_dummies(testdf, columns = ['Embarked', 'Title'], drop_first=True)
# print(traindf)
# print(testdf)

#------------------------------------------Splitting Data
xtrain = traindf.iloc[:, 1:]
ytrain = traindf.iloc[:, 0]
# print(xtrain)
# print(ytrain)

#-----------------------------------------------Implementing ML models--------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# #Logistic Regression
# lr = LogisticRegression()
# lr.fit(xtrain, ytrain)
# a = lr.predict(testdf)
# resultdf = resultdf.iloc[:, 1:]
# b = np.array(resultdf).flatten()
# count = 0
# count = sum([count+1 for i, j in zip(a,b) if i!=j])
# print(count)

#SVC
svc = SVC()
svc.fit(xtrain,ytrain)
a1 = svc.predict(testdf)
# resultdf = resultdf.iloc[:, 1:]
# b1 = np.array(resultdf).flatten()
# count = 0
# count = sum([count+1 for i, j in zip(a1,b1) if i!=j])
# print(count)
# a1 = pd.DataFrame(a1)
# a1['PassengerId'] = a1.index + 892
# a1.columns = ['Survived', 'PassengerID']
# a1 = a1[['PassengerID', 'Survived']]
# export_csv = a1.to_csv(r'/home/sidharth/Python/ML Algos/Kaggle/01. Titanic ML/MySubmission.csv', header = True)

# #RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(xtrain,ytrain)
a1 = rfc.predict(testdf)
a1 = pd.DataFrame(a1)
a1['PassengerId'] = a1.index + 892
a1.columns = ['Survived', 'PassengerID']
a1 = a1[['PassengerID', 'Survived']]
export_csv = a1.to_csv(r'/home/sidharth/Python/ML Algos/Kaggle/01. Titanic ML/MySubmissionRFC.csv', header = True)
# resultdf = resultdf.iloc[:, 1:]
# b2 = np.array(resultdf).flatten()
# count = 0
# count = sum([count+1 for i, j in zip(a2,b2) if i!=j])
# print(count)
#
# #DescisionTreeClassifier
# dtc = DecisionTreeClassifier()
# dtc.fit(xtrain,ytrain)
# a2 = dtc.predict(testdf)
# resultdf = resultdf.iloc[:, 1:]
# b2 = np.array(resultdf).flatten()
# count = 0
# count = sum([count+1 for i, j in zip(a2,b2) if i!=j])
# print(count)











# #############################################################################################################################
# #############################################################################################################################
# #############################################################################################################################
# #############################################################################################################################
# #############################################################################################################################
#``````````````````````Handling Missing Values``````````````````````````
# for data in combinedf:
#     data['Age'].fillna(data['Age'].median(), inplace = True)
#     testdf['Fare'].fillna(testdf['Fare'].median(),inplace = True)
#     data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
# traindfcopy.drop(['Cabin', 'PassengerId', 'Ticket'],axis = 1, inplace = True) #Because lots of missing values
#
#
# # print(testdf.describe(include = 'all'))
# # print(traindfcopy.describe(include = 'all'))
# # print(traindfcopy.isna().sum())

#``````````````````````Feature Engineering``````````````````````````````
# for data in combinedf:
#     data['family size'] = data['SibSp'] + data['Parch'] + 1
#     data['IsAlone'] = 1
#     data['IsAlone'].loc[data['family size'] > 1] = 0


#``````````````````````Splitting Title from Name`````````````````````````
    # data['Title'] = data['Name'].str.split(',', expand = True)[1].str.split('.', expand=True)[0]

# traindf.groupby(by = 'Survived').mean()
# traindf.groupby(by = 'Survived').mean().plot(kind = 'bar')
# plt.legend(loc = 'upper left')

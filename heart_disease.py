# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:03:16 2019

@author: Admin
"""
#https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

disease = pd.read_csv('E:\\Data science\\DSothers\\projects\\heart_disease\\dataset.csv')

"""age: The person's age in years
sex: The person's sex (1 = male, 0 = female)
cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
chol: The person's cholesterol measurement in mg/dl
fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
thalach: The person's maximum heart rate achieved
exang: Exercise induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
ca: The number of major vessels (0-3)
thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
target: Heart disease (0 = no, 1 = yes)"""

disease.shape
disease.info()
disease.isnull().sum()
disease.describe()

disease.head()
disease['target'].value_counts()    #data is balaced

# see data is overlap or not(nonlinear)
sns.pairplot(disease , hue='target')    #data is overlapped apply KNN and Random_forest 

#correlation
corr = disease.corr()
col = corr.index
plt.figure(figsize=(20,20))
g = sns.heatmap(disease[col].corr(),annot =True , cmap="RdYlGn")
#cheast_pain , thalach(heart_rate) , slope high correlation with target (dependent) varible


#Let's change the column names to be a bit clearer
disease.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

# to above describtion wies we change our varible to catogorical terms
disease['sex'][disease['sex'] == 0] = 'female'
disease['sex'][disease['sex'] == 1] = 'male'

disease['chest_pain_type'][disease['chest_pain_type'] == 1] = 'typical angina'
disease['chest_pain_type'][disease['chest_pain_type'] == 2] = 'atypical angina'
disease['chest_pain_type'][disease['chest_pain_type'] == 3] = 'non-anginal pain'
disease['chest_pain_type'][disease['chest_pain_type'] == 4] = 'asymptomatic'

disease['fasting_blood_sugar'][disease['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
disease['fasting_blood_sugar'][disease['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

disease['rest_ecg'][disease['rest_ecg'] == 0] = 'normal'
disease['rest_ecg'][disease['rest_ecg'] == 1] = 'ST-T wave abnormality'
disease['rest_ecg'][disease['rest_ecg'] == 2] = 'left ventricular hypertrophy'

disease['exercise_induced_angina'][disease['exercise_induced_angina'] == 0] = 'no'
disease['exercise_induced_angina'][disease['exercise_induced_angina'] == 1] = 'yes'

disease['st_slope'][disease['st_slope'] == 1] = 'upsloping'
disease['st_slope'][disease['st_slope'] == 2] = 'flat'
disease['st_slope'][disease['st_slope'] == 3] = 'downsloping'

disease['thalassemia'][disease['thalassemia'] == 1] = 'normal'
disease['thalassemia'][disease['thalassemia'] == 2] = 'fixed defect'
disease['thalassemia'][disease['thalassemia'] == 3] = 'reversable defect'


#Getting bar-plot for catogorical varible
# 0 not Exited , 1 for Exited from bank
disease.columns
sns.countplot(x="target", data = disease ,palette="hls")   #balanced data , heart_disease = 1 , 0 for other
sns.countplot(x="sex", data=disease)                 #female 100 and male 200 
sns.countplot(x="chest_pain_type", data=disease)                 #cheast pain
sns.countplot(x="st_slope", data=disease)                #flat and downslopping is more

pd.crosstab(disease.target, disease.st_slope).plot(kind='bar')   #most of heart disease patient has flat slope
pd.crosstab(disease.target , disease.chest_pain_type).plot(kind='bar')    #most of non-anginal pain patient has heart disease
pd.crosstab(disease.target , disease.sex).plot(kind='bar')      #both male and female heart disease problem

#Data visulization using boxplot of countineous varibles wrt to each category
sns.boxplot(x='target', y='age', data= disease, palette='hls')  #above age of 45 , the patient has heart disease problem
sns.boxplot(x="target",y="max_heart_rate_achieved", data= disease,palette = "hls")   #150 to 170 heart_rate patient has heart disease problem
sns.boxplot(x="target",y="resting_blood_pressure", data= disease,palette = "hls")

#Preprocessing : To make dummies 
disease = pd.get_dummies(disease, drop_first=True)
disease.head()

from sklearn.preprocessing import StandardScaler  #scale down your 
standardScaler = StandardScaler()
disease.columns
columns_to_scale = ['age', 'resting_blood_pressure', 'cholesterol','max_heart_rate_achieved', 'st_depression']
disease[columns_to_scale] = standardScaler.fit_transform(disease[columns_to_scale])

X = disease.drop('target', axis=1)
Y = disease['target']

#-------------------------------LR-----------------------------------
import statsmodels.api as sm
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())  #0,2,3,10,11,12,13,14,15,16

#another method of feature selections
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X, Y)
# display the relative importance of each attribute
print(model.feature_importances_)   

X_new = X.iloc[:,[1,4,5,6,7,8,9,17,18]]
logit_model=sm.Logit(Y,X_new)
result=logit_model.fit()
print(result.summary2())  #41% 
#--------------------------------------------------------------------

#-------------------------------KNN-----------------------------------
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNC(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,Y,cv=10)   #cv=Experiments
    knn_scores.append(score.mean())

plt.plot([k for k in range(1,21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')

knn_classifier = KNC(n_neighbors = 7)  #select 7 is best
#cross validation score : train and test data gives differnt accuracy , to do cross_val_score it work like iteration base train and test , he do all combination experimetns
#below cv=10 means , 10 experiments like( train and test 9:1 out of 10, to 10 iternation till the end), finally cal avarage of 10 experiments results
score=cross_val_score(knn_classifier,X,Y,cv=10)   #its also help to which type alogorithm select for this dataset
score.mean()      #0.8174

#-------------------------------RF-----------------------------------
# used ensembled technique to build combine multiple model (Decision Tree), bagging , finally prediction depends upon multiple DT o/p majority
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier= RandomForestClassifier(n_estimators=10 , max_depth=5)
score=cross_val_score(randomforest_classifier,X,Y,cv=10)
score.mean()  #0.8007
#--------------------------------------------------------------------

#-------------we select KNN-----------------
# Training and Testing data: stratified sampling 
# beacause of y has catogorical to make sense of do stratified sampling : to equal propotional of train and test data
from sklearn.model_selection import StratifiedShuffleSplit as sss
split = sss(n_splits = 5, test_size = 0.2 , random_state = 42)
for train_index , test_index in split.split(disease, disease['target']):
    d_train = disease.loc[train_index]
    d_test = disease.loc[test_index]

d_train.target.value_counts()
d_test.target.value_counts()

X = d_train.drop('target', axis=1)
Y = d_train['target']

X_test = d_test.drop('target', axis=1)
Y_test = d_test['target']

# build with train and test data calculate accuracy
acc= []
#running KNN model to 3 to 21 neighbors
for i in range(1,21,2):
    neigh = KNC(n_neighbors = i)
    neigh.fit(X,Y)
    train_acc = np.mean(neigh.predict(X) == Y)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])
    
plt.plot(np.arange(1,21,2),[i[0] for i in acc], "bo-")
plt.plot(np.arange(1,21,2), [i[1] for i in acc], "ro-")
plt.legend(["train","test"])
# we select our k =7

near5 = KNC(n_neighbors = 7)
near5.fit(X, Y)
train_acc5 = np.mean(near5.predict(X) == Y)
train_acc5   #86.7768
test_acc5 = np.mean(near5.predict(X_test)== Y_test)
test_acc5  #86.8852

X_test['pred'] = near5.predict(X_test)
from sklearn.metrics import confusion_matrix, recall_score, precision_score , f1_score  , classification_report , accuracy_score
confusion_matrix(Y_test , X_test['pred'])
#Y_test.value_counts()

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(Y_test , X_test['pred'])
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
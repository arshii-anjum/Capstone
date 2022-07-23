#!/usr/bin/env python
# coding: utf-8

# # WATER POTABILITY ANALYSIS

# In our project, we will check the potability of the water based on the ph level, hardness level, solids contents, chloramines contents, sulphate contents, its conductivity, organic carbon, trihalomethanes and turbidity.This analysis will help us determine if the water is fit for human consumption or not. 
# 

# #### Loading Dataset and Libraries

# In[327]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
import missingno as msno
import pandas_profiling as pp
from IPython.display import IFrame
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor

import warnings
warnings.filterwarnings("ignore")


# In[328]:


#Load Dataset
df = pd.read_csv('./water_potability.csv')
df.head()


# Potability of the water in our dataset is based on the ph level, hardness level, solids contents, chloramines contents, sulphate contents, its conductivity, organic carbon, trihalomethanes and turbidity.
# 

# In[329]:


#Profile Report
DataReport = pp.ProfileReport(df)
DataReport.to_file('WaterReport.html')
display(IFrame('WaterReport.html', width=900, height=350))


# ### DATA PRE-PROCESSING

# By WHO's standard, the TDS should be between 50 to 300 ppm but the dataset contains value that goes unto 50,000 ppm for TDS. As the dataset is synthetically generated, we assume the values were generated with incorrect decimal placement. Hence we will be shifting the decimal to correct it for our analysis. 

# In[330]:


#Shift the decimal in solids by dividing the value by 100
df ['Solids'] = df['Solids']/100


# In[331]:


#Load Dataset to check the processed Data
df.head()


# # EXPLORATORY DATA ANALYSIS 

# In[332]:


#Overview of Dataset Characteristics
df.info()


# In[333]:


#Statistics of the dataset
df.describe()


# In[334]:


#Statistics of the dataset
df.median()


# In[335]:


#Potalibility grouped by feature mean
df.groupby('Potability').mean()


# In[336]:


#Distribution of potalibility with Pie Chart
colors=['#1186AB', '#0DB75E']
labels=['Not Potable','Potable']
pieplot = df.groupby('Potability').size()
pieplot.plot(kind='pie', colors=colors, subplots=True,shadow=True, figsize=(7, 7), fontsize=9, autopct='%1.1f%%')
plt.title("Potability Values Distribution")
plt.legend(labels)
plt.ylabel("")


# ### Univariate Statistics

# In[337]:


#Histogram of numeric variables
num_bins = 10

df.hist(bins=num_bins, figsize=(20,15))
plt.savefig("water_histogram_plots")
plt.show()


# In[338]:


plt.figure(figsize=(12,10))
for i, column in enumerate(df.columns[:9]):
    plt.subplot(3,3,i+1)
    sns.histplot(df[column],kde=True,alpha=0.3, bins=10, color='green',common_norm=False)


# In[339]:


#Skewness
df.skew().sort_values(ascending = False)


# Most features are normal distribution. Values between 0.5 to -0.5 will be considered as the normal distribution. 
# Though Solids has value slightly above 0.5, we still consider it doesn't have skewness.

# ### Bivariate Statistics

# In[340]:


sns.countplot(data = df, x = 'Potability')


# In[341]:


#Pairplot to check for clusters
sns.pairplot(df, hue ='Potability')


# A pairplot showing a bivariable pairwise relationships between potalibility and other features. 
# * Blue = 0 (Non potable)
# * Orange = 1 (Potable)

# In[342]:


#Potability and Ph
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'ph', y = 'Potability', orient = 'h').set(title = 'Ph distribution');


# In[343]:


#Potability and hardness distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Hardness', y = 'Potability', orient = 'h').set(title = 'Hardness distribution');


# In[344]:


#Potability and  Solids distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Solids', y = 'Potability', orient = 'h').set(title = 'Solids distribution');


# In[345]:


#Potability and Chloramines distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Chloramines', y = 'Potability', orient = 'h').set(title = 'Chloramines distribution');


# In[346]:


#Potability and Sulfate distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Sulfate', y = 'Potability', orient = 'h').set(title = 'Sulfate distribution');


# In[347]:


#Potability and Conductivity distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Conductivity', y = 'Potability', orient = 'h').set(title = 'Conductivity distribution');


# In[348]:


#Potability and Organic_carbon distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Organic_carbon', y = 'Potability', orient = 'h').set(title = 'Organic_carbon distribution');


# In[349]:


#Potability and Trihalomethanes distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Trihalomethanes', y = 'Potability', orient = 'h').set(title = 'Trihalomethanes distribution');


# In[350]:


#Potability and Turbidity distribution
fig,ax  = plt.subplots(figsize = (12,5))
sns.boxplot(data =df, x = 'Turbidity', y = 'Potability', orient = 'h').set(title = 'Turbidity distribution');


# ### Multivariate Statistics

# In[351]:


# Correlation heatmap among features

fig,ax = plt.subplots(figsize = (15,7))
sns.heatmap(df.corr(),annot = True)


# From the above map, we can see that there is minimal to no correlation between any of the features. 
# 

# In[352]:


#Correlation of features with Potability
plt.figure(figsize=(7, 10))
heatmap = sns.heatmap(df.corr()[['Potability']].sort_values(by='Potability', ascending=False),annot=True, cmap='GnBu_r')
plt.title('Descending Correlation with Potability',pad=20, fontsize=16)


# There is no strong or significant correlation of the features with portability. 
# 

# ## HANDLING OUTLIERS

# In[353]:


df1=df


# In[354]:


#outliers in the data.

i=1
plt.figure(figsize=(15,25))
for feature in df.columns:
    plt.subplot(6,3,i)
    sns.boxplot(y=df[feature])
    i+=1


# From the boxplot, we can see that there are many outliers in the data. Hence, we will remove the outliers to further process the data. 
# 

# In[355]:


#Removing outliers

cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'] # one or more

Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Old Shape: ", df1.shape)
print("New Shape: ", df.shape)


# After removing the outliers, the dimensions of our data has 2951 rows and 10 columns. 

# In[356]:


#Boxplot after removing the outliers.

i=1
plt.figure(figsize=(15,25))
for feature in df.columns:
    plt.subplot(6,3,i)
    sns.boxplot(y=df[feature])
    i+=1


# ## DUPLICATE VALUES

# In[357]:


df.duplicated()


# There are no duplicate values that have been detected in our data.

# ## NULL VALUES - PREPROCESSING

# In[358]:


#Summary of Null Values
df.isnull().sum()


# In[359]:


#bar plot to show counts of non-null values
msno.bar(df, figsize = (16,5),color = "#0F9555")
plt.show()


# In[360]:


# Get the number and percentage of missing data points for 3 columns affected
null=pd.DataFrame(df.isnull().sum(),columns=["Null Values"])
null["% Missing Values"]=(df.isna().sum()/len(df)*100)
null = null[null["% Missing Values"] > 0]
null.style.background_gradient(cmap='viridis',low =0.5,high=0.2) 


# The missing values are columns - 
# 
# * PH - 14.98%                 |   7.0367 (median) 7.0737 (mean)
# * Sulphate - 23.84%           |   331.8381 (median) 332.5670 (mean)
# * Trihalomethanes - 4.94%     |   66.6782 (median) 66.5397 (mean)
# 
# As our dataset is small, it might not be a good idea to drop all the missing value columns. Also, We see that the difference between mean and median values is small. Hence, we can use the overall median of the feature to impute values and fill the missing data.
# 

# #### FILL THE GAP IN DATA

# In[361]:


#Replacing the missing values with median
df['ph'].fillna(value=df['ph'].median(), inplace=True)
df['Sulfate'].fillna(value=df['Sulfate'].median(), inplace=True)
df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median(), inplace=True)


# In[362]:


#Checking the value count after filling gap
df.info()


# In[363]:


#Summary of null values after filling the gaps
df.isnull().sum()


# #### EDA after Cleansing of Data

# In[364]:


#Skewness 

plt.style.use('seaborn-dark')
colors=['#00a8e8', '#00afb9',  '#48bfe3', '#006e90', '#20a4f3', '#00b4d8', '#0466c8', '#20a4f3', '#00008B','#1E90FF']
i=0
while i<10:
    for col in df.columns:
        plt.figure(figsize=(6,4))
        sns.distplot(df[col],color=colors[i])
        plt.title(f'Distribution plot for {col}')
        plt.xlabel(f'Skewness = {round(df[col].skew(),3)}',fontsize=14)
        i+=1
        plt.show()


# # MODELLING

# The first step is to scale the data.This is important because scaling can ensure that one factor will not impact the model just because of their large magnitude.

# In[365]:


#Scale the data, split the independent and dependent variable

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = df.drop('Potability', axis =1)
y = df['Potability']
features = X.columns
X[features] = sc.fit_transform(X[features])
X


# In[366]:


# import train test split
from sklearn.model_selection import train_test_split
# assign 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[367]:


#Shape of train data
X_train.shape


# In[368]:


#Shape of test data
X_test.shape


# ### 1. Decision Tree

# In[369]:


# create the model
DeTree = DecisionTreeClassifier(max_depth = 4, random_state = 42, min_samples_leaf = 1, criterion ='entropy')
# model training
DeTree.fit(X_train, y_train)
# prediction
DeTree_pred = DeTree.predict(X_test)
# accuracy
DeTree_acc = accuracy_score(y_test, DeTree_pred)
# precision
DeTree_prec = precision_score(y_test, DeTree_pred)


# In[370]:


print("The accuracy for Decision Tree is", DeTree_acc)
print("The classification report using Decision Tree is:")
print(classification_report(y_test, DeTree_pred))


# In[371]:


# let's plot confusion matrix
DeTree_cm = confusion_matrix(y_test, DeTree_pred)
sns.heatmap(DeTree_cm/np.sum(DeTree_cm), annot = True, fmt = '0.2%', cmap = 'Blues')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Decision Tree')
plt.savefig('Decision Tree')


# In[372]:


#Confusion Matrix for Decision Tree
DeTree_cm = confusion_matrix(y_test, DeTree_pred)
sns.heatmap(DeTree_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Decision Tree')
plt.savefig('Decision Tree')


# ### 2. Random Forest

# In[373]:


# create the model
RmTree = RandomForestClassifier(n_estimators =100,min_samples_leaf =2, random_state = 42)
# model training
RmTree.fit(X_train, y_train)
# prediction
RmTree_pred = RmTree.predict(X_test)
# accuracy
RmTree_acc = accuracy_score(y_test, RmTree_pred)
# precision
RmTree_prec = precision_score(y_test, RmTree_pred)


# In[374]:


print("The accuracy for Random Forest is", RmTree_acc)
print("The classification report using Random Forest is:")
print(classification_report(y_test, RmTree_pred))


# In[375]:


# let's plot confusion matrix
RmTree_cm = confusion_matrix(y_test, RmTree_pred)
#RmTree_cm
sns.heatmap(RmTree_cm/np.sum(RmTree_cm), annot = True, fmt = '0.2%', cmap = 'Blues')


# In[376]:


#Confusion Matrix for Random Forest
DeTree_cm = confusion_matrix(y_test, RmTree_pred)
sns.heatmap(RmTree_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('Random Forest')


# ### 3. Logistic Regression

# In[450]:


# create the model
LogReg = LogisticRegression(random_state = 42, class_weight='balanced')
# model training
LogReg.fit(X_train, y_train)
# prediction
LogReg_pred = LogReg.predict(X_test)
# accuracy
LogReg_acc = accuracy_score(y_test, LogReg_pred)
# precision
LogReg_prec = precision_score(y_test, LogReg_pred)


# In[451]:


print("The accuracy for Logistic Regression is", LogReg_acc)
print("The classification report using Logistic Regression is:")
print(classification_report(y_test, LogReg_pred))


# In[452]:


# let's plot confusion matrix
LogReg_cm = confusion_matrix(y_test, LogReg_pred)
sns.heatmap(LogReg_cm/np.sum(LogReg_cm), annot = True, fmt = '0.2%', cmap = 'Blues')


# In[453]:


#Confusion Matrix for Logistic Regression
DeTree_cm = confusion_matrix(y_test, LogReg_pred)
sns.heatmap(LogReg_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('Logistic Regression')


# ### 4. XGBoost

# In[454]:


# create the model
XGB = XGBClassifier(max_depth= 8, n_estimators= 250, random_state= 0,  learning_rate= 0.03, n_jobs=5)
# model training
XGB.fit(X_train, y_train)
# prediction
XGB_pred = XGB.predict(X_test)
# accuracy
XGB_acc = accuracy_score(y_test, XGB_pred)
# precision
XGB_prec = precision_score(y_test, XGB_pred)


# In[455]:


print("The accuracy for XGBoost is", XGB_acc)
print("The classification report using XGBoost is:", XGB_acc)
print(classification_report(y_test, XGB_pred))


# In[456]:


# let's plot confusion matrix
XGB_cm = confusion_matrix(y_test, XGB_pred)
sns.heatmap(XGB_cm/np.sum(XGB_cm), annot = True, fmt = '0.2%', cmap = 'Blues')


# In[457]:


#Confusion Matrix for XGB
DeTree_cm = confusion_matrix(y_test, XGB_pred)
sns.heatmap(XGB_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('XGB')
plt.savefig('XGB')


# ### 5. KNeighbors Classifier

# In[476]:


# create the model
KNN = KNeighborsClassifier(n_neighbors = 8, leaf_size =20)
# model training
KNN.fit(X_train, y_train)
# prediction
KNN_pred = KNN.predict(X_test)
# accuracy
KNN_acc = accuracy_score(y_test, KNN_pred)
# precision
KNN_prec = precision_score(y_test, KNN_pred)


# In[477]:


print("The accuracy for KNeighbors is", KNN_acc)
print("The classification report using KNeighbors is:", KNN_acc)
print(classification_report(y_test, KNN_pred))


# In[478]:


# let's plot confusion matrix
KNN_cm = confusion_matrix(y_test, KNN_pred)
sns.heatmap(KNN_cm/np.sum(KNN_cm), annot = True, fmt = '0.2%', cmap = 'Blues')


# In[479]:


#Confusion Matrix for KNN
DeTree_cm = confusion_matrix(y_test, KNN_pred)
sns.heatmap(KNN_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('KNN')
plt.savefig('KNN')


# ### 6. AdaBoost Classifier

# In[480]:


# create the model
AdaBoost = AdaBoostClassifier(learning_rate = 0.08, n_estimators = 200, random_state = 42)
# model training
AdaBoost.fit(X_train, y_train)
# prediction
AdaBoost_pred = AdaBoost.predict(X_test)
# accuracy
AdaBoost_acc = accuracy_score(y_test, AdaBoost_pred)
# precision
AdaBoost_prec = precision_score(y_test, AdaBoost_pred)


# In[481]:


print("The accuracy for AdaBoost is", AdaBoost_acc)
print("The classification report using AdaBoost is:", AdaBoost_acc)
print(classification_report(y_test, AdaBoost_pred))


# In[482]:


# let's plot confusion matrix
AdaBoost_cm = confusion_matrix(y_test, AdaBoost_pred)
sns.heatmap(AdaBoost_cm/np.sum(AdaBoost_cm), annot = True, fmt = '0.2%', cmap = 'Blues')


# In[483]:


#Confusion Matrix for AdaBoost
AdaBoost_cm = confusion_matrix(y_test, AdaBoost_pred)
sns.heatmap(AdaBoost_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('AdaBoost')
plt.savefig('AdaBoost')


# ### 7. SVM

# In[484]:


from sklearn.calibration import CalibratedClassifierCV
SVM = SVC()
clf = CalibratedClassifierCV(SVM)
clf.fit(X_train, y_train)
CalibratedClassifierCV(base_estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
                       cv=3, method='sigmoid')
clf.predict_proba(X_test)
    


# In[485]:


# create the model
SVM = SVC(kernel ='rbf', random_state = 42)
# model training
SVM.fit(X_train, y_train)
# prediction
SVM_pred = SVM.predict(X_test)
# accuracy
SVM_acc = accuracy_score(y_test, SVM_pred)
# precision
SVM_prec = precision_score(y_test, SVM_pred)


# In[486]:


print("The accuracy for SVM is", SVM_acc)
print("The classification report using SVM is:", SVM_acc)
print(classification_report(y_test, SVM_pred))


# In[487]:


# let's plot confusion matrix
SVM_cm = confusion_matrix(y_test, SVM_pred)
sns.heatmap(SVM_cm/np.sum(SVM_cm), annot = True, fmt = '0.2%', cmap = 'Blues')


# In[488]:


#Confusion Matrix for SVM
SVM_cm = confusion_matrix(y_test, SVM_pred)
sns.heatmap(SVM_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM')
plt.savefig('SVM')


# # SUMMARY

# In[489]:


models = pd.DataFrame({
    'Model':['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'KNeighbours', 'SVM', 'AdaBoost'],
    'Accuracy' :[LogReg_acc, DeTree_acc, RmTree_acc, XGB_acc, KNN_acc, SVM_acc, AdaBoost_acc]
})
models.sort_values(by='Accuracy', ascending=False)


# In[490]:


models1 = pd.DataFrame({
    'Model':['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'KNeighbours', 'SVM', 'AdaBoost'],
    'Precision' :[LogReg_prec, DeTree_prec, RmTree_prec, XGB_prec, KNN_prec, SVM_prec, AdaBoost_prec]
})
models1.sort_values(by='Precision', ascending=False)


# In[491]:


plt.figure(figsize=(10,5))
sns.barplot(x='Model', y='Accuracy', data = models, 
            order = models.sort_values("Accuracy").Model,
           palette = 'Blues_d')


# In[492]:


plt.figure(figsize=(10,5))
sns.barplot(x='Model', y='Precision', data = models1, 
            order = models1.sort_values("Precision").Model,
           palette = 'Blues_d')


# In[493]:


#set up plotting area
plt.figure(0).clf()
plt.figure(figsize=(12,8))

#fit logistic regression model and plot ROC curve
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

#fit AdaBoostClassifier model and plot ROC curve
model = AdaBoostClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="AdaBoostClassifier, AUC="+str(auc))

#fit DecisionTreeClassifier model and plot ROC curve
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="DecisionTreeClassifier, AUC="+str(auc))


#fit KNeighborsClassifier model and plot ROC curve
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="KNeighborsClassifier, AUC="+str(auc))

#fit XGBClassifier model and plot ROC curve
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="XGBClassifier, AUC="+str(auc))

#fit Random Forest model and plot ROC curve
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="RandomForestClassifier, AUC="+str(auc))

#fit SVC model and plot ROC curve
model = SVC(probability=True)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="SVM, AUC="+str(auc))

#add legend
plt.legend()


# In[ ]:





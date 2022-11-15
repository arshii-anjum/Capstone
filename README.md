# WATER POTABILITY ANALYSIS
#### Readme file

## ABOUT PROJECT 
The goal of this project is to perform exploratory data analytics and machine learning prediction analytics on a variety of water quality indicators, such as pH, hardness, carbon, solids etc. The aim of this study is the prediction of water potability using machine learning algorithms including Logistic regression, SVM, Decision tree, Adaboost classifier, K-neighbors, XGB, and random forest and finally using the most accurate model to use for future predictions. The significant sources of water contamination can be determined using various factors and the findings may serve as useful predictors for evaluating if the water quality is fit for human consumption. This analysis will provide us the following outcomes – 
	• Predict whether the water quality is feasible for human consumption.
	• Anticipate the relevant measures to be taken if the water is contaminated.
	• Find features that contribute the most for water to be potable. 


## GETTING STARTED 
#### Prerequisites : 
The code can be run on windows using jupyter notebook, loaded from the anaconda navigator. 
            Version tested on : 
            Anaconda Navigator 2.1.1
            JupyterLab 3.2.1
	    
#### Dataset : 
The dataset contains water quality metrics for 3276 different water bodies and consists of 10 columns. Each row defines the quality of water. The features of the data have been explained in the table below. It has been sourced from kaggle.com. 

#### Installing and packages necessary: 
To run this code you would need the following packages. You can install the packages on jupyter notebook and they can be imported using the command lines below in the code. 
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

### UPDATED
15-11-2022
### DEPLOYMENT 
Windows 11 (21H2) 22000.593 OS Build
### AUTHOR 
Arshiya Anjum Muslim Noor
### ACKNOWLEDGEMENTS
This was a final project on water potability analysis as a part of academics during Data Analytics for Business Decision Making post-graduation course at Durham College in 2022.  

https://github.com/arshii-anjum/Water-Potability-Analysis-Project.git

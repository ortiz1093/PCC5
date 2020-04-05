"""
Python Competency Check 5
IE 6910: Python Machine Learning Applications in IE
Thu Apr  2 17:01:07 2020
Joshua Ortiz
"""

#%% Import packages and Data

# For analysis
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
df = pd.read_csv("Bread Composition.csv")


#%% Statistical Cleanup and Visualization

# Check nan
print(df[df==np.nan].sum())

# Statistical Summary
print(df.describe)

# Tabulate the mean of all features for each quality level
qualityMeans = df.groupby("quality").mean()

# Correlation of each category with quality (sorted according to abs val)
qualityCorr = abs(df.corrwith(df.quality).drop(labels="quality")).sort_values()
print(qualityCorr)

# Capture the top factors from the correlation list
significantFactors = qualityCorr.index[-4:]

# Create histogram of significant factors from main data
df.loc[:,significantFactors].hist(figsize=(15,15))

# Plot significant factors against quality to verify trends exist
qualityMeans.loc[:,significantFactors].plot.line(figsize=(15,15),subplots=True)

# Pairwise plot of chosen factors to see if there is any clear separation by quality
sns.lmplot('quality','volatile acidity',data = df, hue='quality',palette='Set1',fit_reg=False,scatter_kws={"s":70})
sns.lmplot('quality','alcohol',data = df, hue='quality',palette='Set1',fit_reg=False,scatter_kws={"s":70})
sns.lmplot('quality','sulphates',data = df, hue='quality',palette='Set1',fit_reg=False,scatter_kws={"s":70})


#%% Separate Relevant Data

# Trying different factors

# Based on correlation alone (original)
#significantFactors = ['citric acid','sulphates','volatile acidity','alcohol']

# Based on std alone
#significantFactors = ['total sulfur dioxide','residual sugar','free sulfur dioxide','fixed acidity']

# Based on normCorr * normSTD
#significantFactors = ['total sulfur dioxide','alcohol','free sulfur dioxide','fixed acidity']

# Based on avg(normCorr,normSTD)
significantFactors = ['total sulfur dioxide','alcohol','volatile acidity','sulphates']


# Create model data DF
sigData = df.loc[:,significantFactors]

# Check relative proportions of each quality level, we see there is a significant imbalance away from the extremes
print(df.quality.value_counts())

# Specify model inputs
X = sigData.to_numpy();
y = np.array(df.quality);

#%% Initial Model Eval
# Support Vector Machnine

exp=0
svc_model = svm.SVC(C=2**exp,gamma=2**exp,kernel='rbf',decision_function_shape='ovo',random_state=42)
print(cross_val_score(svc_model, X, y,cv=10,scoring='accuracy').mean())
print(scores.mean())

#val_range=2.0**np.array(range(0,6))
#param_grid = dict(C=val_range,gamma=val_range)
#svc_model = svm.SVC(kernel='rbf',decision_function_shape='ovo',random_state=42)
#grid = GridSearchCV(svc_model,param_grid,cv=10,scoring='accuracy')
#grid.fit(X,y)
#print(grid.best_score_)


# Decision Tree
DT = DecisionTreeClassifier(random_state=42)
print(cross_val_score(DT,X,y,cv=10,scoring='accuracy').mean())

# Random Forest
RF = RandomForestClassifier(random_state = 42,n_estimators=100)
print(cross_val_score(RF,X,y,cv=10,scoring='accuracy').mean())

#%% Sensitivity to Training/Testing Proportions

# Initialize arrays for each classifier's accuracy score
SVC_scores = []
DT_scores = []
RF_scores = []
AVG_scores = []

# Test the classifiers' accuracy scores at various test set sizes
sz_range = np.array(range(1,10))/10
for sz in sz_range:
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=sz,stratify=y,random_state=12)
	svc_model.fit(X_train,y_train)
	DT.fit(X_train,y_train)
	RF.fit(X_train,y_train)
	S = svc_model.score(X_test,y_test)
	D = DT.score(X_test,y_test)
	R = RF.score(X_test,y_test)
	SVC_scores.append(S)
	DT_scores.append(D)
	RF_scores.append(R)
	AVG_scores.append(np.mean([S,D,R]))

# Line plots of each model's accuracy score as the size of the test set increases
plt.plot(sz_range,SVC_scores,'b',label='SVC')
plt.plot(sz_range,DT_scores,'r',label='DT')
plt.plot(sz_range,RF_scores,'g',label='RF')
plt.plot(sz_range,AVG_scores,'k',label='AVG')
plt.show()

# Box plots of accuracy scores for each model across the train/test proportions
box_x = ['SVC','DT','RF','AVG']
box_y = [SVC_scores,DT_scores,RF_scores,AVG_scores]
sns.boxplot(x=box_x,y=box_y)

#%%
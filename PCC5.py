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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from numpy.random import random_sample

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

#%%  Read in data
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
fig = plt.figure()
plt.plot(sz_range,SVC_scores,'b',label='SVC')
plt.plot(sz_range,DT_scores,'r',label='DT')
plt.plot(sz_range,RF_scores,'g',label='RF')
plt.plot(sz_range,AVG_scores,'k',label='AVG')
plt.show()
plt.close(fig)

# Box plots of accuracy scores for each model across the train/test proportions
box_x = ['SVC','DT','RF','AVG']
box_y = [SVC_scores,DT_scores,RF_scores,AVG_scores]
sns.boxplot(x=box_x,y=box_y)

#%% Report Findings

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def labeled_confusion(conf_matrix,labels):
	if np.shape(conf_matrix)[0] != np.shape(conf_matrix)[1]:
		print('Matrix not square; no action taken.')
		return conf_matrix
	elif len(labels) != np.shape(conf_matrix)[0]:
		print('Incorrect number of labels; no action taken.')
		return conf_matrix

	pred_labels = [str(label)+'_pred' for label in labels]
	true_labels = [str(label)+'_true' for label in labels]

	lbl_conf = pd.DataFrame(conf_matrix)
	lbl_conf.index = pd.Index(pred_labels)
	lbl_conf.rename(columns=pd.Series(true_labels),inplace=True)
	return lbl_conf

labels = [3,4,5,6,7,8]
# SVC
y_pred = svc_model.predict(X_test)
print('\n\nSupport Vector Classifier Report')
print(classification_report(y_test,y_pred))
print('\n\nSupport Vector Classifier Confusion Matrix')
print(labeled_confusion(confusion_matrix(y_test,y_pred,labels=labels),labels))

# DT
y_pred = DT.predict(X_test)
print('\n\nDecision Trees Classifier Report')
print(classification_report(y_test,y_pred))
print('\n\nDecision Trees Classifier Confusion Matrix')
print(labeled_confusion(confusion_matrix(y_test,y_pred,labels=labels),labels))

# RF
y_pred = RF.predict(X_test)
print('\n\nRandom Forest Classifier Report')
print(classification_report(y_test,y_pred))
print('\n\nRandom Forest Classifier Confusion Matrix')
print(labeled_confusion(confusion_matrix(y_test,y_pred,labels=labels),labels))
print()


#%% Out-of-Sample Predictions

# Create new df
newData = pd.DataFrame()

for factor in significantFactors:
	minVal = min(df[factor])
	maxVal = max(df[factor])
	interval = maxVal - minVal
	newVals = interval*random_sample((10,1)) + minVal
	newData[factor] = newVals.flatten()

X_new = newData.to_numpy()


# Train models on full original dataset
svc_model.fit(X,y)
DT.fit(X,y)
RF.fit(X,y)

# Create predictions for new model
scv_predictions = svc_model.predict(X_new)
DT_predictions = DT.predict(X_new)
RF_predictions = RF.predict(X_new)

# Add predictions to new dataset and report
DataPredictions = newData
models = ['SVC','DT','RF']
DataPredictions['SVC_Predictions']  = scv_predictions
DataPredictions['DT_Predictions']  = DT_predictions
DataPredictions['RF_Predictions']  = RF_predictions
print(DataPredictions)
"""
Python Competency Check 5
IE 6910: Python Machine Learning Applications in IE
Thu Apr  2 17:01:07 2020
Joshua Ortiz
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Bread Composition.csv")

#%% Statistical Cleanup and Visualization

# Check nan
print(df[df==np.nan].sum())

# Statistical Summary
print(df.describe)

# Tabulate quality versus other categories
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

# New DF with only significant factors
significantFactors = significantFactors.append(pd.Index(['quality']))
sigData = df.loc[:,significantFactors]

#%% Support Vector Machine

# Pairwise plot of chosen factors to see if there is any clear separation by quality
sns.lmplot('sulphates','volatile acidity',data = df, hue='quality',palette='Set1',fit_reg=False,scatter_kws={"s":70})
sns.lmplot('sulphates','alcohol',data = df, hue='quality',palette='Set1',fit_reg=False,scatter_kws={"s":70})
sns.lmplot('alcohol','volatile acidity',data = df, hue='quality',palette='Set1',fit_reg=False,scatter_kws={"s":70})

# Specify model inputs
X = sigData.values;
y = np.array(df.quality);
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)


# Fit the SVC model
svc_model = svm.SVC(C=2**-5,kernel='linear',decision_function_shape='ovo')
svc_model.fit(x_train,y_train)
#svc_model.score(x_test, y_test)
scores = cross_val_score(svc_model, X, y, cv=5)
#preds = svc_model.predict(x_test)
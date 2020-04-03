"""
Python Competency Check 5
IE 6910: Python Machine Learning Applications in IE
Thu Apr  2 17:01:07 2020
Joshua Ortiz
"""

import pandas as pd
import numpy as np

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

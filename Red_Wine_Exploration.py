#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:14:22 2018

@author: dominikklepl
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for displaying DataFrames

import matplotlib.pyplot as plt
import seaborn as sns

# Import supplementary visualization code visuals.py from project root folder
import visuals as vs


# Load the Red Wines dataset
data = pd.read_csv("data/winequality-red.csv", sep=';')

#check for missing values
data.isnull().any() #nothing

#check the format of the features
data.info()

#now explore the balance of the groups (quality column)
n_wines = data.shape[0]

# Number of wines with quality rating below 5
quality_below_5 = data.loc[(data['quality'] < 5)]
n_below_5 = quality_below_5.shape[0]

#percentage
percent_below_5 = n_below_5*100/n_wines

# Number of wines with quality rating between 5 to 6
quality_between_5 = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
n_between_5 = quality_between_5.shape[0]
percent_between_5 = n_between_5*100/n_wines

# Number of wines with quality rating above 6
quality_above_6 = data.loc[(data['quality'] > 6)]
n_above_6 = quality_above_6.shape[0]
percent_above_7 = n_above_6*100/n_wines

# Print the results
print("Total number of wine data: {}.".format(n_wines))
print("Wines with rating less than 5: {}".format(n_below_5))
print("which is {:.2f}% of the data".format(percent_below_5))
print("Wines with rating 5 and 6: {}".format(n_between_5))
print("which is {:.2f}% of the data.".format(percent_between_5))
print("Wines with rating 7 and above: {}".format(n_above_6))
print("which is {:.2f}% of the data.".format(percent_above_7)) #the :.2f is rounding to 2 decimals

#Exploring relationships
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde'); #scatterplots between all variables
#+distribution of data

correlation = data.corr()
#display(correlation)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")

#Explore relatioinships in more detail

#Create a new dataframe containing only pH and fixed acidity columns to visualize their co-relations
fixedAcidity_pH = data[['pH', 'fixed acidity']]

#Initialize a joint-grid with the dataframe, using seaborn library
gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, size=6)

#Draws a regression plot in the grid 
gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})

#Draws a distribution plot in the same grid
gridA = gridA.plot_marginals(sns.distplot)

#Later do that for others too

#Relationships between categorical values are nicer in boxplots
volatileAcidity_quality = data[['quality', 'volatile acidity']]
fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='volatile acidity', data=volatileAcidity_quality, ax=axs)
plt.title('quality VS volatile acidity')

plt.tight_layout()
plt.show()
plt.gcf().clear()

#Detection of Outliers 
# For each feature find the data points with extreme high or low values
for feature in data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(data[feature], q=25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(data[feature], q=75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    interquartile_range = Q3 - Q1
    step = 1.5 * interquartile_range
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [243]

# Remove the outliers, if any were specified
good_data = data.drop(data.index[outliers]).reset_index(drop = True)

good_data.to_csv("data/clean_red_data.csv", sep=';')




import numpy as np
import pandas as pd
from sklearn.cross_validation import shufflesplit

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print "Boston housing dataset has {} data points with {} variable each.".format(*data.shape)

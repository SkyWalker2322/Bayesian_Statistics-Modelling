# implementing bayesian statistics on dataset phd-delays.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')

# reading the dataset
df = pd.read_csv('phd-delays.csv', delimiter=';')
df.columns = ['B3_difference_extra','E4_having_child','E21_sex','E22_Age','E22_Age_Squared']

#split the dataset
X = df[['E22_Age','E22_Age_Squared']]
y = df['B3_difference_extra']
X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.2, random_state=42)

# creating a model for y = b_intercept + b_0*age + b_1*age^2 + e
with pm.Model() as model:
    b_intercept = pm.Normal('b_intercept',mu = 0, sigma=1)
    b_0 = pm.Normal('b_0' , mu = 2.5 , sigma=1)
    b_1 = pm.Normal('b_1' , mu = 0, sigma = 1)

    likelihood = pm.Normal('y', mu=b_intercept + b_0*X_train['E22_Age'] + b_1*X_train['E22_Age_Squared'], sigma=1, observed=y_train['B3_difference_extra'])

    # perform inference
    trace = pm.sample(init='adapt_diag')

az.plot_trace(trace)
plt.show()
az.plot_posterior(trace)
plt.show()
az.plot_density(trace)
plt.show()

# perform accuracy test on this test dataset
az.summary(trace) #summarise our trace run with MCMC

# implementing bayesian statistics on dataset phd-delays.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import arviz as az
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# reading the dataset
df = pd.read_csv('phd-delays.csv', delimiter=';')
# the dataset contains the following columns
# B3_difference_extra 
# E4_having_child
# E21_sex
# E22_Age
# E22_Age_Squared
# E4_having_child is a categorical variable

# reading the dataset
# df = pd.read_csv('phd-delays.csv', delimiter=';')
df.columns = ['B3_difference_extra','E4_having_child','E21_sex','E22_Age','E22_Age_Squared']

# create a model on age as a parameter as y = beta0 + beta1 * age + beta2 * age^2 + e
# where e is the error term

# first we create a prior distribution for the parameters
# we use a normal distribution for the parameters

# #  cross validation from sklearn
from sklearn.model_selection import train_test_split
X = df[['E22_Age','E22_Age_Squared']]
y = df['B3_difference_extra']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# create model
with pm.Model() as model:
    # define priors for the parameters beta0, beta1, beta2
    beta0 = pm.Normal('beta0', mu=0, sigma=0.75)
    beta1 = pm.Normal('beta1', mu=0, sigma=0.5)
    beta2 = pm.Normal('beta2', mu=0, sigma=0.25)
    sigma = pm.HalfNormal('sigma', sigma=2.25)
    
    # define likelihood
    likelihood = pm.Normal('y', mu=beta0 + beta1*df['E22_Age'] + beta2*df['E22_Age_Squared'], sigma=sigma, observed=df['B3_difference_extra'])
    
    # inference
    trace = pm.sample(1000, tune=1000, cores=3)
    # trace = pm.sample(1000, tune=1000, cores=2, chains=2)

# plot the trace
# pm.traceplot(trace)
# plt.show()

# plot the posterior distribution
pm.plot_posterior(trace)
plt.plot(beta0, beta1, beta2)
plt.show()

# scatter plot of beta1
plt.scatter(beta2,beta1)
plt.show()


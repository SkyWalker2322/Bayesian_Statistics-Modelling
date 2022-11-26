

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as tt
import theano
import scipy.stats as stats
import scipy.optimize as opt
import arviz as az
import warnings
warnings.filterwarnings('ignore')


# data = pd.read_csv('data.csv')
# data.head()

df=pd.read_csv('phd-delays.csv')
data=df.to_numpy()
X=[]
Y=[]

for i in range(len(data)):
    arr=data[i][0]
    nums=arr.split(';')
    X.append(int(nums[3]))
    Y.append(int(nums[0]))

X=np.array(X)
Y=np.array(Y)


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)
    # define likelihood
    mu = alpha + beta * data['x']
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=data['y'])
    # inference
    trace = pm.sample(1000, tune=1000, cores=1)

pm.traceplot(trace)
plt.show()
pm.plot_posterior(trace)
plt.show()
pm.plot_posterior_predictive_glm(trace, samples=100, label='posterior predictive regression lines')
plt.scatter(data['x'], data['y'], label='observed data', color='C0')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Team-3-Bayesian_Statistics-Modelling

This repository is created as a part of the Statistic Methods in AI Course done as a part of 3-1 in IIIT Hyderabad. It aims to implement Bayesian Modelling and statistics in Python using popular frameworks such as PyMC3 

## Project Team:
  - Pothuri Praneeth Varma
  - Polakampalli Sai Namrath
  - Boyina Vamsi Krishna
  - Thakkallapally Rohan Rao

### Introduction:
Bayesian Statistics is an approach to data analysis and parameter estimation based on Bayes theorem. We discuss the importance of prior and posterior predictive checking, selecting a proper technique for sampling from a posterior distribution,variational inference and variable selection.


Typically, Bayesian workflow consists of three steps:
- Capturing prior knowledge about a given parameter in a statistical model via the prior distribution (before data)
- Determining the likelihood function using the information about the parameters available in the observed data
- Combining both the prior distribution and the likelihood function using Bayes theorem in the form of the posterior distribution.


### Model fitting:
 - The dataset we will be working with in this project is a Ph.D delay data sheet along with some attributes namely: Age, Age^2 , Have Children or not, Sex.
 - We will try to model the delay in Ph.D in months in terms of Age, which is expected to be of the order 2, implying that our expected model is of the form:
	y = β intercept + β age Age + β age2 Age2 + ε , where y is the expected delay,                     β’s are the coefficients of the parameter Age and  ε is the residual.
The following is a visualisation of the dataset provided in this project analysis.

### Modelling Procedure:
- We assume a Normal distribution for all of the β’s and an Inverse Gamma distribution ε, as priors and try to find the posterior distribution
- The MCMC simulation is done using PyMC3, a Python framework for Bayesian analysis and statistics.
- The likelihood is estimated as a Normal distribution (refer to the code for specifics), from which we sample a large chunk of samples, which are then used to perform inference.
- Inference can be explained as performing a large number of simultaneous random walk simulations starting at a random sample from the obtained chunk, from which we repeatedly try to sample the normal parameters, also known as hyperparameters until a convergence is obtained.

### Validating/Assessing the model parameters :
- This can be achieved by performing PPC (Prior and Posterior Predictive Checks)
- We plot the Posterior Predictive distribution and try to compare it with the dataset’s histogram or observed data.
- If they are similar, it verifies that the posterior is likely a close resemblance to the observation set given.

![posterior.png](https://github.com/SkyWalker2322/Team-3-Bayesian_Statistics-Modelling/blob/main/posterior.png)

- Plotting the correlation plots between parameters across various chains. It is important to avoid correlation between parameters to create a robust model.
- This plot is expected to converge to 0 as the iterations increase, implying a decline in correlation.
- Convergence in the MCMC chains is very important to ensure that our outcome of the model is valid. Divergences usually occur when the prior is too flat or sparse, implying the chain starts to diverge trying to explore a funnel/encounter a sudden change in slope produced by the samples given (not always consistent)
- R̂ statistic is an excellent parameter to ensure this convergence. It is defined as the ratio between inter and intra chain variability. It is expected to converge, hence must tend to 1, as the iterations increase.

![Obtained Stats](https://github.com/SkyWalker2322/Team-3-Bayesian_Statistics-Modelling/blob/main/Obtained_stats.jpeg)

The above are the obtained statistics and we can clearly identify that r_hat has converged to 1.

### Results of the Inference:
- Upon plotting various parameters and statistics of the model and its distributions, we obtain some interesting observations that ensure that the model being trained is valid.
- Single Point Parameters can be obtained using Maximum a Posteriori method, which acts similar to MLE from Baye’s theorem discussed in SMAI..
- It tries to find the maximum likelihood parameter set by sampling from the posteriors of each parameter, which usually gives a value close to the mean obtained of the posterior.
- The parameters obtained in our model are:
```{'b_intercept': -35.02104053, '
b_0': 2.07642817, 'b_1': -0.0201467, 
'eps_log__': 1.10557563, 
'sg_log__': 2.45786061, 
'eps': 0.33102028, 
'sg': 11.67979715}
```
- For comparison, upon performing MSE on the dataset with these parameters, we get a slightly better error value than a Quadratic regression performed using sklearn, implying that the behaviour is as expected, and the modelling is fairly effective.

![coefficients.png](https://github.com/SkyWalker2322/Team-3-Bayesian_Statistics-Modelling/blob/main/coefficients.png)

- The above plots are the trace plots obtained after the MCMC simulation performed on 4 chains.
- The right-side plots explain the values of parameters explored at each step of the random walk by the chains for each parameter, hence the hugely populated plot. The x-axis refers to the number of iterations , and y-axis refers to the value of the parameter.
- The dotted lines on the left are the individual posteriors obtained by each of the 4 chains in simulation, while the bold line is the final inference performed, and the posterior obtained as a result of all 4 chains which is essentially an intermediary of all four chains.

### Comparison with the Paper:
- The below three are plots given in the paper, that they have obtained after sampling the posterior for each parameter from MCMC , compared with the prior taken.
- Below them are the graphs obtained from our model, upon performing sampling on the posteriors obtained and plotting it against the prior we have considered. It is quite clear that trends are similar if we compare the plots.
- Although not perfectly the same, which is understandable, as keep in mind - we perform a RANDOM walk as a part of MCMC chain simulation hence the final values can be slightly different, but the trend is fairly similar, establishing the modelling has been a success. 

![plots](https://github.com/SkyWalker2322/Team-3-Bayesian_Statistics-Modelling/blob/main/Plots.jpeg)

- Some important implications of the plots are that there is a clear shortening in the variance of the posterior distributions of the parameters compared to the priors considered, implying there has been significant learning that the model underwent during the fitting.
- Another way to compare the obtained posterior with the given dataset is as mentioned before, comparing the KDE (Kernel Density Estimation) with the Posterior we have obtained, for the model.

![kernel.png](https://github.com/SkyWalker2322/Team-3-Bayesian_Statistics-Modelling/blob/main/kernel.png)

- Below is the expected line fit of regression obtained upon considering the MAP (Maximum a posteriori) parameter sample plotted against the Observed sample.
- Do note, that this regression is not expected to be perfectly fitting of the data, as the data is very inconsistent and is not likely a poly linear regression in the age parameter alone.


![regression.png](https://github.com/SkyWalker2322/Team-3-Bayesian_Statistics-Modelling/blob/main/regression.png)

# riverflow


## Problem Statement

Given Temperature and Precipitation at 9 stations and river flow at one location, predict river flow one week later.

*Predictors/Features*: Temperature anomalies (${^o}C$) and Precipitation anomalies (mm) at 9 stations; river flow ($m^{3}s^{-1}$) at specific location (Norway).

*Predictand/Target*: River Flow ($m^{3}s^{-1}$) at the specific location (Norway) one week later


## Science Background

*Why would we expect to be able to predict river flow with precipitation, temperature, and the previous river flow? How are these variables related?*

* In the absence of new water input, the river flow will decrease slowly over time. 
* River flow has high autocorrelation over some timescale (??)
* Increased precipitation -> increased river flow
* River flow is an "integrated" response in time to precipation
* In regions with snow, temperature near/above freezing leads to melting snow and increased river flow during the melt season (i.e., spring)
* In the non-melt season, higher temperatures could lead to more evaportation and result in less river flow, but this is likely a small impact 

*Other aspects of the data to consider:*

* All datasets could have a climate change related trend and/or changes in variability 
* All data will have a seasonal cycle

## Modeling Approach

I like to look carefully at the data, start with simple models, and then move to more complex models when/if warranted.  I focus on the data first, then the model.

*How do I look at the data?*
* Plot the timeseries and subsets of the timeseries
* Mean
* Variance
* Trend
* Climatology
* Simultaneous Correlations between all features and target

*What types of models do I use?*
* Linear Regression (with and without regularization) 
* Shallow Fully Connected Neural Network: Input(nfeatures)->8->8->Ouput(1)

*How do I test the models to determine if they are trustworthy?*
* Compare train and test
* Look at the prediction


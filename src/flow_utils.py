import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression

import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import optimizers

def lasso(X,Y):

    """ 
    Fit regression model using Lasso (R1) regularization

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model, coefficients, r-squared, and the predicted values of Y

    """
  
    regr = LassoCV(cv=5,max_iter=5000).fit(X,Y)
    Y_pred = regr.predict(X)
    
    r_squared,Y_pred=get_r2(X,Y,regr)
    
    return regr,regr.coef_,r_squared,Y_pred

def ridge(X,Y):

    """ 
    Fit regression model using Ridge (R2) regularization

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model, coefficients, r-squared, and the predicted values of Y

    """

    regr = RidgeCV(cv=5).fit(X,Y)
    Y_pred = regr.predict(X)

    r_squared,Y_pred=get_r2(X,Y,regr)

    return regr,regr.coef_,r_squared,Y_pred

def lr(X,Y):

    """ 
    Fit regression model using standard regression model without regularization

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model, coefficients, r-squared, and the predicted values of Y

    """

    regr = LinearRegression().fit(X,Y)
    Y_pred = regr.predict(X)

    r_squared,Y_pred=get_r2(X,Y,regr)

    return regr,regr.coef_,r_squared,Y_pred

def tomsensomodel_regression(X,Y):

    """ 
    Fit fully connected neural network Input(nfeatures)->8->8->Output(1)

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model

    """
    model = Sequential()

    model.add(Dense(8, input_dim=X.shape[1],activation='tanh',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l1(0.02),
                bias_initializer='he_normal'))

    model.add(Dense(8, activation='tanh',
                kernel_initializer='he_normal',
                bias_initializer='he_normal'))

    model.add(Dense(1,name='output'))

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss ='mean_squared_error',
                  metrics = ['mse'])

    model.fit(X,Y,epochs=250, batch_size=100,verbose=0)

    return(model)

def daily_climo(ds,varname,data_type):

    """ 
    Calculate climatological annual cycle of daily data following Pegion et. al. 2019, SubX BAMS paper

    Args:
    ds : xarray.Dataset of daily data with dimensions (time,lat,lon)
    varname : string containing name of xarray.Dataarray to access in Dataset
    data_type: string indicating target or feature

    Returns:
    Climatological annual cycle of Dataarray

    """
    # Average daily data
    da_day_clim = ds[varname].groupby('time.dayofyear').mean('time')
    da_day_clim = da_day_clim.chunk({'dayofyear': 366})

    # Pad the daily climatolgy with nans

    if (data_type=='target'):
        x = np.empty((366))
        x.fill(np.nan)
        _da = xr.DataArray(x, coords=[da_day_clim.dayofyear],
                           dims = da_day_clim.dims)
    else:
        x = np.empty((366,len(da_day_clim.stations)))
        x.fill(np.nan)

        _da = xr.DataArray(x, coords=[da_day_clim.dayofyear,
                                      da_day_clim.stations],
                           dims = da_day_clim.dims)

    da_day_clim_wnan = da_day_clim.combine_first(_da)

    # Period rolling twice to make it triangular smoothing
    # See https://bit.ly/2H3o0Mf
    da_day_clim_smooth = da_day_clim_wnan.copy()

    for i in range(2):
        # Extand the DataArray to allow rolling to do periodic
        da_day_clim_smooth = xr.concat([da_day_clim_smooth[-15:],
                                        da_day_clim_smooth,
                                        da_day_clim_smooth[:15]],
                                        'dayofyear')
        # Rolling mean
        da_day_clim_smooth = da_day_clim_smooth.rolling(dayofyear=31,
                                                        center=True,
                                                        min_periods=1).mean()
        # Drop the periodic boundaries
        da_day_clim_smooth = da_day_clim_smooth.isel(dayofyear=slice(15, -15))

    # Extract the original days
    da_day_clim_smooth = da_day_clim_smooth.sel(dayofyear=da_day_clim.dayofyear)
    da_day_clim_smooth.name=varname

    return da_day_clim_smooth

def get_r2(X,Y,model):

    """ 
    Calculate r-squared of a for a given model, features, and target

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)
    model : model returned from calls to keras or scikit-learn models

    Returns:
    r-squared value of predicted value of Y given X and target value of Y based on specified model

    """
    pred = model.predict(X).squeeze()
    rsq=np.corrcoef(Y,pred)[0,1]
    return rsq,pred


def standardize(ds):
    """ 
    Standardize the dataset as (X-mu)/sigma

    Args:
    ds : xarray.Dataset with dimensions time, ..., ...

    Returns:
    Standardized xarray.Dataset

    """
    ds_scaled=(ds-ds.mean(dim='time'))/ds.std(dim='time')
    return ds_scaled

def heatmap(X,Y):

    """ 
    Plot the seaborn heatmap of correlations between all values of X and Y
    
    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    Nothing; display plot to screen

    """
    tmp=np.hstack((X,np.expand_dims(Y, axis=1)))
    d = pd.DataFrame(data=tmp)
    corr=d.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(11,11))
    sns.set(font_scale=1)
    ax=sns.heatmap(corr,square=True,linewidths=0.5,fmt=" .2f", \
                   annot=True,mask=mask,cmap='seismic', \
                   vmin=-1,vmax=1,cbar=False,annot_kws={"size": 10})
    
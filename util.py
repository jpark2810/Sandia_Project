import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error

countries = gpd.read_file(
               gpd.datasets.get_path("naturalearth_lowres"))

def rmse(y_true, y_pred, labels,weighted=True):
    if weighted:
        weight = np.cos(labels["lat"].values * np.pi / 180)
    else:
        weight=None
    return mean_squared_error(
        y_true, y_pred, sample_weight=weight,
        squared=False
    )


def rmse_by_month(y_true, y_pred, labels,weighted=True):
    if weighted:
        weight = np.cos(labels["lat"].values * np.pi / 180)
    else:
        weight=None
    return mean_squared_error(
        y_true, y_pred, sample_weight=weight,
        squared=False,multioutput="raw_values"
    )



def plot_mean_errors(y_true, y_pred, labels, title, figsize=(20,6)):
    """
    *y_true* is be a numpy array of shape (6, n)
    *y_pred* is be a numpy array of shape (6, n)
    *labels* is a pandas dataframe containing the latitude 
         and longitude values for each row of *y_true* and *y_pred*
    """
    
    err = (y_pred - y_true).mean(axis=1)

    err_df = labels.copy()
    err_df["err"] = err

    mean_errs = err_df.groupby(["lon", "lat"]).mean().reset_index()
    
    # set up latlongs for pcolormesh
    lons = labels.lon.unique()
    lons.sort()
    
    lats = labels.lat.unique()
    lats.sort()
    
    lats = np.vstack([lats] * len(lons))
    lons = np.vstack([lons] * lats.shape[1]).T
    
    errs = mean_errs.err.values.reshape(lats.shape)
    
    # initialize an axis
    fig, ax = plt.subplots(figsize=figsize)

    
    norm = TwoSlopeNorm(vcenter=0)
    c = plt.pcolormesh(lons, lats, errs,
                       norm=norm,
                   cmap='RdBu_r', 
                   shading='gouraud')
    plt.colorbar(c)
    
    countries.geometry.boundary.plot(
        color=None,edgecolor='gray',linewidth = 2,ax=ax)    
    
    ax.set_ylim(-60, 60)
    ax.set_title(title)
    
    return(err_df, mean_errs)


def animate_predictions(y_true, y_pred, labels, filename,
                        lags=list(range(1,7))):
    """
    y_true: numpy array of shape (n x 6) containing the true
            temperature values of the horizon in chronological
            order.  Column 0 is 1 month after the end of the 
            context, column 1 is 2 months after, etc.
    y_true: numpy array of shape (n x 6) containing the predicted
            temperature values of the horizon in chronological
            order.  Column 0 is 1 month after the end of the 
            context, column 1 is 2 months after, etc.
    labels: pd.DataFrame with columns ["lon", "lat", "date"].  
                NOTE: Each row of labels must match y_true and y_pred
    lags: Time lags between context and prediction to use in animation.
                With a six-month horizon possible lags are 1 through 6.
    filename: must be ".mov", ".avi" or ".mp4"
    """
    # set up the information required for animation
    month_offset = labels.groupby("date").lon.count().iloc[0]
    dates = labels.date.unique()
    horizon = y_true.shape[1]
    
    # save all predictions in a new DataFrame
    # lag columns must be realigned so that the
    # predictions for a fixed date are in the same
    # column, rather than the predictions from a
    # fixed context.
    tmp = labels.copy()
    tmp["T"] = y_true[:, -1]
    
    keys = []
    for lag in lags:
        key = f"pred_lag_{lag}"
        keys.append(key)
        
        # columns indices are off by 1
        # since column 0 has a 1-month lag
        # and column 5 has a 6-month lag
        tmp[key] = y_pred[:, lag-1]
        
        # The six-month lag prediction is aligned
        # with "T" in the initial DataFrame.  The five-month
        # lag is one month back, the four-month lag is
        # two months back, etc.
        tmp[key] = tmp[key].shift((lag-horizon) * month_offset)
        
    # set up values and latlongs for pcolormesh
    # insure that all values are in the same order.
    tmp.sort_values(["date", "lon", "lat"], inplace=True)

    lons = labels.lon.unique()
    lons.sort()
    
    lats = labels.lat.unique()
    lats.sort()
    
    lats = np.vstack([lats] * len(lons))
    lons = np.vstack([lons] * lats.shape[1]).T
    
    # setup plot for the animation
    n_axes = len(lags) + 1
    fig, ax = plt.subplots(n_axes, 1, figsize=(20,6 * n_axes))
    
    # starting data for first frame
    T = tmp[tmp.date == dates[0]]["T"].values.reshape(lons.shape)
    
    lag_preds = [
        tmp[tmp.date == dates[0]][key].values.reshape(lons.shape)
        for key in keys
    ]
    
    # compute min and max values for colorbars
    # so they remain consistent across all frames
    vmin = np.nanmin(tmp[tmp.columns[3:]].values)
    vmax = np.nanmax(tmp[tmp.columns[3:]].values)
    
    # initialize colormaps for first frame
    cmesh_T = ax[0].pcolormesh(lons, lats, T, vmin=vmin, 
                       vmax=vmax, cmap='RdBu_r', shading='gouraud')
    
    # use ax[i + 1] since ax[0] contains the true temperatures.
    cmesh_lags = [
        ax[i+1].pcolormesh(lons, lats, lag_preds[i], 
                           vmin=vmin, vmax=vmax, cmap='RdBu_r', 
                           shading='gouraud')
        for i in range(len(lags))
    ]
    
    # plot outlines of countries on all subplots
    for i in range(n_axes):
        countries.geometry.boundary.plot(
                        color=None,edgecolor='gray',linewidth = 2,ax=ax[i])  
        ax[i].set_ylim(-60, 60)
        fig.colorbar(cmesh_T, ax=ax[i])
        
    # This function creates the ith frame
    # of the animation
    def animate(i):
        filtered = tmp[tmp.date == dates[i]]
        
        # update true temperature colormesh
        T = filtered["T"].values.reshape(lons.shape)
        cmesh_T.set_array(T)
        ax[0].set_title(f"Observed temperatures: {dates[i]}")
        
        # update predicted temperature colormeshes
        for j, key in enumerate(keys):
            pred = filtered[key].values.reshape(lons.shape)
            cmesh_lags[j].set_array(pred)
            # index is off by 1 since first subplot
            # contains the observed temperatures
            ax[j + 1].set_title(f"{key}: {dates[i]}")

    # write animation
    anim = animation.FuncAnimation(fig, animate, interval=500,
                                   frames=len(dates))
    writervideo = animation.FFMpegWriter(fps=2) 
    anim.save(filename, writer=writervideo)
    
    

def animate_residuals(y_true, y_pred, labels, filename,
                        lags=list(range(1,7))):
    """
    y_true: numpy array of shape (n x 6) containing the true
            temperature values of the horizon in chronological
            order.  Column 0 is 1 month after the end of the 
            context, column 1 is 2 months after, etc.
    y_true: numpy array of shape (n x 6) containing the predicted
            temperature values of the horizon in chronological
            order.  Column 0 is 1 month after the end of the 
            context, column 1 is 2 months after, etc.
    labels: pd.DataFrame with columns ["lon", "lat", "date"].  
                NOTE: Each row of labels must match y_true and y_pred
    lags: Time lags between context and prediction to use in animation.
                With a six-month horizon possible lags are 1 through 6.
    filename: must be ".mov", ".avi" or ".mp4"
    """
    # set up the information required for animation
    month_offset = labels.groupby("date").lon.count().iloc[0]
    dates = labels.date.unique()
    horizon = y_true.shape[1]
    
    # save all predictions in a new DataFrame
    # lag columns must be realigned so that the
    # predictions for a fixed date are in the same
    # column, rather than the predictions from a
    # fixed context.
    tmp = labels.copy()
    tmp["T"] = y_true[:, -1]
    
    keys = []
    for lag in lags:
        key = f"pred_lag_{lag}"
        keys.append(key)
        
        # columns indices are off by 1
        # since column 0 has a 1-month lag
        # and column 5 has a 6-month lag
        tmp[key] = y_pred[:, lag-1]
        
        # The six-month lag prediction is aligned
        # with "T" in the initial DataFrame.  The five-month
        # lag is one month back, the four-month lag is
        # two months back, etc.
        tmp[key] = tmp[key].shift((lag - horizon) * month_offset)
        
    # set up values and latlongs for pcolormesh
    # insure that all values are in the same order.
    tmp.sort_values(["date", "lon", "lat"], inplace=True)

    lons = labels.lon.unique()
    lons.sort()
    
    lats = labels.lat.unique()
    lats.sort()
    
    lats = np.vstack([lats] * len(lons))
    lons = np.vstack([lons] * lats.shape[1]).T
    
    # setup plot for the animation
    n_axes = len(lags)
    fig, ax = plt.subplots(n_axes, 1, figsize=(20,6 * n_axes))
    
    # starting data for first frame
    T = tmp[tmp.date == dates[0]]["T"].values.reshape(lons.shape)
    
    lag_preds = [
        tmp[tmp.date == dates[0]][key].values.reshape(lons.shape)
        for key in keys
    ]
    
    # compute min and max values for colorbars
    # so they remain consistent across all frames
    pred_values = tmp[tmp.columns[4:]].values
    true_values = np.vstack([tmp["T"]] * len(lags)).T
    all_residuals = (pred_values - true_values) / pred_values
    vmin = np.nanmin(all_residuals)
    vmax = np.nanmax(all_residuals)
    
    # setup the initial pcolormesh
    cmesh_lags = [
        ax[i].pcolormesh(lons, lats, lag_preds[i] - T, 
                           vmin=vmin, vmax=vmax, cmap='RdBu_r', 
                           shading='gouraud')
        for i in range(len(lags))
    ]
    
    # plot outlines of countries on all subplots
    for i in range(n_axes):
        countries.geometry.boundary.plot(
                        color=None,edgecolor='gray',linewidth = 2,ax=ax[i])  
        ax[i].set_ylim(-60, 60)
        fig.colorbar(cmesh_lags[0], ax=ax[i])
        
    # This function creates the ith frame
    # of the animation
    def animate(i):
        filtered = tmp[tmp.date == dates[i]]

        # update true temperature colormesh
        T = filtered["T"].values.reshape(lons.shape)
        
        # update predicted temperature colormeshes
        for j, key in enumerate(keys):
            pred = filtered[key].values.reshape(lons.shape)
            cmesh_lags[j].set_array((pred - T) / pred)
            # index is off by 1 since first subplot
            # contains the observed temperatures
            ax[j].set_title(f"{key} residuals: {dates[i]}")

    # write animation
    anim = animation.FuncAnimation(fig, animate, interval=500,
                                   frames=len(dates))
    writervideo = animation.FFMpegWriter(fps=2) 
    anim.save(filename, writer=writervideo)